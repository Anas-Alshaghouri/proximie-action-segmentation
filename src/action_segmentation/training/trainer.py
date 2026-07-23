from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from action_segmentation.config import AppConfig
from action_segmentation.data.dataset import create_dataloaders
from action_segmentation.data.ingestion import ingest_precomputed_feature_batch
from action_segmentation.models.fusion import build_fusion_layer
from action_segmentation.models.temporal_tcn import (
    CausalTemporalConvNet,
    build_temporal_model,
)
from action_segmentation.training.losses import masked_cross_entropy

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingResult:
    device: str
    epochs_completed: int
    best_epoch: int
    best_validation_loss: float
    checkpoint_path: Path
    history: tuple[EpochMetrics, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "epochs_completed": self.epochs_completed,
            "best_epoch": self.best_epoch,
            "best_validation_loss": round(self.best_validation_loss, 6),
            "checkpoint_path": str(self.checkpoint_path),
            "history": [item.to_dict() for item in self.history],
        }


def resolve_device(requested: str) -> torch.device:
    """Resolve `auto`, `cpu`, or a concrete CUDA device string."""
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device '{requested}' was requested, but CUDA is unavailable."
            )
        device = torch.device(normalized)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index {device.index} is unavailable; "
                f"found {torch.cuda.device_count()} device(s)."
            )
        return device
    raise ValueError("training.device must be 'auto', 'cpu', or a CUDA device.")


def _run_epoch(
    *,
    model: CausalTemporalConvNet,
    fusion_layer: nn.Module,
    loader: Any,
    config: AppConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_valid_timestamps = 0

    for raw_batch in loader:
        ingested = ingest_precomputed_feature_batch(raw_batch, config)
        features = ingested.features.to(device)
        labels = ingested.labels.to(device)
        view_mask = ingested.view_mask.to(device)

        with torch.set_grad_enabled(training):
            fused = fusion_layer(features, view_mask)
            output = model(fused.fused_features, fused.time_mask)
            loss = masked_cross_entropy(output.logits, labels, output.time_mask)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        valid_count = int(output.time_mask.sum().item())
        total_loss += float(loss.detach().item()) * valid_count
        total_valid_timestamps += valid_count

    if total_valid_timestamps == 0:
        raise RuntimeError("The epoch contained no valid camera timestamps.")
    return total_loss / total_valid_timestamps


def save_checkpoint(
    *,
    path: Path,
    model: CausalTemporalConvNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    history: list[EpochMetrics],
    config: AppConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "epoch": epoch,
            "validation_loss": validation_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_history": [item.to_dict() for item in history],
            "model_contract": {
                "type": config.model.type,
                "input_dim": config.model.input_dim,
                "hidden_dim": config.model.hidden_dim,
                "num_classes": config.model.num_classes,
                "kernel_size": config.model.kernel_size,
                "dilations": list(config.model.dilations),
                "phase_names": list(config.phases.names),
            },
        },
        path,
    )


def load_model_checkpoint(
    *,
    checkpoint_path: str | Path,
    config: AppConfig,
    device: torch.device,
) -> tuple[CausalTemporalConvNet, dict[str, Any]]:
    """Build the configured model and restore a saved checkpoint."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint format: {path}")

    contract = checkpoint.get("model_contract", {})
    expected = {
        "type": config.model.type,
        "input_dim": config.model.input_dim,
        "hidden_dim": config.model.hidden_dim,
        "num_classes": config.model.num_classes,
        "kernel_size": config.model.kernel_size,
        "dilations": list(config.model.dilations),
        "phase_names": list(config.phases.names),
    }
    for key, expected_value in expected.items():
        if key in contract and contract[key] != expected_value:
            raise ValueError(
                f"Checkpoint/config mismatch for '{key}': "
                f"checkpoint={contract[key]!r}, config={expected_value!r}."
            )

    model = build_temporal_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def train_model(
    config: AppConfig,
    *,
    checkpoint_path: str | Path | None = None,
) -> TrainingResult:
    """Train the causal TCN and retain the lowest-validation-loss checkpoint."""
    device = resolve_device(config.training.device)
    loaders = create_dataloaders(config)
    fusion_layer = build_fusion_layer(config).to(device)
    model = build_temporal_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    if checkpoint_path is None:
        checkpoint_path = (
            config.output.artifacts_directory / config.output.checkpoint_filename
        )
    checkpoint = Path(checkpoint_path).expanduser().resolve()

    history: list[EpochMetrics] = []
    best_validation_loss = float("inf")
    best_epoch = 0

    LOGGER.info("Training on device: %s", device)
    for epoch in range(1, config.training.epochs + 1):
        train_loss = _run_epoch(
            model=model,
            fusion_layer=fusion_layer,
            loader=loaders["train"],
            config=config,
            device=device,
            optimizer=optimizer,
        )
        with torch.no_grad():
            validation_loss = _run_epoch(
                model=model,
                fusion_layer=fusion_layer,
                loader=loaders["validation"],
                config=config,
                device=device,
                optimizer=None,
            )

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=round(train_loss, 6),
            validation_loss=round(validation_loss, 6),
        )
        history.append(metrics)
        LOGGER.info(
            "Epoch %02d/%02d | train_loss=%.6f | validation_loss=%.6f",
            epoch,
            config.training.epochs,
            train_loss,
            validation_loss,
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            save_checkpoint(
                path=checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=validation_loss,
                history=history,
                config=config,
            )
            LOGGER.info("Saved new best checkpoint: %s", checkpoint)

    if best_epoch == 0:
        raise RuntimeError("Training completed without creating a checkpoint.")

    return TrainingResult(
        device=str(device),
        epochs_completed=config.training.epochs,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        checkpoint_path=checkpoint,
        history=tuple(history),
    )
