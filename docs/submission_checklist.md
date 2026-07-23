# Submission Checklist

## Required deliverables

- [x] Structured Python repository
- [x] Standalone mock feature pipeline
- [x] Temporal classification layer
- [x] Segmentation timeline generator
- [x] Model and product metric stack
- [x] Patient Present and Operation error analysis
- [x] Technical Architecture Report in PDF format
- [x] AWS architecture diagram
- [x] Aligned timeline visualization and reusable plotting command

## Reproduction commands

```bash
pip install -r requirements.txt
pip install -e .
pytest
python scripts/train_mock.py --config configs/default.yaml
python scripts/evaluate_mock.py --config configs/default.yaml
python scripts/visualize_timeline.py --metrics artifacts/metrics.json --sample-index 0 --show-confidence
```

## Generated files

```text
artifacts/model.pt
artifacts/metrics.json
artifacts/timeline_<sample_id>.png
```

## Repository hygiene

- [x] No generated checkpoints committed
- [x] No test caches committed
- [x] No local virtual environment committed
- [x] No hard-coded machine paths
- [x] Configuration values stored in YAML
- [x] Random seeds configured
- [x] Report, architecture, and example timeline stored under `docs/`
