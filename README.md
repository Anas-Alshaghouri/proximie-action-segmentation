# Proximie Action Segmentation Prototype

A lightweight, reproducible prototype for live multi-view temporal action segmentation in an operating room.

The repository demonstrates the complete path from synchronized precomputed camera features to a clean workflow timeline. It includes training, checkpointing, evaluation, product-facing metrics, error analysis, an AWS architecture design, and the required technical report.

## Deliverables

- Structured Python repository with a working end-to-end pipeline
- Technical Architecture Report, 5 pages: [`docs/technical_architecture_report.pdf`](docs/technical_architecture_report.pdf)
- Editable report source: [`docs/technical_architecture_report.docx`](docs/technical_architecture_report.docx)
- AWS architecture explanation: [`docs/aws_architecture.md`](docs/aws_architecture.md)
- AWS architecture diagram: [`docs/aws_architecture.png`](docs/aws_architecture.png)
- Example aligned workflow timeline: [`docs/sample_timeline.png`](docs/sample_timeline.png)

## What the prototype includes

- Deterministic synthetic operating-room workflow sequences
- One to three simulated camera feature streams
- A validated precomputed-feature ingestion contract
- Mask-aware mean fusion across available cameras
- A lightweight causal Temporal Convolutional Network
- Masked loss for timestamps with valid camera evidence
- Best-validation checkpointing with contract validation
- Causal probability smoothing and phase confirmation
- Frame, segment, boundary, and product metrics
- Validation-wide mock error analysis with cause attribution
- End-to-end training, evaluation, timeline generation, and JSON export
- PNG visualization of ground truth, raw output, cleaned output, camera availability, and confidence
- Automated tests for causality, masks, metrics, checkpoints, and reproducibility

## Prototype assumptions

- Phases: `empty`, `patient_present`, `preparation`, `operation`, `closing`
- Feature rate: 1 Hz
- Sequence duration: 600 seconds
- Camera views: 1 to 3
- Feature dimension: 64
- Fusion: masked mean across valid cameras
- Temporal model: causal TCN with dilations `[1, 2, 4, 8]`
- Receptive field: 61 seconds
- Timeline smoothing: causal 5-second probability average
- Phase confirmation: 8 consecutive seconds before switching
- Raw video decoding and visual-backbone extraction are outside the prototype scope

## Quick start

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

### 1. Run the test suite

```bash
pytest
```

Expected result:

```text
57 passed
```

### 2. Train the model

```bash
python scripts/train_mock.py \
    --config configs/default.yaml
```

The command:

1. Generates deterministic training and validation sequences.
2. Validates the multi-view feature contract.
3. Fuses only available camera views.
4. Trains the causal TCN with masked cross-entropy.
5. Measures validation loss after every epoch.
6. Saves the checkpoint with the lowest validation loss.

Default checkpoint:

```text
artifacts/model.pt
```

### 3. Evaluate the trained checkpoint

```bash
python scripts/evaluate_mock.py \
    --config configs/default.yaml \
    --checkpoint artifacts/model.pt \
    --split test \
    --output artifacts/metrics.json
```

After training, the shorter command also works:

```bash
python scripts/evaluate_mock.py \
    --config configs/default.yaml
```

The generated `artifacts/metrics.json` contains:

- Checkpoint metadata
- Test loss
- Raw and cleaned frame metrics
- Segmental F1 at 10, 25, and 50 percent overlap
- Mean edit score
- Boundary errors
- Patient Present fragmentation and delay metrics
- Operation coverage, delay, and occlusion metrics
- Ground-truth, raw, and cleaned timelines for every evaluated sequence
- Camera availability intervals and maximum-confidence traces

### 4. Visualize an evaluated timeline

```bash
python scripts/visualize_timeline.py \
    --metrics artifacts/metrics.json \
    --sample-id test_0011 \
    --output artifacts/timeline_test_0011.png \
    --show-confidence
```

The aligned figure shows:

- Ground-truth phase segments
- Raw model predictions
- Cleaned predictions after causal post-processing
- Availability for each camera slot
- Raw and smoothed maximum-confidence traces

List all available sample IDs with:

```bash
python scripts/visualize_timeline.py \
    --metrics artifacts/metrics.json \
    --list-samples
```

The installed command is also available:

```bash
proximie-visualize \
    --metrics artifacts/metrics.json \
    --sample-id test_0011 \
    --show-confidence
```

A checked-in example is available at [`docs/sample_timeline.png`](docs/sample_timeline.png).

### 5. Validate the architecture without training

```bash
python scripts/validate_pipeline.py \
    --config configs/default.yaml
```

This command validates synthetic data, ingestion, fusion, model causality, post-processing, metric behavior, and mock error analysis without requiring a trained checkpoint.

### 6. Inspect generated data

```bash
python scripts/inspect_mock_data.py \
    --config configs/default.yaml \
    --split validation \
    --sample-index 0
```

This shows tensor shapes, phase timelines, camera masks, missing intervals, and selected feature values.

### 7. Run the dedicated mock error analysis

```bash
python scripts/analyze_mock_errors.py \
    --config configs/default.yaml \
    --output artifacts/mock_error_analysis.json
```

The routine injects known Patient Present disturbances and Operation occlusions so each failure can be linked to a concrete cause.

## End-to-end data flow

```text
Synthetic multi-camera features
        ↓
Ingestion and tensor validation
        ↓
Masked multi-view fusion
        ↓
Causal TCN
        ↓
Per-second phase logits
        ↓
Causal smoothing and phase confirmation
        ↓
Clean contiguous workflow timeline
        ↓
Model and product metrics
```

## Model design

### Input and output

```text
Input:  [batch, time, 64]
Output: [batch, time, 5]
```

The five output scores correspond to the five workflow phases.

### Causal temporal context

The model contains four residual temporal blocks with dilations:

```text
1 → 2 → 4 → 8
```

Each block contains two causal convolutions. A prediction at second `t` can use second `t` and previous seconds. It cannot read future features.

The receptive field is:

```text
1 + 2 × (3 - 1) × (1 + 2 + 4 + 8) = 61 steps
```

At 1 Hz, one prediction can use up to 61 seconds of history.

### Missing evidence

The pipeline uses two masks:

- `view_mask` identifies which camera is valid at each timestamp.
- `time_mask` identifies whether at least one camera is valid.

If every camera is unavailable, the timestamp is excluded from loss and metrics. The timeline uses label `-1`, shown as `unavailable`, rather than incorrectly treating missing evidence as `empty`.

## Timeline post-processing

The timeline generator applies:

1. A causal 5-second probability average
2. An 8-second confirmation rule before switching phases

This removes brief false phase islands and improves timeline continuity. It also delays phase boundaries. Evaluation reports raw and cleaned predictions separately so the stability and latency trade-off remains visible.

## Verified reference run

The following values came from the verified default CUDA run on the held-out synthetic test split:

```text
Best epoch:                         9
Best validation loss:               0.005698
Test cross-entropy loss:             0.005552
Raw frame accuracy:                  99.70%
Raw macro F1:                        99.67%
Raw segmental F1@50:                 97.12%
Raw mean edit score:                 95.09
Cleaned mean edit score:             100.00
Cleaned segmental F1@50:             96.04%
Raw Patient Present extra fragments: 3
Cleaned Patient Present fragments:   0
Mean cleaned boundary error:         5.86 seconds
```

These results confirm that the pipeline is reproducible and learnable on the deliberately simple synthetic feature distribution. They are not claims about real surgical-video performance.

## Evaluation layers

### Frame metrics

- Accuracy
- Per-class precision, recall, and F1
- Macro F1
- Confusion matrix

### Segment metrics

- Segmental F1@10, F1@25, and F1@50
- Edit score
- Start and end boundary errors
- Boundary tolerance rate

### Product metrics

Patient Present:

- Extra fragments
- False-positive duration
- Missed duration
- State switches per hour
- Confirmation delay

Operation:

- Start and end delay
- Coverage
- Coverage during occlusion
- Missed duration
- Recovery after occlusion

## AWS production architecture

The production design is documented in:

- [`docs/aws_architecture.md`](docs/aws_architecture.md)
- [`docs/aws_architecture.svg`](docs/aws_architecture.svg)
- [`docs/aws_architecture.png`](docs/aws_architecture.png)

The selected design uses:

- Kinesis Video Streams for governed video ingestion
- EKS CPU workers for synchronization
- EKS GPU workers for visual feature extraction
- Kinesis Data Streams for room-keyed feature events
- ElastiCache for rolling temporal state
- DynamoDB for durable room and product state
- S3 lifecycle rules for approved distilled hard-example packages
- EventBridge for confirmed phase events
- CloudWatch for stream, latency, model, and infrastructure monitoring

Raw live video is not duplicated into a permanent S3 data lake.

## Repository structure

```text
configs/                                  Runtime configuration
scripts/train_mock.py                     End-to-end training
scripts/evaluate_mock.py                  Trained-checkpoint evaluation
scripts/validate_pipeline.py              Structural pipeline validation
scripts/analyze_mock_errors.py            Cause-attributed mock error analysis
scripts/inspect_mock_data.py              Synthetic data inspection
scripts/visualize_timeline.py             Aligned timeline PNG generation
src/action_segmentation/data/             Data generation and ingestion
src/action_segmentation/models/           Multi-view fusion and causal TCN
src/action_segmentation/training/         Loss, training, and checkpoints
src/action_segmentation/postprocessing/   Causal timeline generation
src/action_segmentation/evaluation/       Metrics, evaluation, and error analysis
src/action_segmentation/visualization/    Timeline plotting and report parsing
artifacts/                                Generated checkpoints and reports
tests/                                    Automated tests
docs/                                     Technical report and AWS architecture
```

## Main limitations

- The data is synthetic and uses precomputed features.
- The five-phase order is simplified.
- Sequences have a fixed length.
- All valid camera views receive equal fusion weight.
- One global post-processing rule is used for all phases.
- The visual backbone and real AWS deployment are design-only components.

The technical report explains how I would extend this baseline for real multi-site surgical data, privacy-aware hard-example collection, active learning, stronger multi-view fusion, class-specific boundary handling, and cost-controlled AWS deployment.
