# Audio Classification Pipeline — Development Notes

> **Status**: Active development — MVP phase
> **Last updated**: 2026-03-21
> **Stage**: Config loader implemented. Starting Stage 1 (Data Preparation).

---

## 1. Project Goal

Build a production-level MLOps pipeline for **AI-generated audio detection**.
Binary classification: **human-made audio** vs **AI-generated audio**.

### Two target models (same pipeline, different configs):

| Model | Use case | Input characteristics |
|-------|----------|-----------------------|
| **Music Detection** | Classify full songs as human or AI-generated | Full-length tracks, offline analysis |
| **Broadcast Monitoring** | Detect AI-generated music in broadcast streams | Shorter segments, potentially lower latency requirements |

Both models share the same pipeline, codebase, and architecture — only the
configs differ (dataset, feature params, possibly lighter model for broadcast).

---

## 2. Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| Language | Python | |
| ML Framework | TensorFlow / Keras | Torch support planned but not in MVP |
| Containerization | Docker + Docker Compose | Single Dockerfile for MVP |
| Orchestration | Makefile | CLI entrypoints per stage |
| Configuration | YAML (composed) | Experiment references dataset/features/model configs |
| Experiment Tracking | MLflow | Self-hosted via docker-compose |
| Audio Processing | librosa, soundfile | Framework-agnostic feature extraction |
| Testing | pytest | Unit tests first, integration tests deferred |
| Linting | ruff, mypy | |

---

## 3. Pipeline Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    EXPERIMENT CONFIG                         │
│              (points to dataset/features/model)              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: DATA PREPARATION                                  │
│                                                             │
│  Input:  Raw audio dirs (real + fake)                       │
│  Steps:  1. Scan & validate audio files                     │
│          2. Stratified train/val/test split                  │
│          3. Export manifest JSON                             │
│  Output: data/manifests/{dataset_name}_split.json           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: FEATURE EXTRACTION                                │
│                                                             │
│  Input:  Manifest + feature config                          │
│  Steps:  1. Load audio per manifest entry                   │
│          2. Preprocess (mono, resample, hf cut)             │
│          3. Compute STFT spectrogram (full length)          │
│          4. Normalize                                       │
│          5. Save one .npy per source file (full spectro)    │
│          6. Export feature manifest                          │
│  Output: data/features/{feature_name}/                      │
│          ├── train/ val/ test/                               │
│          │   └── {filename}.npy  (n_freq, n_frames)         │
│          └── feature_manifest.json                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: TRAINING                                          │
│                                                             │
│  Input:  Feature manifest + model config + training params  │
│  Steps:  1. Build tf.data pipeline from .npy files          │
│          2. Random crop windows (0.78s) on-the-fly          │
│          3. Apply augmentation (if enabled)                  │
│          4. Build model from config                         │
│          5. Train with callbacks                            │
│          6. Log params/metrics/artifacts to MLflow          │
│  Output: MLflow run with model checkpoint + metrics         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4: EVALUATION                                        │
│                                                             │
│  Input:  Trained model + test feature split                 │
│  Steps:  1. Load best checkpoint                            │
│          2. Deterministic strided crops over test files      │
│          3. Per-window predictions → per-file aggregation   │
│          4. Compute metrics (F1, AUC-ROC, EER)             │
│          5. Generate plots (confusion matrix, DET, ROC)     │
│          6. Log to MLflow                                   │
│  Output: MLflow artifacts (plots, reports, metrics)         │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

**Data prep is separated from feature extraction** because:
- Splitting is done once per dataset version
- Feature extraction runs multiple times (different feature configs)
- Features can be cached and reused across experiments

**Full spectrograms are stored, not pre-cut windows** because:
- 0.78s windows are very short — pre-cutting would create massive file counts
- Random cropping at training time provides more augmentation variety
- Evaluation uses deterministic strided windows for reproducibility
- Source-level prediction aggregation requires knowing which file each
  window came from

**Augmentation happens at training time (Stage 3)** because:
- Baking augmentation into .npy files means every epoch sees the same data
- On-the-fly augmentation via tf.data gives different variations per epoch

---

## 4. Config Architecture

### Composed config strategy

The experiment config references sub-configs rather than inlining everything.
This avoids duplication when mixing datasets, features, and models.

```text
configs/experiments/exp_001.yaml
  ├── references → configs/datasets/full_songs_v1.yaml
  ├── references → configs/features/stft_db_44k.yaml
  ├── references → configs/models/spec_cnn_v1.yaml
  └── owns: training params, augmentation, evaluation settings
```

### What each config owns

| Config | Responsibility |
|--------|---------------|
| **dataset** | Paths to audio dirs, file format, split strategy/ratios/seed |
| **features** | Sample rate, mono, hf cut, STFT params, normalization, storage format |
| **model** | Architecture type, backbone (filters/kernels/strides), head, output task |
| **experiment** | References above three + training hyperparams, crop/window config, augmentation, evaluation metrics/plots |

### Config files (MVP)

```text
configs/
├── datasets/
│   └── full_songs_v1.yaml
├── features/
│   └── stft_db_44k.yaml
├── models/
│   └── spec_cnn_v1.yaml
└── experiments/
    └── exp_001.yaml
```

<details>
<summary><strong>configs/datasets/full_songs_v1.yaml</strong></summary>

```yaml
dataset:
  name: full_songs_v1
  description: "FMA small vs AI-rebuilt FMA small"

  paths:
    real: /datasets/fma_small
    fake: /datasets/fma_rebuilt_small

  format: mp3

  split:
    strategy: stratified
    ratios: [0.70, 0.10, 0.20]   # train / val / test
    seed: 123
```

</details>

<details>
<summary><strong>configs/features/stft_db_44k.yaml</strong></summary>

```yaml
features:
  name: stft_db_44k
  type: stft_magnitude_db

  audio:
    sample_rate: 44100
    mono: true
    hf_cut_hz: 16000

  stft:
    n_fft: 2048
    win_length: 2048
    hop_length: 512

  normalization:
    strategy: fixed
    mean: -4.0
    std: 3.0

  storage: full
  output_format: npy
```

</details>

<details>
<summary><strong>configs/models/spec_cnn_v1.yaml</strong></summary>

```yaml
model:
  name: spec_cnn_v1
  type: spectrogram_cnn

  backbone:
    filters:  [16, 32, 64, 128, 256, 512]
    kernels:  [3, 3, 3, 3, 3, 3]
    strides:  [2, 2, 2, 2, 2, 2]
    use_maxpool: true
    activation: relu
    batch_norm: true
    pooling: average

  head:
    dense_units: 128
    activation: swish
    dropout: 0.2

  output:
    type: binary
    activation: sigmoid
    loss: binary_crossentropy
    threshold: 0.5
```

</details>

<details>
<summary><strong>configs/experiments/exp_001.yaml</strong></summary>

```yaml
experiment:
  name: specnn_amplitude_baseline
  description: "Baseline 2D CNN on STFT dB, FMA small dataset"
  tags:
    task: full_song_detection

dataset: configs/datasets/full_songs_v1.yaml
features: configs/features/stft_db_44k.yaml
model: configs/models/spec_cnn_v1.yaml

training:
  epochs: 100
  batch_size: 50
  learning_rate: 0.001
  optimizer: adam
  seed: 123

  crop:
    window_sec: 0.78
    strategy: random
    samples_per_file: 5

  shuffle_buffer_multiplier: 2

  callbacks:
    early_stopping:
      monitor: val_loss
      patience: 15
    reduce_lr:
      monitor: val_loss
      factor: 0.5
      patience: 5

  augmentation:
    enabled: false

evaluation:
  metrics: [accuracy, f1, auc_roc, eer]
  plots: [confusion_matrix, det_curve, roc_curve]
  threshold_sweep:
    enabled: true
    range: [0.1, 0.9]
    step: 0.05
```

</details>

### Model config: structured backbone/head/output vs layer-by-layer

We chose the **structured** approach (backbone defines a repeating pattern)
over layer-by-layer YAML (which would essentially recreate Keras in YAML).

The `model.type` field maps to a Python class. Different model types can have
completely different config shapes:

```yaml
# Current: spectrogram_cnn uses backbone/head/output
model:
  type: spectrogram_cnn
  backbone: { ... }

# Future: a transformer would have its own structure
model:
  type: audio_transformer
  num_heads: 8
  embed_dim: 256
```

### Output config supports future expansion

```yaml
# MVP: binary
output:
  type: binary
  activation: sigmoid
  loss: binary_crossentropy

# Future: multiclass
output:
  type: multiclass
  num_classes: 10
  activation: softmax
  loss: categorical_crossentropy

# Future: multi-head (deferred)
output:
  type: multi_head
  heads:
    deepfake: { type: binary, activation: sigmoid, loss: binary_crossentropy, weight: 1.0 }
    encoder:  { type: multiclass, num_classes: 10, activation: softmax, weight: 0.5 }
```

---

## 5. Feature Storage Strategy

One `.npy` file per source audio file containing the **full spectrogram**:

```text
data/features/stft_db_44k/
├── train/
│   ├── song_001.npy    # shape: (n_freq_bins, n_frames) — variable n_frames
│   ├── song_002.npy
│   └── ...
├── val/
├── test/
└── feature_manifest.json
```

**Feature manifest structure** (produced by Stage 2):

```json
{
  "feature_config": "stft_db_44k",
  "dataset": "full_songs_v1",
  "entries": [
    {
      "feature_path": "data/features/stft_db_44k/train/song_001.npy",
      "source_audio": "/datasets/fma_small/song_001.mp3",
      "label": 0,
      "split": "train",
      "n_frames": 12900,
      "duration_sec": 150.3
    }
  ]
}
```

**Training** random-crops `window_sec` slices from the full spectrogram.
**Evaluation** uses deterministic strided windows and aggregates per file.

---

## 6. Project Structure (MVP)

```text
audio-classification-pipeline/
│
├── configs/
│   ├── datasets/
│   │   └── full_songs_v1.yaml
│   ├── features/
│   │   └── stft_db_44k.yaml
│   ├── models/
│   │   └── spec_cnn_v1.yaml
│   └── experiments/
│       └── exp_001.yaml
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Config loader + dataclasses
│   ├── data/
│   │   ├── __init__.py
│   │   ├── scanner.py             # Scan audio dirs, validate files
│   │   ├── splitter.py            # Stratified train/val/test split
│   │   └── manifest.py            # Read/write manifest files
│   ├── features/
│   │   ├── __init__.py
│   │   ├── audio_loader.py        # Load + resample audio (librosa)
│   │   ├── spectrogram.py         # STFT / mel spectrogram extraction
│   │   └── windowing.py           # Sliding window (for eval)
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn.py                 # SimpleSpectrogramCNN
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Training loop + callbacks
│   │   └── data_loader.py         # tf.data pipeline from .npy
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluator.py           # Inference + per-file aggregation
│       └── metrics.py             # Metrics + plots
│
├── pipelines/
│   ├── prepare_data.py            # CLI: Stage 1
│   ├── extract_features.py        # CLI: Stage 2
│   ├── train.py                   # CLI: Stage 3
│   └── evaluate.py                # CLI: Stage 4
│
├── docker/
│   └── Dockerfile
│
├── tests/
│   └── unit/
│       ├── test_config.py
│       ├── test_splitter.py
│       ├── test_spectrogram.py
│       └── test_windowing.py
│
├── data/                          # Git-ignored
│   ├── raw/
│   ├── manifests/
│   └── features/
│
├── artifacts/                     # Git-ignored
├── docker-compose.yml             # MLflow only
├── Makefile
├── pyproject.toml
├── .env.example
├── .gitignore
├── NOTES.md                       # ← this file
└── README.md
```

### Deferred from full design (to add post-MVP)

| Item | When to add |
|------|-------------|
| Abstract model base class + registry | When adding model #2 or Torch |
| `src/models/tf/` subfolder nesting | When adding Torch alongside TF |
| `src/tracking/mlflow_client.py` wrapper | When MLflow patterns repeat across stages |
| Separate `base.Dockerfile` + `training.Dockerfile` | When dependencies diverge |
| `docker-compose.pipeline.yml` | When stages need inter-container networking |
| `notebooks/` folder | When needed for exploration |
| `tests/integration/` | After pipeline runs end-to-end manually |
| Model Registry / Promotion (Stage 5) | When a model is worth promoting |
| Pydantic config schema validation | When configs get complex or bugs appear |

---

## 7. Infrastructure

### Makefile (MVP)

```makefile
EXP ?= configs/experiments/exp_001.yaml
IMAGE := audio-cls
DOCKER_RUN := docker run --rm \
    --gpus all \
    -v $(PWD)/configs:/app/configs \
    -v $(PWD)/data:/app/data \
    --env-file .env \
    $(IMAGE)

build:          ## Build Docker image
up:             ## Start MLflow
down:           ## Stop MLflow
prepare:        ## Stage 1: Scan audio + split
features:       ## Stage 2: Extract features
train:          ## Stage 3: Train model
evaluate:       ## Stage 4: Evaluate model
run-all:        ## Full pipeline (prepare → features → train → evaluate)
test:           ## Run unit tests
lint:           ## Run linters
```

### Docker Compose (MLflow only for MVP)

Single `docker-compose.yml` running MLflow server with persistent volume.

### Usage

```bash
make run-all EXP=configs/experiments/exp_001.yaml
# or stage by stage:
make prepare  EXP=configs/experiments/exp_001.yaml
make features EXP=configs/experiments/exp_001.yaml
make train    EXP=configs/experiments/exp_001.yaml
make evaluate EXP=configs/experiments/exp_001.yaml
```

---

## 8. Implementation Progress

| Component | Status | File |
|-----------|--------|------|
| Config dataclasses | ✅ Done | `src/config.py` |
| Config loader (YAML → ExperimentConfig) | ✅ Done | `src/config.py` |
| Config unit tests | ✅ Done | `tests/unit/test_config.py` |
| Dataset scanner | ⬜ Next | `src/data/scanner.py` |
| Stratified splitter | ⬜ Next | `src/data/splitter.py` |
| Manifest read/write | ⬜ Next | `src/data/manifest.py` |
| Stage 1 CLI entrypoint | ⬜ Next | `pipelines/prepare_data.py` |
| Audio loader | ⬜ Pending | `src/features/audio_loader.py` |
| Spectrogram extraction | ⬜ Pending | `src/features/spectrogram.py` |
| Stage 2 CLI entrypoint | ⬜ Pending | `pipelines/extract_features.py` |
| CNN model builder | ⬜ Pending | `src/models/cnn.py` |
| tf.data loader | ⬜ Pending | `src/training/data_loader.py` |
| Trainer | ⬜ Pending | `src/training/trainer.py` |
| Stage 3 CLI entrypoint | ⬜ Pending | `pipelines/train.py` |
| Evaluator | ⬜ Pending | `src/evaluation/evaluator.py` |
| Metrics + plots | ⬜ Pending | `src/evaluation/metrics.py` |
| Stage 4 CLI entrypoint | ⬜ Pending | `pipelines/evaluate.py` |
| Dockerfile | ⬜ Pending | `docker/Dockerfile` |
| docker-compose.yml | ⬜ Pending | `docker-compose.yml` |
| Makefile | ⬜ Pending | `Makefile` |

---

## 9. Reference: Original Model Code

The existing codebase has two model classes to port:

- `SimpleCNN` — 1D CNN on raw audio (out of scope for MVP)
- `SimpleSpectrogramCNN` — 2D CNN on spectrograms (in scope)

Key features from original code to preserve:
- Configurable Conv2D blocks (filters, kernels, strides)
- MaxPool vs strided conv toggle (`use_maxpool`)
- Optional BatchNormalization per block
- Global Average/Max Pooling
- Dense head with dropout
- Sigmoid output for binary classification
- `detect_encoder` multi-head support (deferred)
- `get_embedder` intermediate model extraction (deferred)

Original preprocessing pipeline (`effects`):
- `mono` → static preprocessing (Stage 2)
- `affine` → augmentation (Stage 3, on-the-fly)
- `stft_db` → feature computation (Stage 2)
- `normalise` → static preprocessing (Stage 2)
- `slice_hf` → static preprocessing (Stage 2, `hf_cut_hz`)