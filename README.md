# Audio Classification Pipeline — Development Notes

> **Status**: MVP complete — all 4 pipeline stages implemented and running
> **Last updated**: 2026-03-21
> **Stage**: Full pipeline operational (Data Prep → Feature Extraction → Training → Evaluation)

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
│          │   └── {filename}.npy  (n_frames, n_freq)         │
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
  ├── references → configs/datasets/fma_v1.yaml
  ├── references → configs/features/stft_db_44k.yaml
  ├── references → configs/models/spec_cnn_v1.yaml
  └── owns: training params, augmentation, evaluation settings
```

### What each config owns

| Config | Responsibility |
|--------|---------------|
| **dataset** | Paths to audio dirs, file format, split strategy/ratios/seed |
| **features** | Sample rate, mono, feature type, STFT params, optional mel params, normalization, storage format |
| **model** | Architecture type, backbone (filters/kernels/strides), head, output task |
| **experiment** | References above three + training hyperparams, crop/window config, augmentation, evaluation metrics/plots |

### Config files (MVP)

```text
configs/
├── datasets/
│   └── fma_v1.yaml
├── features/
│   ├── stft_db_44k.yaml
│   └── mel_16k_128.yaml
├── models/
│   └── spec_cnn_v1.yaml
└── experiments/
    └── exp_001.yaml
```

<details>
<summary><strong>configs/datasets/fma_v1.yaml</strong></summary>

```yaml
dataset:
  name: full_songs_v1
  description: "FMA small vs AI-rebuilt FMA small"

  paths:
    real: /datasets/real
    fake: /datasets/fake

  format: mp3

  split:
    strategy: stratified
    ratios: [0.70, 0.10, 0.20]   # train / val / test
    seed: 123
```

</details>

You can now choose `features.type` per config:
- `stft_magnitude_db`: linear-frequency STFT magnitude in dB (optional `hf_cut_hz`)
- `mel_spectrogram_db`: mel spectrogram in dB (requires `mel.n_mels`, optional `fmin/fmax/power`)

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

dataset: configs/datasets/fma_v1.yaml
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

One `.npy` file per source audio file containing the **full spectrogram**.

### Tensor layout: time-major `(n_frames, n_freq_bins)`

Arrays on disk and model inputs use **time on axis 0**, **frequency on axis 1** (channel last: `(time, freq, 1)` for `Conv2D`). This matches `tf.signal.stft`’s frame×bin ordering and aligns with the Deezer reference pipeline. Librosa’s STFT is transposed once when building features so all stages share one convention.

**Implication:** Any `.npy` produced **before** this layout change must be **re-extracted** (Stage 2); old checkpoints are **not** compatible with the new input geometry unless you transpose inputs to match how the model was trained.

```text
data/features/stft_db_44k/
├── train/
│   ├── real/
│   │   ├── song_001.npy    # shape: (n_frames, n_freq_bins) — variable n_frames
│   │   ├── song_002.npy
│   │   └── ...
│   └── fake/
│       ├── song_001.npy
│       ├── song_002.npy
│       └── ...
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
      "feature_path": "data/features/stft_db_44k/train/real/song_001.npy",
      "source_audio": "/datasets/real/song_001.mp3",
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
detector-models/
│
├── configs/
│   ├── datasets/
│   │   └── fma_v1.yaml            # FMA small dataset config
│   ├── features/
│   │   └── stft_db_44k.yaml       # STFT magnitude dB features
│   ├── models/
│   │   └── spec_cnn_v1.yaml       # 2D CNN architecture
│   └── experiments/
│       └── exp_001.yaml           # Baseline experiment
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Config loader + dataclasses
│   ├── mlflow_logger.py           # Graceful MLflow wrapper (shared)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── scanner.py             # Scan audio dirs, validate files
│   │   ├── splitter.py            # Stratified train/val/test split
│   │   └── manifest.py            # Read/write manifest files
│   ├── features/
│   │   ├── __init__.py
│   │   ├── audio_loader.py        # Load + resample audio (librosa)
│   │   ├── spectrogram.py         # STFT / mel spectrogram extraction
│   │   └── windowing.py           # Deterministic strided windows (eval)
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn.py                 # SimpleSpectrogramCNN (Keras)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Training loop + callbacks + MLflow
│   │   └── data_loader.py         # tf.data pipeline from .npy
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluator.py           # Inference + per-file aggregation
│       └── metrics.py             # Metrics + plots (sklearn/matplotlib)
│
├── pipelines/
│   ├── prepare_data.py            # CLI: Stage 1
│   ├── extract_features.py        # CLI: Stage 2
│   ├── train.py                   # CLI: Stage 3
│   └── evaluate.py                # CLI: Stage 4
│
├── docker/
│   ├── Dockerfile                 # CPU/default runtime image (python:3.11-slim)
│   └── Dockerfile.gpu             # GPU runtime image (TF 2.11 base)
│
├── tests/
│   └── unit/
│       ├── test_config.py
│       ├── test_splitter.py
│       ├── test_audio_loader.py
│       ├── test_spectrogram.py
│       ├── test_cnn.py
│       ├── test_data_loader.py
│       ├── test_windowing.py
│       └── test_metrics.py
│
├── data/                          # Git-ignored
│   ├── manifests/                 # Stage 1 output
│   ├── features/                  # Stage 2 output (.npy spectrograms)
│   └── artifacts/                 # Stage 3+4 output (models, plots, reports)
│
├── compose.yaml                   # MLflow tracking server
├── Makefile                       # Pipeline orchestration via Docker
├── pyproject.toml                 # Dependencies (hatchling)
├── .env.example                   # Environment variable template
├── .env                           # Local overrides (git-ignored)
├── .gitignore
├── .dockerignore
└── README.md
```

### Deferred (post-MVP)

| Item | When to add |
|------|-------------|
| Abstract model base class + registry | When adding model #2 or Torch |
| `src/models/tf/` subfolder nesting | When adding Torch alongside TF |
| Separate `base.Dockerfile` + `training.Dockerfile` | When dependencies diverge |
| `docker-compose.pipeline.yml` | When stages need inter-container networking |
| `notebooks/` folder | When needed for exploration |
| `tests/integration/` | After pipeline runs end-to-end manually |
| Model Registry / Promotion (Stage 5) | When a model is worth promoting |
| Pydantic config schema validation | When configs get complex or bugs appear |

---

## 7. Infrastructure

### Quick Start

```bash
# 1. Copy and edit environment variables
cp .env.example .env
# Edit .env: set DATASETS paths, MLflow vars, and optional BASE_DOCKERFILE/GPU_DEVICE

# 2. Start MLflow tracking server
make up

# 3. Build the Docker image
make build

# Optional: build with GPU image
# BASE_DOCKERFILE=docker/Dockerfile.gpu make build

# 4. Run the full pipeline
make run-all EXP=configs/experiments/exp_001.yaml

# Or run stage by stage:
make prepare  EXP=configs/experiments/exp_001.yaml
make features EXP=configs/experiments/exp_001.yaml
# Optional: parallelize Stage 2 feature extraction (default is serial)
# make features EXP=configs/experiments/exp_001.yaml WORKERS=8
make train    EXP=configs/experiments/exp_001.yaml
make evaluate EXP=configs/experiments/exp_001.yaml

# Use LIMIT for quick test runs
make prepare EXP=configs/experiments/exp_001.yaml LIMIT=100
```

### Makefile Targets

```text
build            Build Docker image
gpu-check        Check TensorFlow GPU visibility inside image
up               Start MLflow tracking server
down             Stop MLflow
prepare          Stage 1: Scan audio + split
features         Stage 2: Extract features
train            Stage 3: Train model
evaluate         Stage 4: Evaluate model
run-all          Full pipeline (prepare → features → train → evaluate)
test             Run unit tests (locally)
lint             Run ruff linter
```

`WORKERS=<n>` (optional) can be used with `make features` to parallelize Stage 2 feature extraction.
`GPU_DEVICE=<id|all>` (optional) controls which GPU(s) are exposed to Docker runs on Linux.

### Docker Compose

`compose.yaml` runs a self-hosted MLflow server (`ghcr.io/mlflow/mlflow:v3.5.1`)
with SQLite backend and persistent named volume. The MLflow UI is available at
`http://localhost:${MLFLOW_PORT}` (default 5000).

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `DATASETS_DIR` | Host path mounted read-only at `/datasets` (use when all datasets live under one root) | `/Users/you/datasets/fma` |
| `DATASETS_REAL_DIR` | Host path for the *real* dataset (mounted read-only at `/datasets/real`) | `/srv/.../fma/fma_medium` |
| `DATASETS_FAKE_DIR` | Host path for the *fake* dataset (mounted read-only at `/datasets/fake`) | `/home/.../encodec3` |
| `MLFLOW_PORT` | Host port for MLflow UI | `8010` |
| `MLFLOW_TRACKING_URI` | MLflow server URL (used inside Docker containers; Linux typically uses `172.17.0.1`, Mac/Windows typically use `host.docker.internal`) | `http://172.17.0.1:8010` |
| `BASE_DOCKERFILE` | Dockerfile used by `make build` | `docker/Dockerfile.gpu` |
| `GPU_DEVICE` | GPU selection for pipeline containers (`all`, `0`, `1`, `0,1`) | `1` |

---

## 8. Implementation Progress

### Pipeline Stages

| Stage | Status | CLI Entrypoint |
|-------|--------|----------------|
| Stage 1: Data Preparation | ✅ Done | `pipelines/prepare_data.py` |
| Stage 2: Feature Extraction | ✅ Done | `pipelines/extract_features.py` |
| Stage 3: Training | ✅ Done | `pipelines/train.py` |
| Stage 4: Evaluation | ✅ Done | `pipelines/evaluate.py` |

### Source Modules

| Module | Status | Key Files |
|--------|--------|-----------|
| Config loader + dataclasses | ✅ Done | `src/config.py` |
| MLflow logger (shared) | ✅ Done | `src/mlflow_logger.py` |
| Data scanner + splitter | ✅ Done | `src/data/scanner.py`, `splitter.py`, `manifest.py` |
| Audio loader | ✅ Done | `src/features/audio_loader.py` |
| Spectrogram extraction | ✅ Done | `src/features/spectrogram.py` |
| Deterministic windowing | ✅ Done | `src/features/windowing.py` |
| CNN model builder | ✅ Done | `src/models/cnn.py` |
| tf.data pipeline | ✅ Done | `src/training/data_loader.py` |
| Training orchestrator | ✅ Done | `src/training/trainer.py` |
| Evaluator (inference + aggregation) | ✅ Done | `src/evaluation/evaluator.py` |
| Metrics + plots | ✅ Done | `src/evaluation/metrics.py` |

### Infrastructure

| Component | Status | File |
|-----------|--------|------|
| Dockerfile | ✅ Done | `docker/Dockerfile` |
| Docker Compose (MLflow) | ✅ Done | `compose.yaml` |
| Makefile | ✅ Done | `Makefile` |
| pyproject.toml | ✅ Done | `pyproject.toml` |
| Unit tests | ✅ Done | `tests/unit/` (8 test files) |

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