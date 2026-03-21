# Audio Classification MLOps Pipeline — Detailed Design

Great context. This is a well-scoped problem. Let me address everything.

---

## Config Strategy: Composed vs. Monolithic

**Why composed configs are the best practice:**

Imagine you have 2 datasets, 3 feature configs, and 4 model architectures. With monolithic configs, you'd need **24 YAML files**, each with duplicated sections. With composed configs:

```yaml
# configs/experiments/exp_001.yaml
experiment:
  name: "music_detection_mel128_cnn_v1"
  description: "Full song AI detection, mel128 features, baseline CNN"

dataset: "datasets/full_songs_v1.yaml"      # pointer
features: "features/mel_128.yaml"            # pointer
model: "models/cnn_baseline.yaml"            # pointer

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  early_stopping_patience: 10

evaluation:
  metrics: ["accuracy", "f1", "auc_roc", "det_curve"]
  threshold_sweep: true
```

```yaml
# configs/datasets/full_songs_v1.yaml
dataset:
  name: "full_songs_v1"
  human_audio_dir: "/data/raw/human/songs"
  ai_audio_dir: "/data/raw/ai/songs"
  split_ratios: [0.7, 0.15, 0.15]  # train/val/test
  split_strategy: "stratified"
  seed: 42
  sample_rate: 22050
  max_duration_sec: 30
```

```yaml
# configs/features/mel_128.yaml
features:
  type: "mel_spectrogram"
  sample_rate: 22050
  n_fft: 2048
  hop_length: 512
  n_mels: 128
  window_size_sec: 5.0
  hop_size_sec: 2.5         # overlapping windows
  normalization: "per_channel"  # or "global", "none"
  output_format: "npy"
```

```yaml
# configs/models/cnn_baseline.yaml
model:
  framework: "tensorflow"
  architecture: "cnn_baseline"
  input_shape: [128, 216, 1]  # n_mels x time_frames x channels
  layers:
    - { type: "conv2d", filters: 32, kernel: [3,3], activation: "relu" }
    - { type: "maxpool2d", pool_size: [2,2] }
    - { type: "conv2d", filters: 64, kernel: [3,3], activation: "relu" }
    - { type: "maxpool2d", pool_size: [2,2] }
    - { type: "flatten" }
    - { type: "dense", units: 128, activation: "relu", dropout: 0.5 }
    - { type: "dense", units: 1, activation: "sigmoid" }
  optimizer: "adam"
  loss: "binary_crossentropy"
```

**Benefits:**
- Change one feature config → automatically applies to all experiments referencing it
- Easy to diff what changed between experiments
- Avoids copy-paste drift
- Each file is independently validated with its own schema
- Training params stay in the experiment file since they're unique per run

You **can override** composed values at the experiment level too if needed (experiment-level overrides take precedence).

---

## MLflow: Is It Suitable?

**Yes, MLflow covers both your use cases well:**

| Need | MLflow Feature |
|---|---|
| Log training metrics over epochs | `mlflow.log_metric("loss", val, step=epoch)` — same as TensorBoard |
| Checkpoints / model artifacts | `mlflow.log_artifact()` or `mlflow.tensorflow.log_model()` |
| Confusion matrix, DET curves | Log as artifact images/HTML via `mlflow.log_artifact("confusion_matrix.png")` |
| Compare experiments | Built-in UI comparison view |
| Model versioning & promotion | MLflow Model Registry (staging → production) |
| Reproduce runs | Logs params, configs, git hash, environment |

You can also run TensorBoard **alongside** MLflow — MLflow can log TensorBoard event files. But honestly, MLflow alone will cover 95% of what you need and it unifies everything in one place.

MLflow runs as a container in your Compose stack with a persistent volume. Free, self-hosted, no vendor lock-in.

---

## High-Level Pipeline

Your instinct is correct. Here's a refined version:

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
│  Input:  Raw audio paths (human + AI dirs)                  │
│  Steps:  1. Scan & validate audio files (duration, SR, etc)│
│          2. Generate train/val/test split (stratified)      │
│          3. Export manifest (CSV/JSON with paths + labels)  │
│                                                             │
│  Output: data/manifests/full_songs_v1_split.json            │
│          (no features yet — just file references + labels)  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: FEATURE EXTRACTION                                │
│                                                             │
│  Input:  Manifest + feature config                          │
│  Steps:  1. Load audio per manifest entry                   │
│          2. Compute spectrograms with windowing             │
│          3. Save as .npy arrays (one per window)            │
│          4. Export feature manifest (npy paths + labels +   │
│             source file + window index)                     │
│                                                             │
│  Output: data/features/mel_128/                             │
│          ├── train/  val/  test/                            │
│          │   └── *.npy                                      │
│          └── feature_manifest.json                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: TRAINING                                          │
│                                                             │
│  Input:  Feature manifest + model config + training params  │
│  Steps:  1. Build data loaders from .npy feature manifest   │
│          2. Build model from config                         │
│          3. Train with callbacks (early stop, checkpoints)  │
│          4. Log metrics + artifacts to MLflow               │
│                                                             │
│  Output: MLflow run with:                                   │
│          - Logged params, metrics per epoch                 │
│          - Best model checkpoint                            │
│          - Training curves                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4: EVALUATION                                        │
│                                                             │
│  Input:  Trained model (from MLflow) + test feature split   │
│  Steps:  1. Load best checkpoint                            │
│          2. Run inference on test set                        │
│          3. Compute metrics (F1, AUC-ROC, EER)             │
│          4. Generate reports (confusion matrix, DET curve,  │
│             per-class breakdown, threshold analysis)        │
│          5. Log everything to same MLflow run               │
│                                                             │
│  Output: MLflow artifacts:                                  │
│          - confusion_matrix.png                             │
│          - det_curve.png                                    │
│          - roc_curve.png                                    │
│          - classification_report.json                       │
│          - threshold_analysis.json                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 5 (optional): MODEL REGISTRY & PROMOTION             │
│                                                             │
│  If metrics pass thresholds → register in MLflow registry   │
│  Promote: None → Staging → Production                       │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decision: Separate Data Prep from Feature Extraction

I split your original Stage 1 into two stages. Here's why:

- **Data prep (splitting)** is done **once per dataset version**. You don't want to re-split every time you try a new feature config.
- **Feature extraction** depends on the feature config (mel vs. MFCC, different window sizes, etc.). You'll run this multiple times for the same dataset.
- This also means you can **cache features**. If `mel_128` features already exist for `full_songs_v1`, skip extraction and go straight to training.

---

## Project Structure

```text
audio-classification-pipeline/
│
├── configs/
│   ├── datasets/
│   │   ├── full_songs_v1.yaml
│   │   └── broadcast_monitoring_v1.yaml
│   ├── features/
│   │   ├── mel_128.yaml
│   │   └── mel_64_mfcc.yaml
│   ├── models/
│   │   ├── cnn_baseline.yaml
│   │   └── cnn_deep.yaml
│   └── experiments/
│       ├── exp_001_songs_mel128_baseline.yaml
│       └── exp_002_broadcast_mel128_baseline.yaml
│
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── loader.py             # Loads + resolves composed configs
│   │   └── schema.py             # Pydantic models for validation
│   ├── data/
│   │   ├── __init__.py
│   │   ├── scanner.py            # Scan audio dirs, validate files
│   │   ├── splitter.py           # Stratified train/val/test split
│   │   └── manifest.py           # Read/write manifest files
│   ├── features/
│   │   ├── __init__.py
│   │   ├── audio_loader.py       # Load + resample audio (librosa)
│   │   ├── spectrogram.py        # Mel spectrogram extraction
│   │   ├── windowing.py          # Sliding window over audio
│   │   └── pipeline.py           # Orchestrates: load → window → extract → save
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract base: build(), compile(), summary()
│   │   ├── registry.py           # Maps architecture names → classes
│   │   └── tf/
│   │       ├── __init__.py
│   │       └── cnn.py            # CNN implementations
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # Training loop + callbacks
│   │   ├── callbacks.py          # Custom callbacks (MLflow logging, etc.)
│   │   └── data_loader.py        # tf.data pipeline from .npy manifest
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py          # Run inference + compute all metrics
│   │   ├── metrics.py            # F1, AUC, EER, DET
│   │   └── reports.py            # Generate plots + JSON reports
│   └── tracking/
│       ├── __init__.py
│       └── mlflow_client.py      # Wrapper for MLflow operations
│
├── pipelines/
│   ├── prepare_data.py           # CLI entrypoint: Stage 1
│   ├── extract_features.py       # CLI entrypoint: Stage 2
│   ├── train.py                  # CLI entrypoint: Stage 3
│   └── evaluate.py               # CLI entrypoint: Stage 4
│
├── docker/
│   ├── base.Dockerfile           # Python + audio libs (librosa, soundfile)
│   ├── training.Dockerfile       # base + TF + GPU support
│   └── mlflow.Dockerfile         # MLflow server
│
├── docker-compose.yml            # MLflow server + optional services
├── docker-compose.pipeline.yml   # Pipeline stage services
│
├── Makefile
│
├── tests/
│   ├── unit/
│   │   ├── test_splitter.py
│   │   ├── test_spectrogram.py
│   │   └── test_windowing.py
│   └── integration/
│       └── test_pipeline_e2e.py
│
├── notebooks/                    # Exploration / debugging
│   └── eda_spectrograms.ipynb
│
├── data/                         # Git-ignored, mounted as volumes
│   ├── raw/
│   ├── manifests/
│   └── features/
│
├── artifacts/                    # Git-ignored, MLflow artifact store
│
├── pyproject.toml
├── .env.example                  # MLFLOW_TRACKING_URI, DATA_ROOT, etc.
└── README.md
```

## Makefile Targets

```makefile
# ============================================================
# Audio Classification Pipeline — Makefile
# ============================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Config — override via: make train EXP=configs/experiments/exp_002.yaml
EXP ?= configs/experiments/exp_001_songs_mel128_baseline.yaml

# Docker
COMPOSE         := docker compose
COMPOSE_PIPELINE := $(COMPOSE) -f docker-compose.pipeline.yml
BASE_IMAGE      := audio-cls-base
TRAIN_IMAGE     := audio-cls-training

# ============================================================
# Infrastructure
# ============================================================

.PHONY: build
build: ## Build all Docker images
	docker build -f docker/base.Dockerfile -t $(BASE_IMAGE) .
	docker build -f docker/training.Dockerfile -t $(TRAIN_IMAGE) .

.PHONY: mlflow-up
mlflow-up: ## Start MLflow tracking server
	$(COMPOSE) up -d mlflow

.PHONY: mlflow-down
mlflow-down: ## Stop MLflow tracking server
	$(COMPOSE) down mlflow

# ============================================================
# Pipeline Stages
# ============================================================

.PHONY: prepare
prepare: ## Stage 1: Scan audio + generate train/val/test split
	$(COMPOSE_PIPELINE) run --rm prepare \
		python -m pipelines.prepare_data --config $(EXP)

.PHONY: features
features: ## Stage 2: Extract spectrogram features as .npy
	$(COMPOSE_PIPELINE) run --rm features \
		python -m pipelines.extract_features --config $(EXP)

.PHONY: train
train: ## Stage 3: Train model
	$(COMPOSE_PIPELINE) run --rm training \
		python -m pipelines.train --config $(EXP)

.PHONY: evaluate
evaluate: ## Stage 4: Evaluate model on test set
	$(COMPOSE_PIPELINE) run --rm evaluate \
		python -m pipelines.evaluate --config $(EXP)

.PHONY: run-all
run-all: prepare features train evaluate ## Run full pipeline end-to-end

# ============================================================
# Development
# ============================================================

.PHONY: test
test: ## Run unit tests
	pytest tests/unit -v

.PHONY: test-integration
test-integration: ## Run integration tests
	pytest tests/integration -v

.PHONY: lint
lint: ## Run linters
	ruff check src/ pipelines/
	mypy src/

.PHONY: notebook
notebook: ## Start Jupyter notebook server
	$(COMPOSE_PIPELINE) run --rm --service-ports notebook

.PHONY: clean-features
clean-features: ## Remove all extracted features (forces re-extraction)
	rm -rf data/features/*

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
```

Usage would look like:

```bash
# Full pipeline with default experiment
make run-all

# Train with a different experiment
make train EXP=configs/experiments/exp_002_broadcast_mel128_baseline.yaml

# Just re-evaluate with new thresholds (edit config, re-run)
make evaluate EXP=configs/experiments/exp_001_songs_mel128_baseline.yaml
```

---

## Your Two Models Map Cleanly

| Concern | Model 1: Full Songs | Model 2: Broadcast Monitoring |
|---|---|---|
| Dataset config | `datasets/full_songs_v1.yaml` | `datasets/broadcast_v1.yaml` |
| Feature config | Possibly shared `mel_128.yaml` | Maybe different window/hop (shorter for real-time) |
| Model config | Same or different CNN arch | Likely lighter model for latency |
| Experiment config | `exp_001_songs_*` | `exp_002_broadcast_*` |

Same pipeline, same code, different configs.

---

Want me to start implementing specific parts? I'd suggest starting with:
1. **Config schema** (Pydantic models) — so everything validates early
2. **Stage 1 & 2** (data prep + feature extraction) — the foundation everything else builds on
3. **Dockerfiles + Compose** — so you can run things from day one

--- 
# One .npy Per Window vs. Concatenated Arrays

Honestly, both approaches are valid and each has clear tradeoffs. Let me lay them out:

## Option A: One .npy per window

```text
data/features/mel_128/train/
├── song_001_w000.npy    # shape: (128, 216)
├── song_001_w001.npy
├── song_001_w002.npy
├── song_002_w000.npy
└── ...
```

**Pros:**
- Each file = one training sample, dead simple data loading
- Trivial to shuffle at the file level
- You can inspect/debug individual samples easily
- If one file is corrupt, you lose one window, not the whole dataset

**Cons:**
- Thousands/millions of small files — slow filesystem operations, slow `ls`, slow backups
- I/O overhead: opening and closing many file handles during training
- Manifest file becomes huge

## Option B: One .npy per source audio (stacked windows)

```text
data/features/mel_128/train/
├── song_001.npy    # shape: (N_windows, 128, 216)
├── song_002.npy
└── ...
```

**Pros:**
- Far fewer files — cleaner filesystem
- Keeps source-level grouping (important for evaluation — see below)
- Reasonable I/O: load one file, get multiple samples

**Cons:**
- Data loader is slightly more complex (index into windows within files)
- Shuffling across sources requires a bit more logic

## Option C: One big .npy (or .npz / HDF5) per split

```text
data/features/mel_128/
├── train.npy    # shape: (total_windows, 128, 216)
├── val.npy
└── test.npy
```

**Pros:**
- Fastest I/O — memory-map the whole thing, zero per-file overhead
- Simplest data loader

**Cons:**
- Doesn't fit in memory for large datasets (unless you use `np.load(mmap_mode='r')`)
- Regenerate the entire file if you add one song
- Lose all provenance — which window came from which source file?

---

## My Revised Recommendation: Option B

For your use case, **Option B is the sweet spot**, and here's the critical reason I didn't emphasize enough initially:

> **At evaluation time, you need source-level predictions, not window-level.**

When you classify a song, you'll do something like:

```text
song_001 → 12 windows → 12 predictions → aggregate → final label
```

You need to know "these 8 windows all came from `song_001.mp3`" to aggregate predictions (majority vote, mean probability, etc.). Option B preserves that grouping naturally.

The manifest then stays clean:

```json
{
  "split": "train",
  "entries": [
    {
      "feature_path": "data/features/mel_128/train/song_001.npy",
      "source_audio": "data/raw/human/songs/song_001.mp3",
      "label": 0,
      "n_windows": 12,
      "duration_sec": 28.5
    }
  ]
}
```

The data loader just needs to unpack windows and assign the same label to each:

```python
# Simplified training data loader logic
for entry in manifest["entries"]:
    windows = np.load(entry["feature_path"])  # (N_windows, 128, 216)
    labels = np.full(len(windows), entry["label"])  # all same label
    yield from zip(windows, labels)
```

**Want me to start building the config schemas and the data/feature stages with this approach?**
--- 


# MVP Review: What to Cut, What to Keep

Good question. My take: **don't compromise the structure, compromise the scope within it**. Cutting folders is free. Rewriting tangled code later is expensive.

Here's my concrete assessment:

## Cut or Defer

| Item | Verdict | Reasoning |
|---|---|---|
| `src/models/base.py` + `registry.py` | **Defer** | You have one model and one framework. A simple class is fine. Add the abstraction when you introduce model #2 or Torch. |
| `src/models/tf/` subfolder | **Flatten** | Just `src/models/cnn.py` for now. No need for the `tf/` nesting with one framework. |
| `docker-compose.pipeline.yml` | **Defer** | Run pipeline stages directly via `docker run` or even locally with a virtualenv. One `docker-compose.yml` for MLflow only. |
| `training.Dockerfile` separate from `base.Dockerfile` | **Merge** | One `Dockerfile`. Split when you actually have divergent dependencies. |
| `src/tracking/mlflow_client.py` wrapper | **Defer** | Just call `mlflow.log_*` directly in your training/eval code. Extract a wrapper when patterns repeat. |
| `notebooks/` | **Defer** | Create when you need one. Don't scaffold empty folders. |
| `tests/integration/` | **Defer** | Unit tests first. Integration tests after the pipeline runs end-to-end manually. |
| Model Registry / Promotion (Stage 5) | **Defer** | Not needed until you have a model worth promoting. |
| Config schema validation (Pydantic) | **Simplify** | Start with simple dict access from YAML. Add Pydantic when configs get complex or bugs creep in. |

## Keep — Non-Negotiable

| Item | Why |
|---|---|
| **Composed YAML configs** | This is almost zero effort and saves you from day one. Copy-pasting monolithic configs will bite you within a week. |
| **Separate pipeline stages as CLI entrypoints** | `pipelines/prepare_data.py`, `extract_features.py`, `train.py`, `evaluate.py` — even if they're 50 lines each. This is the backbone. |
| **Manifest-based data flow between stages** | Stages communicate via files (manifests + .npy), not in-memory. This is what makes the pipeline reproducible and debuggable. |
| **Feature extraction separated from data prep** | You'll want different feature configs for the same dataset within the first week. |
| **Makefile** | Even simplified, this is your UI to the project. |
| **MLflow via docker-compose** | One service, one command. Gives you experiment tracking from run #1. |

## Simplified MVP Structure

```text
audio-classification-pipeline/
│
├── configs/
│   ├── datasets/
│   │   └── full_songs_v1.yaml
│   ├── features/
│   │   └── mel_128.yaml
│   ├── models/
│   │   └── cnn_baseline.yaml
│   └── experiments/
│       └── exp_001.yaml
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # Load + resolve composed configs (one file)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── scanner.py
│   │   ├── splitter.py
│   │   └── manifest.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── audio_loader.py
│   │   ├── spectrogram.py
│   │   └── windowing.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn.py                 # Flat, no base class, no registry
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   └── data_loader.py
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluator.py
│       └── metrics.py             # Metrics + plots in one file
│
├── pipelines/
│   ├── prepare_data.py
│   ├── extract_features.py
│   ├── train.py
│   └── evaluate.py
│
├── docker/
│   └── Dockerfile                 # Single Dockerfile
│
├── tests/
│   └── unit/
│       ├── test_splitter.py
│       ├── test_spectrogram.py
│       └── test_windowing.py
│
├── data/                          # Git-ignored
│   ├── raw/
│   ├── manifests/
│   └── features/
│
├── docker-compose.yml             # Just MLflow
├── Makefile
├── pyproject.toml
├── .env.example
└── README.md
```

**What changed from the full design:**
- `src/config/` (3 files) → `src/config.py` (1 file)
- `src/models/base.py`, `registry.py`, `tf/cnn.py` → `src/models/cnn.py`
- `src/tracking/` → removed (call MLflow directly)
- `src/evaluation/reports.py` → merged into `metrics.py`
- 2 Dockerfiles → 1
- 2 compose files → 1
- No notebooks folder, no integration tests folder

**Same architectural boundaries. Same stage contracts. Same config design. ~40% fewer files.**

## Simplified Makefile

```makefile
SHELL := /bin/bash
.DEFAULT_GOAL := help

EXP ?= configs/experiments/exp_001.yaml
IMAGE := audio-cls
DOCKER_RUN := docker run --rm \
	--gpus all \
	-v $(PWD)/configs:/app/configs \
	-v $(PWD)/data:/app/data \
	--env-file .env \
	$(IMAGE)

# ============================================================
# Setup
# ============================================================

.PHONY: build
build: ## Build Docker image
	docker build -f docker/Dockerfile -t $(IMAGE) .

.PHONY: up
up: ## Start MLflow
	docker compose up -d

.PHONY: down
down: ## Stop MLflow
	docker compose down

# ============================================================
# Pipeline
# ============================================================

.PHONY: prepare
prepare: ## Stage 1: Scan audio + split
	$(DOCKER_RUN) python -m pipelines.prepare_data --config $(EXP)

.PHONY: features
features: ## Stage 2: Extract features
	$(DOCKER_RUN) python -m pipelines.extract_features --config $(EXP)

.PHONY: train
train: ## Stage 3: Train
	$(DOCKER_RUN) python -m pipelines.train --config $(EXP)

.PHONY: evaluate
evaluate: ## Stage 4: Evaluate
	$(DOCKER_RUN) python -m pipelines.evaluate --config $(EXP)

.PHONY: run-all
run-all: prepare features train evaluate ## Full pipeline

# ============================================================
# Dev
# ============================================================

.PHONY: test
test: ## Run tests
	pytest tests/ -v

.PHONY: lint
lint: ## Lint
	ruff check src/ pipelines/

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
```

---

**This is your MVP. Same bones, less furniture. Ready to start implementing?**

I'd suggest this order:
1. **Configs + config loader** — 30 min, sets the foundation
2. **Stage 1: prepare_data** — scanner + splitter + manifest
3. **Stage 2: extract_features** — audio loading + spectrograms + windowing
4. **Stage 3: train** — data loader + CNN + training loop + MLflow logging
5. **Stage 4: evaluate** — inference + metrics + plots

Pick a stage and I'll write it.