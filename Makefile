SHELL := /bin/bash
.DEFAULT_GOAL := help
USER_NAME ?= $(shell grep -s '^USER_NAME=' .env | cut -d= -f2-)

ifeq ($(strip $(USER_NAME)),)
	USER_NAME := $(shell whoami)
endif

EXP ?= configs/experiments/exp_001.yaml
IMAGE := genai-audio-detector-$(USER_NAME)
OS := $(shell uname -s)
BASE_DOCKERFILE ?= $(shell grep -s '^BASE_DOCKERFILE=' .env | cut -d= -f2-)
GPU_DEVICE ?= $(shell grep -s '^GPU_DEVICE=' .env | cut -d= -f2-)

ifeq ($(strip $(BASE_DOCKERFILE)),)
	BASE_DOCKERFILE := docker/Dockerfile
endif

ifeq ($(strip $(GPU_DEVICE)),)
	GPU_DEVICE := all
endif

ifeq ($(strip $(GPU_DEVICE)),all)
	GPU_REQUEST := all
else
	GPU_REQUEST := device=$(GPU_DEVICE)
endif

GPU_FLAGS ?= $(if $(filter Linux,$(OS)),--gpus $(GPU_REQUEST),)
DATASETS_DIR ?= $(shell grep -s '^DATASETS_DIR=' .env | cut -d= -f2-)
DATASETS_REAL_DIR ?= $(shell grep -s '^DATASETS_REAL_DIR=' .env | cut -d= -f2-)
DATASETS_FAKE_DIR ?= $(shell grep -s '^DATASETS_FAKE_DIR=' .env | cut -d= -f2-)

ifeq ($(strip $(DATASETS_REAL_DIR)),)
	DATASETS_VOLUME_MOUNTS := -v $(DATASETS_DIR):/datasets:ro
else
	ifeq ($(strip $(DATASETS_FAKE_DIR)),)
		$(error DATASETS_FAKE_DIR must be set when DATASETS_REAL_DIR is set)
	endif
	DATASETS_VOLUME_MOUNTS := \
		-v $(DATASETS_REAL_DIR):/datasets/real:ro \
		-v $(DATASETS_FAKE_DIR):/datasets/fake:ro
endif

DOCKER_RUN := docker run --rm \
	$(GPU_FLAGS) \
	-v $(PWD)/configs:/app/configs \
	-v $(PWD)/data:/app/data \
	$(DATASETS_VOLUME_MOUNTS) \
	--env-file .env \
	$(IMAGE)

LIMIT ?=
MANIFEST ?=
WORKERS ?=

# ============================================================
# Setup
# ============================================================

.PHONY: build
build: ## Build Docker image
	docker build -f $(BASE_DOCKERFILE) -t $(IMAGE) .

.PHONY: gpu-check
gpu-check: ## Check TensorFlow GPU visibility inside image
	$(DOCKER_RUN) python -c "import tensorflow as tf; print('TF_VERSION=', tf.__version__); print('VISIBLE_GPUS=', tf.config.list_physical_devices('GPU'))"

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
	$(DOCKER_RUN) python -m pipelines.prepare_data \
		--config $(EXP) \
		$(if $(LIMIT),--limit $(LIMIT))

.PHONY: features
features: ## Stage 2: Extract features
	$(DOCKER_RUN) python -m pipelines.extract_features \
		--config $(EXP) \
		$(if $(MANIFEST),--manifest $(MANIFEST)) \
		$(if $(WORKERS),--workers $(WORKERS))

.PHONY: train
train: ## Stage 3: Train (override feature manifest with MANIFEST=path/to/feature_manifest.json)
	$(DOCKER_RUN) python -m pipelines.train --config $(EXP) \
		$(if $(MANIFEST),--feature-manifest $(MANIFEST))

.PHONY: evaluate
evaluate: ## Stage 4: Evaluate (override feature manifest with MANIFEST=...)
	$(DOCKER_RUN) python -m pipelines.evaluate --config $(EXP) \
		$(if $(MANIFEST),--feature-manifest $(MANIFEST))

.PHONY: train-eval
train-eval: train evaluate ## Stage 3+4: Train then evaluate (MANIFEST forwarded to both)

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
