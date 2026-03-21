SHELL := /bin/bash
.DEFAULT_GOAL := help

EXP ?= configs/experiments/exp_001.yaml
IMAGE := genai-audio-detector
OS := $(shell uname -s)
GPU_FLAGS ?= $(if $(filter Linux,$(OS)),--gpus all,)
DATASETS_DIR ?= $(shell grep -s '^DATASETS_DIR=' .env | cut -d= -f2-)

DOCKER_RUN := docker run --rm \
	$(GPU_FLAGS) \
	-v $(PWD)/configs:/app/configs \
	-v $(PWD)/data:/app/data \
	-v $(DATASETS_DIR):/datasets:ro \
	--env-file .env \
	$(IMAGE)

LIMIT ?=

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
	$(DOCKER_RUN) python -m pipelines.prepare_data \
		--config $(EXP) \
		$(if $(LIMIT),--limit $(LIMIT))

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
