.DEFAULT_GOAL := help

.PHONY: help install lock lint check format test build docs-install docs docs-preview

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_-]+:.*## / {printf "%-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install locked runtime, training, agent, and quality dependencies
	pdm install --check -G train -G agent -G quality

lock: ## Resolve all groups for the recorded Python 3.12 environment
	pdm lock -G:all --python "==3.12.*"

lint: ## Verify the lock, repository hygiene, and Python lint/formatting
	pdm lock --check
	pdm run prek run --all-files
	pdm run lint
	pdm run format-check

check: lint ## Run all quality checks, including types against installed dependencies
	pdm run typecheck

format: ## Format Python code
	pdm run format

test: ## Run pytest without loading model weights
	pdm run test

build: ## Build the source distribution and wheel
	pdm build

docs-install: ## Install only the locked documentation dependencies
	pdm install --check --no-default --no-self -G docs

docs: ## Build the static website with warnings treated as errors
	pdm run docs

docs-preview: docs ## Serve the built website on localhost:8000
	pdm run docs-serve
