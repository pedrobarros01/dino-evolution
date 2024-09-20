ifneq ("$(wildcard .env)","")
	include .env
	export
endif

.PHONY: install
install: ## Install Python requirements.
	python -m pip install --upgrade pip setuptools wheel poetry
	poetry lock
	poetry install --no-root
	poetry run pre-commit install

.PHONY: best
best: ## Run the project.
	poetry run python src/app/best_agent.py

.PHONY: train
train: ## Run the project.
	poetry run python src/app/training.py

.PHONY: teste
teste: ## Run environment variables usage example script.
	poetry run python src/scripts/teste.py

.DEFAULT_GOAL := help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sed 's/Makefile://g' | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
