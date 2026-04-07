UPSTREAM_DB ?= ../urban-mobility-control-tower/analytics/data/mobility.duckdb

.PHONY: install lint test build validate train evaluate slice2 serve triage clean

install:
	uv sync

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

test:
	uv run pytest tests/ -v

build:
	uv run mobility-feature-pipeline build --db-path $(UPSTREAM_DB) --output-dir ./output

validate:
	uv run mobility-feature-pipeline validate --parquet-path $$(ls -t output/*.parquet | head -1)

train:
	uv run mobility-feature-pipeline train \
		--parquet-path $$(ls -t output/training_dataset_*.parquet | head -1) \
		--output-dir ./models

evaluate:
	uv run mobility-feature-pipeline evaluate \
		--parquet-path $$(ls -t output/training_dataset_*.parquet | head -1) \
		--model-path $$(ls -t models/model_*.lgbm | head -1)

slice2: build validate train

serve:
	uv run mobility-feature-pipeline serve \
		--model-path $$(ls -t models/model_*.lgbm | head -1) \
		--db-path $(UPSTREAM_DB) \
		--port 8000

triage:
	uv run mobility-feature-pipeline triage \
		--model-path $$(ls -t models/model_*.lgbm | head -1) \
		--db-path $(UPSTREAM_DB) \
		--obs-ts "$(OBS_TS)" \
		--top-n $(or $(TOP_N),10)

clean:
	rm -rf output/ dist/ .pytest_cache/ __pycache__/
