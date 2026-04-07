"""CLI entry point for mobility-feature-pipeline."""

from datetime import datetime
from pathlib import Path

import click

from mobility_feature_pipeline.config import Settings
from mobility_feature_pipeline.diagnose import (
    forward_coverage_sensitivity,
    inspect_station,
    station_attrition,
)
from mobility_feature_pipeline.evaluate import load_and_evaluate
from mobility_feature_pipeline.pipeline import build_dataset
from mobility_feature_pipeline.train import train_pipeline
from mobility_feature_pipeline.validate import validate_dataset


@click.group()
def main():
    """Supervised training dataset for next-hour station availability risk."""


@main.command()
@click.option(
    "--db-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to upstream DuckDB file.",
)
@click.option(
    "--output-dir",
    default="./output",
    type=click.Path(path_type=Path),
    help="Output directory for Parquet files.",
)
@click.option(
    "--min-forward-rows",
    default=30,
    type=int,
    help="Minimum 1-min rows in the forward label window.",
)
@click.option("--dry-run", is_flag=True, help="Print SQL and estimated row count only.")
def build(db_path: Path, output_dir: Path, min_forward_rows: int, dry_run: bool):
    """Build the supervised training dataset from upstream DuckDB."""
    settings = Settings(
        upstream_db_path=db_path,
        output_dir=output_dir,
        min_forward_rows=min_forward_rows,
    )
    build_dataset(db_path=db_path, output_dir=output_dir, settings=settings, dry_run=dry_run)


@main.command()
@click.option(
    "--parquet-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the built Parquet file.",
)
def validate(parquet_path: Path):
    """Validate a built training dataset and print a diagnostic report."""
    ok = validate_dataset(parquet_path)
    raise SystemExit(0 if ok else 1)


@main.command()
@click.option(
    "--db-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to upstream DuckDB file.",
)
def attrition(db_path: Path):
    """Report station and observation attrition through each filter stage."""
    station_attrition(db_path)


@main.command()
@click.option(
    "--db-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to upstream DuckDB file.",
)
def sensitivity(db_path: Path):
    """Evaluate dataset retention and positive rate across min_forward_rows thresholds."""
    forward_coverage_sensitivity(db_path)


@main.command("inspect")
@click.option(
    "--db-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to upstream DuckDB file.",
)
@click.option("--station-id", required=True, help="Station ID to inspect.")
@click.option(
    "--start",
    required=True,
    type=click.DateTime(),
    help="Start of inspection window (YYYY-MM-DD HH:MM:SS).",
)
@click.option(
    "--end",
    required=True,
    type=click.DateTime(),
    help="End of inspection window (YYYY-MM-DD HH:MM:SS).",
)
def inspect_cmd(db_path: Path, station_id: str, start: datetime, end: datetime):
    """Inspect one station: features, lags, raw rows, and label justification."""
    inspect_station(db_path, station_id, start, end)


@main.command()
@click.option(
    "--parquet-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the built training dataset Parquet.",
)
@click.option(
    "--output-dir",
    default="./models",
    type=click.Path(path_type=Path),
    help="Output directory for model artifacts.",
)
def train(parquet_path: Path, output_dir: Path):
    """Train baselines + LightGBM and save artifacts."""
    train_pipeline(parquet_path, output_dir)


@main.command()
@click.option(
    "--parquet-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the built training dataset Parquet.",
)
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to saved .lgbm model file.",
)
def evaluate(parquet_path: Path, model_path: Path):
    """Re-load a saved model and reproduce test metrics."""
    load_and_evaluate(parquet_path, model_path)


@main.command()
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to saved .lgbm model file.",
)
@click.option(
    "--db-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to upstream DuckDB file.",
)
@click.option("--port", default=8000, type=int, help="Port to listen on.")
def serve(model_path: Path, db_path: Path, port: int):
    """Start the scoring API server."""
    import uvicorn

    from mobility_feature_pipeline.server import create_app

    app = create_app(model_path=model_path, db_path=db_path)
    uvicorn.run(app, host="0.0.0.0", port=port)


@main.command()
@click.option(
    "--model-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to saved .lgbm model file.",
)
@click.option(
    "--db-path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to upstream DuckDB file.",
)
@click.option(
    "--obs-ts",
    required=True,
    type=click.DateTime(),
    help="Timestamp to triage at (YYYY-MM-DD HH:MM:SS).",
)
@click.option("--top-n", default=10, type=int, help="Number of stations to return.")
def triage(model_path: Path, db_path: Path, obs_ts: datetime, top_n: int):
    """Score all stations and rank by rebalancing urgency."""
    from mobility_feature_pipeline.serve import load_model
    from mobility_feature_pipeline.triage import batch_score, print_triage_report

    booster = load_model(model_path)
    try:
        result = batch_score(
            booster=booster,
            db_path=db_path,
            obs_ts=obs_ts,
            top_n=top_n,
        )
        result["model_name"] = model_path.name
        print_triage_report(result, model_name=model_path.name)
    except Exception as e:
        raise click.ClickException(str(e))
