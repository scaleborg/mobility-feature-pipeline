from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    upstream_db_path: Path = Path("../urban-mobility-control-tower/analytics/data/mobility.duckdb")
    output_dir: Path = Path("./output")
    sample_interval_min: int = 15
    min_forward_rows: int = 30
    min_capacity: float = 5.0
    empty_threshold: float = 1.0

    model_config = {"env_prefix": "MFP_"}


SQL_DIR = Path(__file__).parent / "sql"
