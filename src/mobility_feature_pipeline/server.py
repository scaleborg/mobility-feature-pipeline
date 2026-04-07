"""FastAPI scoring server for online inference."""

from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Query
from pydantic import BaseModel

from mobility_feature_pipeline.model_config import DEFAULT_TOP_N, FEATURE_COLUMNS, MAX_TOP_N
from mobility_feature_pipeline.serve import ScoringError, load_model, score
from mobility_feature_pipeline.triage import batch_score


class ScoreRequest(BaseModel):
    station_id: str
    obs_ts: datetime


class ScoreResponse(BaseModel):
    station_id: str
    obs_ts: str
    risk_score: float
    risk_label: int
    threshold: float
    snapshot_source_ts: str
    model_name: str
    features: dict | None = None


class TriageRequest(BaseModel):
    obs_ts: datetime
    top_n: int = DEFAULT_TOP_N


class TriageStationDebug(BaseModel):
    availability_ratio: float | None
    snapshot_age_min: float
    bikes_delta_60m: float | None


class TriageStation(BaseModel):
    rank: int
    station_id: str
    risk_score: float
    risk_label: int
    bikes_available: float | None
    capacity: float | None
    snapshot_source_ts: str
    debug: TriageStationDebug | None = None


class SkippedStation(BaseModel):
    station_id: str
    reason: str


class TriageResponse(BaseModel):
    obs_ts: str
    model_name: str
    candidate_stations: int
    scored: int
    skipped: int
    top_n: int
    stations: list[TriageStation]
    skipped_stations: list[SkippedStation]


class ErrorResponse(BaseModel):
    error: str
    detail: str


class HealthResponse(BaseModel):
    status: str
    model_name: str
    db_source: str
    feature_count: int


def create_app(model_path: Path, db_path: Path) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="mobility-feature-pipeline scoring API")

    booster = load_model(model_path)
    model_name = model_path.name
    db_source = db_path.name

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(
            status="ok",
            model_name=model_name,
            db_source=db_source,
            feature_count=len(FEATURE_COLUMNS),
        )

    @app.post(
        "/score",
        response_model=ScoreResponse,
        response_model_exclude_none=True,
        responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    )
    def score_endpoint(req: ScoreRequest, debug: bool = Query(False)):
        from fastapi.responses import JSONResponse

        try:
            result = score(
                booster=booster,
                db_path=db_path,
                station_id=req.station_id,
                obs_ts=req.obs_ts,
                debug=debug,
            )
            result["model_name"] = model_name
            return result
        except ScoringError as e:
            return JSONResponse(
                status_code=e.status,
                content={"error": e.code, "detail": e.detail},
            )

    @app.post(
        "/triage",
        response_model=TriageResponse,
        response_model_exclude_none=True,
        responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    )
    def triage_endpoint(req: TriageRequest, debug: bool = Query(False)):
        from fastapi.responses import JSONResponse

        if req.top_n < 1 or req.top_n > MAX_TOP_N:
            return JSONResponse(
                status_code=422,
                content={
                    "error": "invalid_top_n",
                    "detail": f"top_n must be between 1 and {MAX_TOP_N}",
                },
            )

        try:
            result = batch_score(
                booster=booster,
                db_path=db_path,
                obs_ts=req.obs_ts,
                top_n=req.top_n,
                debug=debug,
            )
            result["model_name"] = model_name
            return result
        except ScoringError as e:
            return JSONResponse(
                status_code=e.status,
                content={"error": e.code, "detail": e.detail},
            )

    return app
