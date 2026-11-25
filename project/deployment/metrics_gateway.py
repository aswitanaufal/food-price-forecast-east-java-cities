import json
import math
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Iterator

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest


MODEL_ROUTER_URL = os.getenv("MODEL_ROUTER_URL", "http://model-router:80").rstrip("/")
NUMERIC_FEATURES = {
    "price_lag_1",
    "price_lag_7",
    "rolling_mean_7",
    "price_change",
    "day_of_week",
    "month",
    "temperature_mean",
    "temperature_max",
    "precipitation_sum",
    "humidity_mean",
    "wind_speed_mean",
}
VALUE_BUCKETS = (
    0,
    1,
    5,
    10,
    20,
    30,
    50,
    100,
    500,
    1_000,
    5_000,
    10_000,
    25_000,
    50_000,
    100_000,
    500_000,
)

INFERENCE_REQUESTS = Counter(
    "model_inference_requests_total",
    "Number of inference requests sent to MLflow model replicas.",
    ("replica", "status_code"),
)
INFERENCE_DURATION = Histogram(
    "model_inference_request_duration_seconds",
    "Time spent waiting for an MLflow inference response.",
    ("replica",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
PREDICTION_VALUES = Histogram(
    "model_prediction_value",
    "Distribution of numeric prediction values returned by the model.",
    buckets=VALUE_BUCKETS,
)
INPUT_FEATURE_VALUES = Histogram(
    "model_input_feature_value",
    "Distribution of numeric input feature values for successful inferences.",
    ("feature",),
    buckets=VALUE_BUCKETS,
)


def input_records(payload: Any) -> Iterator[dict[str, Any]]:
    if not isinstance(payload, dict):
        return

    split = payload.get("dataframe_split")
    if isinstance(split, dict):
        columns = split.get("columns")
        rows = split.get("data")
        if isinstance(columns, list) and isinstance(rows, list):
            for row in rows:
                if isinstance(row, list) and len(row) == len(columns):
                    yield dict(zip(columns, row))
                elif isinstance(row, dict):
                    yield row
        return

    for key in ("dataframe_records", "instances"):
        rows = payload.get(key)
        if isinstance(rows, list):
            yield from (row for row in rows if isinstance(row, dict))
            return


def prediction_values(payload: Any) -> Iterator[float]:
    values = payload.get("predictions", payload) if isinstance(payload, dict) else payload
    if not isinstance(values, list):
        values = [values]

    for item in values:
        value = item.get("prediction") if isinstance(item, dict) else item
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric_value = float(value)
            if math.isfinite(numeric_value):
                yield numeric_value


def observe_inputs(payload: Any) -> None:
    for row in input_records(payload):
        for feature in NUMERIC_FEATURES:
            value = row.get(feature)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_value = float(value)
                if math.isfinite(numeric_value):
                    INPUT_FEATURE_VALUES.labels(feature=feature).observe(numeric_value)


def observe_predictions(payload: Any) -> None:
    for value in prediction_values(payload):
        PREDICTION_VALUES.observe(value)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(timeout=60)
    try:
        yield
    finally:
        await app.state.http_client.aclose()


app = FastAPI(
    title="Food Price Model Metrics Gateway",
    version="1.0.0",
    lifespan=lifespan,
)


async def proxy_request(
    request: Request,
    path: str,
    *,
    track_inference: bool,
) -> Response:
    started_at = time.perf_counter()
    content = await request.body()
    headers = {}
    content_type = request.headers.get("content-type")
    if content_type:
        headers["content-type"] = content_type

    try:
        upstream = await request.app.state.http_client.request(
            request.method,
            f"{MODEL_ROUTER_URL}{path}",
            content=content,
            headers=headers,
        )
    except httpx.RequestError as error:
        if track_inference:
            INFERENCE_REQUESTS.labels(
                replica="unavailable",
                status_code="502",
            ).inc()
            INFERENCE_DURATION.labels(replica="unavailable").observe(
                time.perf_counter() - started_at
            )
        raise HTTPException(
            status_code=502,
            detail="MLflow model service is unavailable.",
        ) from error

    replica = upstream.headers.get("x-model-replica", "unknown")
    if track_inference:
        INFERENCE_REQUESTS.labels(
            replica=replica,
            status_code=str(upstream.status_code),
        ).inc()
        INFERENCE_DURATION.labels(replica=replica).observe(
            time.perf_counter() - started_at
        )

        if 200 <= upstream.status_code < 300:
            try:
                request_payload = json.loads(content)
                response_payload = upstream.json()
            except (json.JSONDecodeError, ValueError):
                pass
            else:
                observe_inputs(request_payload)
                observe_predictions(response_payload)

    response_headers = {}
    upstream_content_type = upstream.headers.get("content-type")
    if upstream_content_type:
        response_headers["content-type"] = upstream_content_type
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=response_headers,
    )


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ping")
async def ping(request: Request) -> Response:
    return await proxy_request(request, "/ping", track_inference=False)


@app.post("/invocations")
async def invoke(request: Request) -> Response:
    return await proxy_request(request, "/invocations", track_inference=True)


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
