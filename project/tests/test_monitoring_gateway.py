import asyncio
import importlib.util
from pathlib import Path

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "deployment" / "metrics_gateway.py"
spec = importlib.util.spec_from_file_location("metrics_gateway", MODULE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load monitoring gateway from {MODULE_PATH}")
metrics_gateway = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics_gateway)


def test_input_records_supports_split_and_records_formats():
    split_payload = {
        "dataframe_split": {
            "columns": ["price_lag_1", "city"],
            "data": [[25_000, "Kota Surabaya"]],
        }
    }
    records_payload = {
        "dataframe_records": [{"price_lag_1": 30_000, "city": "Kota Blitar"}]
    }

    assert list(metrics_gateway.input_records(split_payload)) == [
        {"price_lag_1": 25_000, "city": "Kota Surabaya"}
    ]
    assert list(metrics_gateway.input_records(records_payload)) == [
        {"price_lag_1": 30_000, "city": "Kota Blitar"}
    ]


def test_invocation_proxy_records_replica_latency_inputs_and_predictions():
    async def run_test():
        upstream = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={"predictions": [42.5]},
                    headers={"X-Model-Replica": "10.0.0.3:8080"},
                )
            )
        )
        metrics_gateway.app.state.http_client = upstream
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=metrics_gateway.app),
                base_url="http://test",
            ) as client:
                response = await client.post(
                    "/invocations",
                    json={
                        "dataframe_split": {
                            "columns": ["price_lag_1"],
                            "data": [[25_000]],
                        }
                    },
                )
                metrics_response = await client.get("/metrics")

            assert response.status_code == 200
            assert response.json() == {"predictions": [42.5]}
            assert (
                'model_inference_requests_total{replica="10.0.0.3:8080",status_code="200"}'
                in metrics_response.text
            )
            assert 'model_input_feature_value_count{feature="price_lag_1"}' in metrics_response.text
            assert "model_prediction_value_count" in metrics_response.text
        finally:
            await upstream.aclose()

    asyncio.run(run_test())
