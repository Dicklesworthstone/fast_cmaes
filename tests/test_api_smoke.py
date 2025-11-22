import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from fastapi.testclient import TestClient  # type: ignore
    from examples.api_server import app
except ModuleNotFoundError as exc:  # pragma: no cover - guard missing optional deps
    pytest.skip(f"fastapi not available: {exc}", allow_module_level=True)


def test_optimize_endpoint_succeeds():
    client = TestClient(app)
    payload = {"x0": [0.5, 0.5], "sigma": 0.4, "maxfevals": 3000}
    resp = client.post("/optimize", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["dims"] == 2
    assert data["fbest"] < 1e-3
    assert len(data["xmin"]) == 2
