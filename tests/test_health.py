import importlib
import os

import pytest
from starlette.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SKIP_MODEL_INIT", "true")
    import api.settings
    import api.main

    importlib.reload(api.settings)
    importlib.reload(api.main)

    with TestClient(api.main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
