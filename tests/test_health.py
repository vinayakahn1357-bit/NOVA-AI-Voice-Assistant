"""
tests/test_health.py — Tests for the /health endpoint.
Verifies the diagnostic response structure and content.
"""

import pytest
from app import app


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestHealthEndpoint:
    """Verify the /health endpoint returns correct diagnostic data."""

    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_status_ok(self, client):
        r = client.get("/health")
        data = r.get_json()
        assert data["status"] == "healthy"

    def test_health_has_version(self, client):
        r = client.get("/health")
        data = r.get_json()
        assert "version" in data
        assert isinstance(data["version"], str)

    def test_health_has_uptime(self, client):
        r = client.get("/health")
        data = r.get_json()
        assert "uptime_seconds" in data
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0

    def test_health_has_uptime_human(self, client):
        r = client.get("/health")
        data = r.get_json()
        assert "uptime_human" in data

    def test_health_has_memory(self, client):
        r = client.get("/health")
        data = r.get_json()
        assert "memory" in data
        assert "rss_mb" in data["memory"]

    def test_health_has_infrastructure(self, client):
        r = client.get("/health")
        data = r.get_json()
        assert "infrastructure" in data
        infra = data["infrastructure"]
        assert "database" in infra
        assert "cache" in infra

    def test_health_content_type_json(self, client):
        r = client.get("/health")
        assert r.content_type.startswith("application/json")
