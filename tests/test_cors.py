"""
tests/test_cors.py — Tests for CORS whitelist enforcement.
Verifies that allowed origins get headers and unknown origins are blocked.
"""

import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestCORSAllowedOrigins:
    """Verify whitelisted origins receive CORS headers."""

    @pytest.mark.parametrize("origin", [
        "http://localhost:5000",
        "http://localhost:3000",
        "http://127.0.0.1:5000",
    ])
    def test_allowed_origin_gets_headers(self, client, origin):
        r = client.options("/health", headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
        })
        # Should have Access-Control-Allow-Origin matching the request
        acao = r.headers.get("Access-Control-Allow-Origin", "")
        assert acao == origin, \
            f"Origin '{origin}' should be allowed but got ACAO='{acao}'"

    def test_vary_origin_header_present(self, client):
        r = client.get("/health", headers={"Origin": "http://localhost:5000"})
        vary = r.headers.get("Vary", "")
        assert "Origin" in vary, "Vary header should include Origin"


class TestCORSBlockedOrigins:
    """Verify unknown/malicious origins are blocked."""

    @pytest.mark.parametrize("origin", [
        "https://evil.com",
        "https://attacker.example.org",
        "http://malicious-site.net",
    ])
    def test_blocked_origin_no_cors_header(self, client, origin):
        r = client.get("/health", headers={"Origin": origin})
        acao = r.headers.get("Access-Control-Allow-Origin", "")
        assert acao == "", \
            f"Evil origin '{origin}' should NOT get ACAO but got '{acao}'"


class TestCORSNoCreds:
    """Verify Allow-Credentials is only sent for whitelisted origins."""

    def test_no_credentials_for_blocked_origin(self, client):
        r = client.get("/health", headers={"Origin": "https://evil.com"})
        creds = r.headers.get("Access-Control-Allow-Credentials", "")
        assert creds != "true", \
            "Credentials should not be allowed for unknown origins"
