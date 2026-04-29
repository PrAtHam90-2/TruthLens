"""
Basic integration tests for the TruthLens API.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    """Root endpoint returns app info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "TruthLens API"


def test_health():
    """Health endpoint returns ok."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_returns_valid_response():
    """Analyze endpoint returns expected schema shape."""
    response = client.post(
        "/api/v1/analyze",
        json={"text": "The Earth is flat and the moon landing was faked by NASA."},
    )
    assert response.status_code == 200
    data = response.json()

    # Check top-level keys
    assert "verdict" in data
    assert "confidence_score" in data
    assert "uncertainty_note" in data
    assert "explanation" in data
    assert "claims" in data
    assert isinstance(data["claims"], list)
    assert len(data["claims"]) > 0

    # Check claim shape
    claim = data["claims"][0]
    assert "claim" in claim
    assert "status" in claim
    assert "evidence" in claim
    assert "confidence" in claim
    assert claim["status"] in ["Supported", "Contradicted", "Mixed", "Unknown"]
    assert 0.0 <= claim["confidence"] <= 1.0


def test_analyze_rejects_short_text():
    """Analyze endpoint rejects text that is too short."""
    response = client.post(
        "/api/v1/analyze",
        json={"text": "Hi"},
    )
    assert response.status_code == 422


def test_analyze_rejects_empty():
    """Analyze endpoint rejects missing text field."""
    response = client.post(
        "/api/v1/analyze",
        json={},
    )
    assert response.status_code == 422
