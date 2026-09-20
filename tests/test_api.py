"""
Integration tests for FastAPI endpoints and payload validation.
"""
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["system_version"] == "1.0.0"


def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "decision_threshold" in data
    assert "evaluation_metrics" in data
    assert "feature_engineering_lift" in data


def test_predict_endpoint_valid_payload(client):
    payload = {
        "claim_id": "CLM-API-TEST",
        "customer_age": 39,
        "customer_gender": "Male",
        "customer_income": 62000.0,
        "policy_type": "Comprehensive",
        "policy_tenure": 3.5,
        "premium_amount": 1350.0,
        "claim_amount": 12000.0,
        "vehicle_age": 4,
        "vehicle_type": "SUV",
        "accident_type": "Multi-Vehicle",
        "accident_severity": "Moderate",
        "claim_delay_days": 12,
        "police_report": "Yes",
        "witness_available": "No",
        "repair_estimate": 11500.0,
        "number_of_injuries": 0,
        "hospital_expense": 0.0,
        "previous_claims": 1,
        "previous_fraud_flags": 0,
        "location_risk_score": 0.45,
        "policy_risk_score": 0.40,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["claim_id"] == "CLM-API-TEST"
    assert "fraud_probability" in data
    assert "risk_score" in data
    assert "risk_level" in data
    assert "recommended_action" in data
    assert len(data["top_contributing_factors"]) > 0


def test_predict_endpoint_validation_error(client):
    # Invalid customer age (< 18) and missing required fields
    invalid_payload = {
        "customer_age": 14,  # Invalid ge=18
        "claim_amount": -500.0,  # Invalid gt=0
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity
