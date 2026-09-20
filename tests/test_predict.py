"""
Unit tests for inference and prediction engine.
"""
import pytest
from src.predict import ClaimPredictor


@pytest.fixture(scope="module")
def predictor():
    return ClaimPredictor.get_instance()


def test_score_risk_tier_thresholds(predictor):
    # Test lower and upper bounds of business risk tiers
    score, tier, pred, action = predictor.score_risk(0.10)
    assert tier == "LOW"
    assert "Fast-Track" in action
    
    score, tier, pred, action = predictor.score_risk(0.45)
    assert tier == "MEDIUM"
    assert "Standard" in action
    
    score, tier, pred, action = predictor.score_risk(0.70)
    assert tier == "HIGH"
    assert "Manual Investigation" in action
    
    score, tier, pred, action = predictor.score_risk(0.95)
    assert tier == "CRITICAL"
    assert "SIU" in action


def test_predict_single_claim_output_format(predictor):
    sample = {
        "claim_id": "CLM-TEST-999",
        "customer_age": 42,
        "customer_gender": "Female",
        "customer_income": 65000.0,
        "policy_type": "Comprehensive",
        "policy_tenure": 5.0,
        "premium_amount": 1500.0,
        "claim_amount": 2500.0,
        "vehicle_age": 3,
        "vehicle_type": "Sedan",
        "accident_type": "Multi-Vehicle",
        "accident_severity": "Minor",
        "claim_delay_days": 1,
        "police_report": "Yes",
        "witness_available": "Yes",
        "repair_estimate": 2500.0,
        "number_of_injuries": 0,
        "hospital_expense": 0.0,
        "previous_claims": 0,
        "previous_fraud_flags": 0,
        "location_risk_score": 0.25,
        "policy_risk_score": 0.20,
    }
    res = predictor.predict_single(sample)
    
    assert res["claim_id"] == "CLM-TEST-999"
    assert 0.0 <= res["fraud_probability"] <= 1.0
    assert 0 <= res["risk_score"] <= 100
    assert res["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert res["prediction"] in ["Potential Fraud", "Legitimate Claim"]
    assert len(res["top_contributing_factors"]) > 0


def test_predict_batch_claims(predictor):
    batch = [
        {
            "claim_id": "CLM-BATCH-1",
            "customer_age": 50,
            "customer_gender": "Male",
            "customer_income": 80000.0,
            "policy_type": "Comprehensive",
            "policy_tenure": 10.0,
            "premium_amount": 1800.0,
            "claim_amount": 1200.0,
            "vehicle_age": 4,
            "vehicle_type": "Sedan",
            "accident_type": "Multi-Vehicle",
            "accident_severity": "Minor",
            "claim_delay_days": 1,
            "police_report": "Yes",
            "witness_available": "Yes",
            "repair_estimate": 1200.0,
            "number_of_injuries": 0,
            "hospital_expense": 0.0,
            "previous_claims": 0,
            "previous_fraud_flags": 0,
            "location_risk_score": 0.2,
            "policy_risk_score": 0.2,
        },
        {
            "claim_id": "CLM-BATCH-2",
            "customer_age": 35,
            "customer_gender": "Female",
            "customer_income": 40000.0,
            "policy_type": "Comprehensive",
            "policy_tenure": 1.0,
            "premium_amount": 1200.0,
            "claim_amount": 35000.0,
            "vehicle_age": 2,
            "vehicle_type": "Luxury",
            "accident_type": "Single-Vehicle",
            "accident_severity": "Total Loss",
            "claim_delay_days": 35,
            "police_report": "No",
            "witness_available": "No",
            "repair_estimate": 15000.0,
            "number_of_injuries": 0,
            "hospital_expense": 0.0,
            "previous_claims": 3,
            "previous_fraud_flags": 1,
            "location_risk_score": 0.85,
            "policy_risk_score": 0.80,
        }
    ]
    results = predictor.predict_batch(batch)
    assert len(results) == 2
    assert results[0]["risk_score"] < results[1]["risk_score"]
