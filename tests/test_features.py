"""
Unit tests for domain feature engineering.
"""
import pytest
import pandas as pd
import numpy as np

from src.feature_engineering import engineer_features


def test_claim_to_premium_ratio():
    df = pd.DataFrame({
        "claim_amount": [10000.0, 5000.0],
        "premium_amount": [1000.0, 2500.0],
        "policy_tenure": [2.0, 5.0],
        "previous_claims": [1, 0],
        "previous_fraud_flags": [0, 0],
        "accident_severity": ["Minor", "Major"],
        "claim_delay_days": [2, 15],
        "hospital_expense": [0.0, 0.0],
        "repair_estimate": [10000.0, 5000.0],
        "police_report": ["Yes", "No"],
        "witness_available": ["Yes", "No"],
        "location_risk_score": [0.3, 0.7],
        "customer_age": [35, 22],
        "vehicle_type": ["Sedan", "Luxury"],
    })
    
    featured = engineer_features(df)
    assert np.isclose(featured.loc[0, "claim_to_premium_ratio"], 10.0, atol=1e-2)
    assert np.isclose(featured.loc[1, "claim_to_premium_ratio"], 2.0, atol=1e-2)


def test_claim_delay_category_bins():
    df = pd.DataFrame({
        "claim_amount": [3000.0, 3000.0, 3000.0, 3000.0],
        "premium_amount": [1000.0, 1000.0, 1000.0, 1000.0],
        "policy_tenure": [1.0, 1.0, 1.0, 1.0],
        "previous_claims": [0, 0, 0, 0],
        "previous_fraud_flags": [0, 0, 0, 0],
        "accident_severity": ["Minor", "Minor", "Minor", "Minor"],
        "claim_delay_days": [1, 5, 20, 45],
        "hospital_expense": [0.0, 0.0, 0.0, 0.0],
        "repair_estimate": [3000.0, 3000.0, 3000.0, 3000.0],
        "police_report": ["Yes", "Yes", "Yes", "Yes"],
        "witness_available": ["Yes", "Yes", "Yes", "Yes"],
        "location_risk_score": [0.3, 0.3, 0.3, 0.3],
        "customer_age": [40, 40, 40, 40],
        "vehicle_type": ["Sedan", "Sedan", "Sedan", "Sedan"],
    })
    featured = engineer_features(df)
    assert featured.loc[0, "claim_delay_category"] == "Immediate"
    assert featured.loc[1, "claim_delay_category"] == "Standard"
    assert featured.loc[2, "claim_delay_category"] == "Delayed"
    assert featured.loc[3, "claim_delay_category"] == "Critical"


def test_unwitnessed_severe_flag():
    df = pd.DataFrame({
        "claim_amount": [25000.0, 15000.0],
        "premium_amount": [1500.0, 1500.0],
        "policy_tenure": [3.0, 3.0],
        "previous_claims": [0, 0],
        "previous_fraud_flags": [0, 0],
        "accident_severity": ["Major", "Major"],
        "claim_delay_days": [10, 2],
        "hospital_expense": [0.0, 0.0],
        "repair_estimate": [25000.0, 15000.0],
        "police_report": ["No", "Yes"],  # First is unwitnessed without police report
        "witness_available": ["No", "No"],
        "location_risk_score": [0.8, 0.3],
        "customer_age": [35, 45],
        "vehicle_type": ["Luxury", "Sedan"],
    })
    featured = engineer_features(df)
    assert featured.loc[0, "unwitnessed_severe_accident"] == 1
    assert featured.loc[1, "unwitnessed_severe_accident"] == 0
