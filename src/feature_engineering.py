"""
Feature Engineering Module for Insurance Fraud & Risk Prediction.
Extracts domain-specific actuarial and behavioral risk features.
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-specific feature engineering transformations to insurance claims.
    
    Features engineered:
    1. claim_to_premium_ratio: Actuarial loss ratio proxy. Unusually high values signal disproportionate payout requests.
    2. claim_frequency: Annualized claims rate per year of tenure.
    3. high_claim_indicator: Binary flag indicating claim significantly above standard thresholds for that accident severity.
    4. claim_delay_category: Ordinal bucket for reporting lag (delayed claims have significantly higher fraud rates).
    5. customer_claim_risk: Weighted composite index combining claim count and previous fraud flags.
    6. total_financial_exposure: Sum of claim amount, hospital bills, and repair estimates.
    7. discrepancy_repair_ratio: Relative disparity between claimed amount and repair estimate.
    8. unwitnessed_severe_accident: Severe accident with no police report and no witnesses (classic staging indicator).
    9. location_severity_index: Interaction between location risk and severity level.
    10. young_driver_luxury_claim: High-risk demographic/vehicle interaction.
    """
    df = df.copy()
    
    # 1. Claim to Premium Ratio
    premium = df["premium_amount"].replace(0, np.nan).fillna(df["premium_amount"].median())
    df["claim_to_premium_ratio"] = np.round(df["claim_amount"] / (premium + 1e-5), 4)
    
    # 2. Claim Frequency (Claims per tenure year)
    tenure = df["policy_tenure"].clip(lower=0.2)
    df["claim_frequency"] = np.round(df["previous_claims"] / tenure, 4)
    
    # 3. High Claim Indicator based on accident severity benchmarks
    severity_benchmarks = {
        "Minor": 2500.0,
        "Moderate": 7500.0,
        "Major": 20000.0,
        "Total Loss": 35000.0,
    }
    benchmark_series = df["accident_severity"].map(severity_benchmarks).fillna(10000.0)
    df["high_claim_indicator"] = (df["claim_amount"] > (benchmark_series * 1.35)).astype(int)
    
    # 4. Claim Delay Category
    # 0-2: Immediate, 3-10: Standard, 11-30: Delayed, >30: Critical
    delays = df["claim_delay_days"].fillna(0)
    conditions = [
        (delays <= 2),
        (delays > 2) & (delays <= 10),
        (delays > 10) & (delays <= 30),
        (delays > 30),
    ]
    choices = ["Immediate", "Standard", "Delayed", "Critical"]
    df["claim_delay_category"] = np.select(conditions, choices, default="Standard")
    
    # 5. Customer Claim Risk Composite
    claims = df["previous_claims"].fillna(0)
    flags = df["previous_fraud_flags"].fillna(0)
    df["customer_claim_risk"] = np.round((claims * 1.5) + (flags * 4.0), 3)
    
    # 6. Total Financial Exposure
    hosp = df["hospital_expense"].fillna(0)
    rep = df["repair_estimate"].fillna(df["claim_amount"])
    df["total_financial_exposure"] = np.round(df["claim_amount"] + hosp + rep, 2)
    
    # 7. Discrepancy between Claim and Repair Estimate
    discrepancy = np.abs(df["claim_amount"] - rep)
    df["discrepancy_repair_ratio"] = np.round(discrepancy / (rep + 10.0), 4)
    
    # 8. Unwitnessed Severe Accident without Police Report
    is_severe = df["accident_severity"].isin(["Major", "Total Loss"])
    no_police = df["police_report"].astype(str).str.upper().isin(["NO", "FALSE", "0"])
    no_witness = df["witness_available"].astype(str).str.upper().isin(["NO", "FALSE", "0"])
    df["unwitnessed_severe_accident"] = (is_severe & no_police & no_witness).astype(int)
    
    # 9. Location & Severity Interaction
    severity_weights = {
        "Minor": 1.0,
        "Moderate": 2.0,
        "Major": 3.5,
        "Total Loss": 5.0,
    }
    sev_weight = df["accident_severity"].map(severity_weights).fillna(2.0)
    df["location_severity_index"] = np.round(df["location_risk_score"] * sev_weight, 4)
    
    # 10. Young Driver with Luxury/Sports Vehicle
    is_young = df["customer_age"] < 25
    is_luxury = df["vehicle_type"].isin(["Luxury", "Sports"])
    df["young_driver_luxury_claim"] = (is_young & is_luxury).astype(int)
    
    return df


def get_feature_names(include_engineered: bool = True) -> Dict[str, list]:
    """Return dictionary of numerical and categorical feature names."""
    from src.config import (
        NUMERICAL_COLS,
        CATEGORICAL_COLS,
        ENGINEERED_NUMERICAL_COLS,
        ENGINEERED_CATEGORICAL_COLS,
    )
    
    if not include_engineered:
        return {
            "numerical": NUMERICAL_COLS.copy(),
            "categorical": CATEGORICAL_COLS.copy(),
        }
    
    return {
        "numerical": NUMERICAL_COLS + ENGINEERED_NUMERICAL_COLS,
        "categorical": CATEGORICAL_COLS + ENGINEERED_CATEGORICAL_COLS,
    }
