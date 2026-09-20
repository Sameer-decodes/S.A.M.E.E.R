"""
Unit tests for data preprocessing and cleaning pipelines.
"""
import pytest
import pandas as pd
import numpy as np

from src.data_preprocessing import clean_raw_data, build_preprocessor_pipeline
from src.config import NUMERICAL_COLS, CATEGORICAL_COLS


def test_clean_raw_data_removes_duplicates():
    df = pd.DataFrame({
        "claim_id": ["CLM-001", "CLM-002", "CLM-001"],
        "customer_age": [35, 45, 35],
        "premium_amount": [1200.0, 1500.0, 1200.0],
        "claim_amount": [5000.0, 3000.0, 5000.0],
    })
    cleaned_df, stats = clean_raw_data(df)
    assert len(cleaned_df) == 2
    assert stats["duplicates_removed"] == 1
    assert "CLM-001" in cleaned_df["claim_id"].values
    assert "CLM-002" in cleaned_df["claim_id"].values


def test_clean_raw_data_clips_impossible_values():
    df = pd.DataFrame({
        "claim_id": ["CLM-001", "CLM-002"],
        "customer_age": [12, 120],  # out of normal range
        "premium_amount": [-50.0, 2000.0],
        "claim_amount": [-100.0, 5000.0],
        "claim_delay_days": [-5, 300],
    })
    cleaned_df, _ = clean_raw_data(df)
    assert cleaned_df["customer_age"].min() >= 18
    assert cleaned_df["customer_age"].max() <= 100
    assert cleaned_df["premium_amount"].min() >= 100.0
    assert cleaned_df["claim_amount"].min() >= 100.0
    assert cleaned_df["claim_delay_days"].min() >= 0
    assert cleaned_df["claim_delay_days"].max() <= 180


def test_preprocessor_handles_missing_values_and_scales():
    num_cols = ["customer_age", "premium_amount", "claim_amount"]
    cat_cols = ["policy_type", "accident_severity"]
    
    preprocessor = build_preprocessor_pipeline(num_cols, cat_cols)
    
    df_train = pd.DataFrame({
        "customer_age": [30, 40, 50, np.nan],
        "premium_amount": [1000.0, 1500.0, np.nan, 2000.0],
        "claim_amount": [2000.0, 4000.0, 6000.0, 8000.0],
        "policy_type": ["Comprehensive", "Collision", "Comprehensive", np.nan],
        "accident_severity": ["Minor", "Moderate", "Major", "Minor"],
    })
    
    transformed = preprocessor.fit_transform(df_train)
    # Check no NaN values exist in transformed matrix
    assert not np.isnan(transformed).any()
    assert transformed.shape[0] == 4
