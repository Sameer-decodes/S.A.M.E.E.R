"""
Tests for synthetic data generator and evaluation functions.
"""
import pytest
import numpy as np
import pandas as pd

from src.data_generator import generate_insurance_claims_dataset
from src.evaluate import evaluate_predictions, optimize_classification_threshold, compute_roc_pr_curves


def test_data_generator_output_shape_and_imbalance():
    df = generate_insurance_claims_dataset(
        n_samples=500,
        random_state=42,
        target_fraud_rate=0.12,
        introduce_missingness=True,
        introduce_duplicates=False
    )
    assert len(df) == 500
    assert "fraud_flag" in df.columns
    fraud_rate = df["fraud_flag"].mean()
    assert 0.08 <= fraud_rate <= 0.16
    assert df["customer_income"].isnull().sum() > 0  # Missingness verified


def test_evaluate_predictions_metrics():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9])
    
    metrics = evaluate_predictions(y_true, y_prob, threshold=0.50)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0


def test_optimize_threshold_selection():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.35, 0.38, 0.7, 0.9])
    
    best_thresh, df_sweep, summary = optimize_classification_threshold(
        y_true, y_prob, threshold_min=0.20, threshold_max=0.60, step=0.05
    )
    assert 0.20 <= best_thresh <= 0.60
    assert "optimized_f1" in summary
    assert len(df_sweep) > 0


def test_compute_roc_pr_curves():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])
    curves = compute_roc_pr_curves(y_true, y_prob)
    assert "roc" in curves
    assert "pr" in curves
    assert len(curves["roc"]["fpr"]) > 0
    assert len(curves["pr"]["recall"]) > 0
