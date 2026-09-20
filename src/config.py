"""
Central configuration module for R.I.T.E.S.H. AI: Insurance Claim Risk & Fraud Prediction System.
(Risk Intelligence & Threat Evaluation System for Hypothetical-claims).
Defines paths, feature definitions, business risk tiers, and model hyperparameters.
"""
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_NAME = "R.I.T.E.S.H. AI"
PROJECT_FULL_NAME = "Risk Intelligence & Threat Evaluation System for Hypothetical-claims"

# Project Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_FILE = RAW_DATA_DIR / "insurance_claims_raw.csv"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.csv"

# Models Directory
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_FILE = MODELS_DIR / "best_model.joblib"
BASELINE_MODEL_FILE = MODELS_DIR / "baseline_model.joblib"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"
EXPLAINER_FILE = MODELS_DIR / "explainer.joblib"
METADATA_FILE = MODELS_DIR / "model_metadata.json"

# Benchmarks Directory
BENCHMARKS_DIR = BASE_DIR / "benchmarks"

# Reproducibility
RANDOM_SEED = 42

# Target Variable
TARGET_COL = "fraud_flag"

# Raw Feature Definitions
NUMERICAL_COLS = [
    "customer_age",
    "customer_income",
    "policy_tenure",
    "premium_amount",
    "claim_amount",
    "vehicle_age",
    "claim_delay_days",
    "repair_estimate",
    "number_of_injuries",
    "hospital_expense",
    "previous_claims",
    "previous_fraud_flags",
    "location_risk_score",
    "policy_risk_score",
]

CATEGORICAL_COLS = [
    "customer_gender",
    "policy_type",
    "vehicle_type",
    "accident_type",
    "accident_severity",
    "police_report",
    "witness_available",
    "claim_history",
]

# Engineered Features
ENGINEERED_NUMERICAL_COLS = [
    "claim_to_premium_ratio",
    "claim_frequency",
    "customer_claim_risk",
    "total_financial_exposure",
    "discrepancy_repair_ratio",
    "location_severity_index",
]

ENGINEERED_CATEGORICAL_COLS = [
    "high_claim_indicator",
    "claim_delay_category",
    "unwitnessed_severe_accident",
    "young_driver_luxury_claim",
]

# Business Risk Scoring Configuration
RISK_TIERS: Dict[str, Dict] = {
    "LOW": {
        "score_range": (0, 30),
        "action": "Fast-Track Approval: Low fraud probability detected.",
        "color": "#10B981",  # Emerald green
        "badge": "Low Risk",
    },
    "MEDIUM": {
        "score_range": (30, 60),
        "action": "Standard Claims Review: Perform routine documentation verification.",
        "color": "#F59E0B",  # Amber
        "badge": "Medium Risk",
    },
    "HIGH": {
        "score_range": (60, 80),
        "action": "Manual Investigation: Dispatch claim adjuster for secondary inspection.",
        "color": "#EF4444",  # Crimson
        "badge": "High Risk",
    },
    "CRITICAL": {
        "score_range": (80, 100),
        "action": "Escalate to SIU: High probability of syndicate or staged fraud activity.",
        "color": "#7F1D1D",  # Deep Burgundy
        "badge": "Critical Risk",
    },
}

DEFAULT_DECISION_THRESHOLD = 0.38
