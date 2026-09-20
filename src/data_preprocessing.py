"""
Data Preprocessing Pipeline Module.
Implements clean, leakage-free data cleaning, missing value imputation,
outlier clipping, one-hot encoding, and scaling via Scikit-learn ColumnTransformer.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

from src.config import (
    RAW_DATA_FILE,
    TRAIN_DATA_FILE,
    TEST_DATA_FILE,
    PREPROCESSOR_FILE,
    TARGET_COL,
    RANDOM_SEED,
    PROCESSED_DATA_DIR,
)
from src.feature_engineering import engineer_features, get_feature_names


def clean_raw_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Clean raw dataframe: remove duplicates, sanitize negative or impossible values.
    
    Returns:
        Cleaned DataFrame and dict of cleaning statistics.
    """
    stats = {
        "initial_rows": len(df),
        "duplicates_removed": 0,
        "missing_values_before": int(df.isnull().sum().sum()),
    }
    
    # 1. Deduplication on claim_id or entire record
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["claim_id"], keep="first").copy()
    stats["duplicates_removed"] = before_dedup - len(df)
    
    # 2. Sanitize edge bounds (invalid values)
    if "customer_age" in df.columns:
        df["customer_age"] = df["customer_age"].clip(18, 100)
    if "premium_amount" in df.columns:
        df["premium_amount"] = df["premium_amount"].clip(lower=100.0)
    if "claim_amount" in df.columns:
        df["claim_amount"] = df["claim_amount"].clip(lower=100.0)
    if "vehicle_age" in df.columns:
        df["vehicle_age"] = df["vehicle_age"].clip(0, 35)
    if "claim_delay_days" in df.columns:
        df["claim_delay_days"] = df["claim_delay_days"].clip(0, 180)
        
    stats["final_clean_rows"] = len(df)
    return df, stats


def build_preprocessor_pipeline(
    numerical_features: list,
    categorical_features: list,
) -> ColumnTransformer:
    """
    Create Scikit-learn ColumnTransformer for numerical and categorical features.
    Strictly avoids leakage: imputers and scalers are fitted solely on training split.
    """
    # Numerical pipeline: Median imputation + StandardScaler
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    # Categorical pipeline: Most Frequent imputation + OneHotEncoder
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    
    return preprocessor


def prepare_data_splits(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_SEED,
    apply_feature_engineering: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer, Dict]:
    """
    Split into train and test sets, fit preprocessor strictly on train,
    and save processed datasets to disk.
    """
    # 1. Clean data
    df_clean, cleaning_stats = clean_raw_data(df)
    
    # 2. Feature Engineering
    if apply_feature_engineering:
        df_processed = engineer_features(df_clean)
        feature_dict = get_feature_names(include_engineered=True)
    else:
        df_processed = df_clean.copy()
        feature_dict = get_feature_names(include_engineered=False)
        
    num_features = feature_dict["numerical"]
    cat_features = feature_dict["categorical"]
    
    X = df_processed[num_features + cat_features]
    y = df_processed[TARGET_COL]
    
    # 3. Stratified Train-Test Split (preserves class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    # 4. Build and fit preprocessor strictly on training data
    preprocessor = build_preprocessor_pipeline(num_features, cat_features)
    preprocessor.fit(X_train)
    
    # 5. Persist train and test sets for evaluation and dashboard
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_full = df_processed.loc[X_train.index]
    test_full = df_processed.loc[X_test.index]
    
    train_full.to_csv(TRAIN_DATA_FILE, index=False)
    test_full.to_csv(TEST_DATA_FILE, index=False)
    
    # Save preprocessor
    PREPROCESSOR_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_FILE)
    
    metadata = {
        "cleaning_stats": cleaning_stats,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_fraud_rate": float(y_train.mean()),
        "test_fraud_rate": float(y_test.mean()),
        "numerical_features": num_features,
        "categorical_features": cat_features,
        "total_features_pre_encoded": len(num_features) + len(cat_features),
    }
    
    print(f"[Preprocessing] Cleaned {cleaning_stats['initial_rows']:,} rows -> {len(df_clean):,} unique rows.")
    print(f"[Preprocessing] Removed {cleaning_stats['duplicates_removed']} duplicates.")
    print(f"[Preprocessing] Train set: {len(X_train):,} rows ({metadata['train_fraud_rate']:.2%} fraud)")
    print(f"[Preprocessing] Test set:  {len(X_test):,} rows ({metadata['test_fraud_rate']:.2%} fraud)")
    print(f"[Preprocessing] Saved preprocessor to: {PREPROCESSOR_FILE}")
    
    return X_train, X_test, y_train, y_test, preprocessor, metadata


if __name__ == "__main__":
    raw_df = pd.read_csv(RAW_DATA_FILE)
    prepare_data_splits(raw_df)
