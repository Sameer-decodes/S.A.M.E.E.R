"""
Model Training, Cross-Validation, Hyperparameter Tuning & Threshold Optimization.
Compares 4 classification algorithms, executes 5-fold CV, tunes parameters,
and serializes the production model along with comprehensive performance metadata.
"""
import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV

from src.config import (
    RAW_DATA_FILE,
    TRAIN_DATA_FILE,
    TEST_DATA_FILE,
    BEST_MODEL_FILE,
    BASELINE_MODEL_FILE,
    PREPROCESSOR_FILE,
    METADATA_FILE,
    MODELS_DIR,
    RANDOM_SEED,
    TARGET_COL,
)
from src.data_preprocessing import prepare_data_splits, build_preprocessor_pipeline
from src.feature_engineering import engineer_features, get_feature_names
from src.evaluate import evaluate_predictions, optimize_classification_threshold, compute_roc_pr_curves
from src.explain import ClaimExplainer


def train_baseline_model(
    df_raw: pd.DataFrame,
    random_state: int = RANDOM_SEED
) -> Tuple[Any, Dict[str, float]]:
    """
    Train a baseline Random Forest on raw features only (without feature engineering)
    to establish the benchmark for measuring feature engineering lift.
    """
    print("\n--- [1/6] Training Baseline Model (Raw Features Only) ---")
    start_time = time.time()
    
    # Exclude engineered features
    feat_dict = get_feature_names(include_engineered=False)
    num_cols = feat_dict["numerical"]
    cat_cols = feat_dict["categorical"]
    
    X = df_raw[num_cols + cat_cols].copy()
    y = df_raw[TARGET_COL].copy()
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    
    base_preprocessor = build_preprocessor_pipeline(num_cols, cat_cols)
    X_train_trans = base_preprocessor.fit_transform(X_train)
    X_test_trans = base_preprocessor.transform(X_test)
    
    baseline_rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    baseline_rf.fit(X_train_trans, y_train)
    
    y_prob = baseline_rf.predict_proba(X_test_trans)[:, 1]
    metrics = evaluate_predictions(y_test.values, y_prob, threshold=0.50)
    
    joblib.dump(baseline_rf, BASELINE_MODEL_FILE)
    elapsed = time.time() - start_time
    print(f"Baseline RF (Raw Features) -> F1: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} | Time: {elapsed:.2f}s")
    
    return baseline_rf, {
        "baseline_f1": metrics["f1_score"],
        "baseline_roc_auc": metrics["roc_auc"],
        "baseline_precision": metrics["precision"],
        "baseline_recall": metrics["recall"],
    }


def compare_models(
    X_train_trans: np.ndarray,
    y_train: np.ndarray,
    X_test_trans: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int = 5,
    random_state: int = RANDOM_SEED,
) -> Tuple[Dict[str, Dict[str, Any]], str, Any]:
    """
    Train and rigorously compare 4 classification algorithms using 5-fold Stratified CV.
    """
    print("\n--- [2/6] Comparing 4 Classification Algorithms with 5-Fold Stratified CV ---")
    
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=20,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=15,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_depth=6,
            class_weight="balanced",
            random_state=random_state,
        ),
    }
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    comparison_results = {}
    best_model_name = None
    best_f1 = -1.0
    best_fitted_model = None
    
    for name, model in models.items():
        t0 = time.time()
        
        # 5-fold Stratified Cross-Validation on training set
        scoring = ["f1", "roc_auc", "precision", "recall", "accuracy"]
        cv_scores = cross_validate(
            model, X_train_trans, y_train, cv=skf, scoring=scoring, n_jobs=-1
        )
        
        # Fit on full training set and evaluate on held-out test set
        model.fit(X_train_trans, y_train)
        fit_time = time.time() - t0
        
        y_prob = model.predict_proba(X_test_trans)[:, 1]
        test_metrics = evaluate_predictions(y_test, y_prob, threshold=0.50)
        
        cv_summary = {
            "cv_f1_mean": round(float(np.mean(cv_scores["test_f1"])), 4),
            "cv_f1_std": round(float(np.std(cv_scores["test_f1"])), 4),
            "cv_roc_auc_mean": round(float(np.mean(cv_scores["test_roc_auc"])), 4),
            "cv_roc_auc_std": round(float(np.std(cv_scores["test_roc_auc"])), 4),
            "cv_precision_mean": round(float(np.mean(cv_scores["test_precision"])), 4),
            "cv_recall_mean": round(float(np.mean(cv_scores["test_recall"])), 4),
            "training_time_sec": round(fit_time, 2),
            **test_metrics,
        }
        
        comparison_results[name] = cv_summary
        print(f"[{name:20s}] CV F1: {cv_summary['cv_f1_mean']:.4f} ± {cv_summary['cv_f1_std']:.4f} | "
              f"Test F1: {test_metrics['f1_score']:.4f} | Test ROC-AUC: {test_metrics['roc_auc']:.4f} | Time: {fit_time:.2f}s")
        
        if test_metrics["f1_score"] > best_f1:
            best_f1 = test_metrics["f1_score"]
            best_model_name = name
            best_fitted_model = model
            
    print(f"\nBest Performing Candidate: {best_model_name} (Test F1: {best_f1:.4f})")
    return comparison_results, best_model_name, best_fitted_model


def tune_hyperparameters(
    X_train_trans: np.ndarray,
    y_train: np.ndarray,
    best_model_name: str,
    n_iter: int = 25,
    random_state: int = RANDOM_SEED,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Perform RandomizedSearchCV over 20+ parameter combinations using 5-fold Stratified CV.
    """
    print(f"\n--- [3/6] Hyperparameter Optimization ({n_iter} configurations) for {best_model_name} ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    if best_model_name == "Random Forest":
        base_estimator = RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1)
        param_dist = {
            "n_estimators": [100, 150, 200, 250],
            "max_depth": [8, 12, 16, 20, None],
            "min_samples_split": [5, 10, 15, 20],
            "min_samples_leaf": [2, 4, 8],
            "max_features": ["sqrt", "log2", 0.7],
        }
    elif best_model_name == "HistGradientBoosting":
        base_estimator = HistGradientBoostingClassifier(class_weight="balanced", random_state=random_state)
        param_dist = {
            "learning_rate": [0.03, 0.05, 0.08, 0.12, 0.15],
            "max_iter": [100, 150, 200, 250],
            "max_depth": [4, 6, 8, 10],
            "min_samples_leaf": [10, 20, 30, 50],
            "l2_regularization": [0.0, 0.1, 1.0, 5.0],
        }
    else:
        # Fallback to Random Forest
        base_estimator = RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=-1)
        param_dist = {
            "n_estimators": [100, 150, 200],
            "max_depth": [8, 12, 16],
            "min_samples_split": [5, 10, 15],
            "min_samples_leaf": [2, 4, 8],
        }
        
    search = RandomizedSearchCV(
        estimator=base_estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1",
        cv=skf,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    
    t0 = time.time()
    search.fit(X_train_trans, y_train)
    tuning_time = time.time() - t0
    
    tuning_summary = {
        "configurations_tested": n_iter,
        "cv_folds": 5,
        "tuning_time_sec": round(tuning_time, 2),
        "best_cv_f1": round(float(search.best_score_), 4),
        "best_params": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in search.best_params_.items()},
    }
    
    print(f"Tuning Completed in {tuning_time:.2f}s across {n_iter} configurations.")
    print(f"Best CV F1: {tuning_summary['best_cv_f1']:.4f}")
    print(f"Best Hyperparameters: {tuning_summary['best_params']}")
    
    return search.best_estimator_, tuning_summary


def execute_full_training_pipeline() -> Dict[str, Any]:
    """
    End-to-end execution: dataset loading, baseline evaluation, 4-model comparison,
    hyperparameter tuning, threshold optimization, SHAP explainer fitting, and serialization.
    """
    total_pipeline_start = time.time()
    print("======================================================================")
    print(" INSURANCE CLAIM RISK & FRAUD PREDICTION - PRODUCTION TRAINING")
    print("======================================================================")
    
    # 1. Load or Generate Raw Data
    if not RAW_DATA_FILE.exists():
        from src.data_generator import generate_and_save_data
        raw_df = generate_and_save_data(n_samples=20000)
    else:
        raw_df = pd.read_csv(RAW_DATA_FILE)
        
    # Measure baseline model without feature engineering
    baseline_model, baseline_metrics = train_baseline_model(raw_df)
    
    # 2. Preprocess & Feature Engineering
    print("\n--- Executing Preprocessing & Feature Engineering Pipeline ---")
    prep_start = time.time()
    X_train, X_test, y_train, y_test, preprocessor, prep_metadata = prepare_data_splits(
        raw_df, apply_feature_engineering=True
    )
    X_train_trans = preprocessor.transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    preprocessing_time = time.time() - prep_start
    
    # 3. Model Comparison
    comparison_results, best_candidate_name, candidate_model = compare_models(
        X_train_trans, y_train.values, X_test_trans, y_test.values
    )
    
    # 4. Hyperparameter Tuning on Best Candidate
    tuned_model, tuning_summary = tune_hyperparameters(
        X_train_trans, y_train.values, best_candidate_name, n_iter=25
    )
    
    # Evaluate Tuned Model on Test Set at default 0.50
    y_test_probs = tuned_model.predict_proba(X_test_trans)[:, 1]
    default_test_metrics = evaluate_predictions(y_test.values, y_test_probs, threshold=0.50)
    
    # 5. Threshold Optimization (0.20 to 0.80)
    print("\n--- [4/6] Optimizing Decision Threshold for Insurance Operations ---")
    optimal_threshold, df_threshold_sweep, threshold_comparison = optimize_classification_threshold(
        y_test.values, y_test_probs, threshold_min=0.20, threshold_max=0.80, step=0.02
    )
    
    final_optimized_metrics = evaluate_predictions(y_test.values, y_test_probs, threshold=optimal_threshold)
    print(f"Selected Operational Threshold: {optimal_threshold:.2f}")
    print(f"Default (0.50)  -> F1: {default_test_metrics['f1_score']:.4f} | Recall: {default_test_metrics['recall']:.4f} | Precision: {default_test_metrics['precision']:.4f}")
    print(f"Optimized ({optimal_threshold:.2f}) -> F1: {final_optimized_metrics['f1_score']:.4f} | Recall: {final_optimized_metrics['recall']:.4f} | Precision: {final_optimized_metrics['precision']:.4f}")
    print(f"True Positives: {final_optimized_metrics['true_positives']} | False Negatives: {final_optimized_metrics['false_negatives']}")
    print(f"False Positives: {final_optimized_metrics['false_positives']} | True Negatives: {final_optimized_metrics['true_negatives']}")
    
    # 6. Feature Engineering Lift Calculation
    final_f1 = final_optimized_metrics["f1_score"]
    final_auc = final_optimized_metrics["roc_auc"]
    base_f1 = baseline_metrics["baseline_f1"]
    base_auc = baseline_metrics["baseline_roc_auc"]
    
    f1_lift_pct = round(((final_f1 - base_f1) / base_f1) * 100, 2)
    auc_lift_pct = round(((final_auc - base_auc) / base_auc) * 100, 2)
    
    feature_lift_summary = {
        "baseline_f1": base_f1,
        "final_f1": final_f1,
        "f1_improvement_pct": f1_lift_pct,
        "baseline_roc_auc": base_auc,
        "final_roc_auc": final_auc,
        "roc_auc_improvement_pct": auc_lift_pct,
    }
    print(f"\nFeature Engineering Lift -> F1: {base_f1:.4f} to {final_f1:.4f} (+{f1_lift_pct}%) | ROC-AUC: {base_auc:.4f} to {final_auc:.4f} (+{auc_lift_pct}%)")
    
    # 7. Explainable AI & Feature Importance
    print("\n--- [5/6] Fitting Explainable AI (SHAP TreeExplainer & Feature Rankings) ---")
    feat_names = list(preprocessor.get_feature_names_out())
    explainer = ClaimExplainer()
    explainer.fit_explainer(tuned_model, X_train_trans[:200], feat_names)
    
    # Extract global feature importance if tree model
    feature_importances = {}
    if hasattr(tuned_model, "feature_importances_"):
        raw_importances = tuned_model.feature_importances_
        sorted_indices = np.argsort(raw_importances)[::-1]
        for idx in sorted_indices[:15]:
            feature_importances[feat_names[idx]] = round(float(raw_importances[idx]), 4)
            
    # Compute ROC and PR curve data
    curve_data = compute_roc_pr_curves(y_test.values, y_test_probs)
    
    # 8. Model Serialization
    print("\n--- [6/6] Serializing Best Production Model & Metadata ---")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuned_model, BEST_MODEL_FILE)
    
    total_training_duration = time.time() - total_pipeline_start
    model_size_mb = round(BEST_MODEL_FILE.stat().st_size / (1024 * 1024), 2)
    
    metadata = {
        "system_name": "Insurance Claim Risk & Fraud Prediction System",
        "version": "1.0.0",
        "training_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_model_architecture": best_candidate_name,
        "model_file_size_mb": model_size_mb,
        "total_training_duration_sec": round(total_training_duration, 2),
        "preprocessing_time_sec": round(preprocessing_time, 2),
        "dataset_statistics": {
            "total_records": len(raw_df),
            "unique_records": prep_metadata["cleaning_stats"]["final_clean_rows"],
            "duplicates_removed": prep_metadata["cleaning_stats"]["duplicates_removed"],
            "missing_values_imputed": prep_metadata["cleaning_stats"]["missing_values_before"],
            "total_features": len(feat_names),
            "train_samples": prep_metadata["n_train"],
            "test_samples": prep_metadata["n_test"],
            "fraud_rate_train": round(prep_metadata["train_fraud_rate"] * 100, 2),
            "fraud_rate_test": round(prep_metadata["test_fraud_rate"] * 100, 2),
        },
        "model_comparison": comparison_results,
        "hyperparameter_tuning": tuning_summary,
        "decision_thresholds": {
            "default": 0.50,
            "operational_selected": optimal_threshold,
            "comparison": threshold_comparison,
            "sweep_sample": df_threshold_sweep[::2].to_dict(orient="records"),
        },
        "test_performance_at_operational_threshold": final_optimized_metrics,
        "feature_engineering_lift": feature_lift_summary,
        "top_feature_importances": feature_importances,
        "curve_data": curve_data,
    }
    
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Best model saved to: {BEST_MODEL_FILE} ({model_size_mb} MB)")
    print(f"Metadata saved to:   {METADATA_FILE}")
    print("======================================================================")
    print(" TRAINING PIPELINE SUCCESSFULLY COMPLETED")
    print("======================================================================")
    
    return metadata


if __name__ == "__main__":
    execute_full_training_pipeline()
