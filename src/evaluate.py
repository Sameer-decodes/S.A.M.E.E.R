"""
Model Evaluation and Metrics Module.
Computes comprehensive classification metrics, confusion matrices, ROC/PR curves,
threshold optimization sweeps, and feature engineering lift.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics for given probabilities and threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    
    cm = confusion_matrix(y_true, y_pred)
    # cm layout: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    
    return {
        "threshold": round(threshold, 3),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def optimize_classification_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_min: float = 0.20,
    threshold_max: float = 0.80,
    step: float = 0.02,
) -> Tuple[float, pd.DataFrame, Dict[str, Any]]:
    """
    Sweep decision thresholds from min to max, calculating metrics at each step.
    Finds optimal threshold balancing fraud capture (recall) and investigation workload (precision).
    """
    thresholds = np.arange(threshold_min, threshold_max + step / 2, step)
    records = []
    
    for t in thresholds:
        metrics = evaluate_predictions(y_true, y_prob, threshold=float(t))
        records.append({
            "threshold": metrics["threshold"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "true_positives": metrics["true_positives"],
            "false_positives": metrics["false_positives"],
            "true_negatives": metrics["true_negatives"],
            "false_negatives": metrics["false_negatives"],
        })
        
    df_sweep = pd.DataFrame(records)
    
    # In insurance fraud, missed fraud (FN) is typically ~5x more costly than investigating a false alarm (FP).
    # Cost function: Total Cost = FP * C_investigation + FN * C_fraud_payout
    # Where C_investigation = 1, C_fraud_payout = 5
    df_sweep["business_cost"] = (df_sweep["false_positives"] * 1.0) + (df_sweep["false_negatives"] * 5.0)
    
    # Primary selection: Max F1 with fallback to minimum business cost
    best_row = df_sweep.loc[df_sweep["f1_score"].idxmax()]
    best_threshold = float(best_row["threshold"])
    
    comparison_summary = {
        "default_threshold": 0.50,
        "optimized_threshold": best_threshold,
        "default_f1": float(df_sweep.loc[np.isclose(df_sweep["threshold"], 0.50, atol=0.011), "f1_score"].values[0]) if np.any(np.isclose(df_sweep["threshold"], 0.50, atol=0.011)) else None,
        "optimized_f1": float(best_row["f1_score"]),
        "default_recall": float(df_sweep.loc[np.isclose(df_sweep["threshold"], 0.50, atol=0.011), "recall"].values[0]) if np.any(np.isclose(df_sweep["threshold"], 0.50, atol=0.011)) else None,
        "optimized_recall": float(best_row["recall"]),
        "default_precision": float(df_sweep.loc[np.isclose(df_sweep["threshold"], 0.50, atol=0.011), "precision"].values[0]) if np.any(np.isclose(df_sweep["threshold"], 0.50, atol=0.011)) else None,
        "optimized_precision": float(best_row["precision"]),
        "false_positive_reduction_or_tradeoff": int(best_row["false_positives"]),
        "false_negatives_captured": int(best_row["true_positives"]),
    }
    
    return best_threshold, df_sweep, comparison_summary


def compute_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """Compute coordinates for ROC and Precision-Recall curves."""
    fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
    prec, rec, pr_thresh = precision_recall_curve(y_true, y_prob)
    
    # Subsample for lightweight JSON storage and fast web rendering
    idx_roc = np.linspace(0, len(fpr) - 1, min(100, len(fpr))).astype(int)
    idx_pr = np.linspace(0, len(prec) - 1, min(100, len(prec))).astype(int)
    
    return {
        "roc": {
            "fpr": [round(float(v), 4) for v in fpr[idx_roc]],
            "tpr": [round(float(v), 4) for v in tpr[idx_roc]],
        },
        "pr": {
            "recall": [round(float(v), 4) for v in rec[idx_pr]],
            "precision": [round(float(v), 4) for v in prec[idx_pr]],
        },
    }
