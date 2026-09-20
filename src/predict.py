"""
Prediction & Inference Module.
Loads the trained pipeline, runs feature engineering, scores fraud probability,
assigns risk tier, generates actionable guidance, and explains top drivers.
"""
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional, Tuple
from pathlib import Path

from src.config import (
    BEST_MODEL_FILE,
    PREPROCESSOR_FILE,
    METADATA_FILE,
    RISK_TIERS,
    DEFAULT_DECISION_THRESHOLD,
)
from src.feature_engineering import engineer_features, get_feature_names
from src.explain import ClaimExplainer


class ClaimPredictor:
    """
    Singleton or cached inference engine for insurance claims fraud scoring.
    """
    _instance = None
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.metadata = {}
        self.explainer = None
        self.threshold = DEFAULT_DECISION_THRESHOLD
        self._load_artifacts()
        
    def _load_artifacts(self):
        """Load serialized model, preprocessor, and metadata from disk."""
        if BEST_MODEL_FILE.exists():
            self.model = joblib.load(BEST_MODEL_FILE)
            print(f"[Predictor] Loaded model from: {BEST_MODEL_FILE}")
            
        if PREPROCESSOR_FILE.exists():
            self.preprocessor = joblib.load(PREPROCESSOR_FILE)
            print(f"[Predictor] Loaded preprocessor from: {PREPROCESSOR_FILE}")
            
        if METADATA_FILE.exists():
            with open(METADATA_FILE, "r") as f:
                self.metadata = json.load(f)
                self.threshold = self.metadata.get(
                    "decision_thresholds", {}
                ).get("operational_selected", DEFAULT_DECISION_THRESHOLD)
                print(f"[Predictor] Loaded operational threshold: {self.threshold}")
                
        self.explainer = ClaimExplainer.load()
        
    @classmethod
    def get_instance(cls) -> "ClaimPredictor":
        if cls._instance is None:
            cls._instance = ClaimPredictor()
        return cls._instance
        
    def score_risk(self, fraud_probability: float) -> Tuple[int, str, str, str]:
        """
        Convert continuous fraud probability into a standardized 0-100 risk score,
        risk level tier, prediction label, and actionable guidance.
        """
        # Risk score on scale 0 to 100
        risk_score = int(np.clip(round(fraud_probability * 100), 0, 100))
        
        # Categorize into business risk tier
        if risk_score < 30:
            tier_key = "LOW"
        elif risk_score < 60:
            tier_key = "MEDIUM"
        elif risk_score < 80:
            tier_key = "HIGH"
        else:
            tier_key = "CRITICAL"
            
        tier_info = RISK_TIERS[tier_key]
        prediction = "Potential Fraud" if fraud_probability >= self.threshold else "Legitimate Claim"
        recommended_action = tier_info["action"]
        risk_level = tier_key
        
        return risk_score, risk_level, prediction, recommended_action
        
    def predict_single(self, claim_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single insurance claim payload.
        """
        if self.model is None or self.preprocessor is None:
            self._load_artifacts()
            if self.model is None:
                raise RuntimeError("Trained model not found. Run 'py -3.12 -m src.train' first.")
                
        # Convert to single-row DataFrame
        df_raw = pd.DataFrame([claim_dict])
        
        # Apply domain feature engineering
        df_featured = engineer_features(df_raw)
        
        # Feature column selection
        feat_dict = get_feature_names(include_engineered=True)
        cols_to_use = feat_dict["numerical"] + feat_dict["categorical"]
        
        # Ensure all columns exist
        for col in cols_to_use:
            if col not in df_featured.columns:
                df_featured[col] = np.nan
                
        X_df = df_featured[cols_to_use]
        
        # Transform through Scikit-learn preprocessor pipeline
        X_trans = self.preprocessor.transform(X_df)
        
        # Model inference probability
        prob = float(self.model.predict_proba(X_trans)[0, 1])
        
        # Assign risk score and operational action
        risk_score, risk_level, prediction, recommended_action = self.score_risk(prob)
        
        # Explain top contributing risk factors
        top_factors = self.explainer.explain_instance(claim_dict, X_trans, top_k=5)
        
        return {
            "claim_id": claim_dict.get("claim_id", "CLM-UNASSIGNED"),
            "fraud_probability": round(prob, 4),
            "fraud_probability_pct": f"{prob * 100:.1f}%",
            "risk_score": risk_score,
            "risk_level": risk_level,
            "prediction": prediction,
            "decision_threshold_used": self.threshold,
            "recommended_action": recommended_action,
            "top_contributing_factors": top_factors,
        }
        
    def predict_batch(self, claims_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        High-throughput batch scoring for multiple claims.
        """
        if not claims_list:
            return []
            
        if self.model is None or self.preprocessor is None:
            self._load_artifacts()
            
        df_raw = pd.DataFrame(claims_list)
        df_featured = engineer_features(df_raw)
        
        feat_dict = get_feature_names(include_engineered=True)
        cols_to_use = feat_dict["numerical"] + feat_dict["categorical"]
        
        for col in cols_to_use:
            if col not in df_featured.columns:
                df_featured[col] = np.nan
                
        X_df = df_featured[cols_to_use]
        X_trans = self.preprocessor.transform(X_df)
        probs = self.model.predict_proba(X_trans)[:, 1]
        
        results = []
        for i, (claim_dict, prob) in enumerate(zip(claims_list, probs)):
            p = float(prob)
            risk_score, risk_level, prediction, action = self.score_risk(p)
            factors = self.explainer.explain_instance(claim_dict, X_trans[i:i+1], top_k=3)
            
            results.append({
                "claim_id": claim_dict.get("claim_id", f"CLM-BATCH-{i+1:04d}"),
                "fraud_probability": round(p, 4),
                "fraud_probability_pct": f"{p * 100:.1f}%",
                "risk_score": risk_score,
                "risk_level": risk_level,
                "prediction": prediction,
                "recommended_action": action,
                "top_contributing_factors": factors,
            })
            
        return results
