"""
FastAPI REST API Service for Insurance Claim Risk & Fraud Prediction.
Exposes endpoints for single claim scoring, batch scoring, health diagnostics,
and model governance metadata.
"""
from contextlib import asynccontextmanager
from typing import Dict, Any, List
import time
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.predict import ClaimPredictor
from api.schemas import (
    ClaimInput,
    BatchClaimInput,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)

# Global inference engine reference
predictor: ClaimPredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan event handler.
    Loads models and preprocessor pipeline into memory once at startup.
    """
    global predictor
    print("[API Startup] Initializing ClaimPredictor model service...")
    predictor = ClaimPredictor.get_instance()
    print("[API Startup] Service initialized and ready to receive requests.")
    yield
    print("[API Shutdown] Releasing resources.")


app = FastAPI(
    title="R.I.T.E.S.H. AI - Insurance Claim Risk & Fraud Prediction API",
    description="Risk Intelligence & Threat Evaluation System for Hypothetical-claims (RITESH-Risk) Enterprise REST API.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable Cross-Origin Resource Sharing (CORS) for external frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health & Model Status Check",
    tags=["Monitoring"],
)
def get_health():
    """Verify that the API service and ML inference engine are fully operational."""
    is_ready = predictor is not None and predictor.model is not None
    arch = predictor.metadata.get("best_model_architecture", "Random Forest") if predictor else None
    
    return HealthResponse(
        status="healthy" if is_ready else "initializing",
        model_loaded=is_ready,
        model_architecture=arch,
        system_version="1.0.0",
        api_environment="production",
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model Architecture & Performance Metadata",
    tags=["Governance"],
)
def get_model_info():
    """Retrieve model training metrics, 5-fold CV results, decision thresholds, and feature importances."""
    if predictor is None or not predictor.metadata:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata is currently unavailable. Run training pipeline first.",
        )
        
    meta = predictor.metadata
    test_metrics = meta.get("test_performance_at_operational_threshold", {})
    
    return ModelInfoResponse(
        model_name=meta.get("system_name", "Insurance Fraud Classifier"),
        version=meta.get("version", "1.0.0"),
        architecture=meta.get("best_model_architecture", "Random Forest Classifier"),
        decision_threshold=predictor.threshold,
        dataset_summary=meta.get("dataset_statistics", {}),
        evaluation_metrics=test_metrics,
        feature_engineering_lift=meta.get("feature_engineering_lift", {}),
        top_feature_importances=meta.get("top_feature_importances", {}),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Evaluate Single Claim Risk",
    tags=["Scoring"],
)
def predict_claim(payload: ClaimInput):
    """
    Score a single claim payload: computes fraud probability, assigns risk tier,
    provides actionable guidance, and explains top risk drivers.
    """
    if predictor is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model service not ready.")
        
    try:
        claim_dict = payload.model_dump()
        result = predictor.predict_single(claim_dict)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@app.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    summary="Batch Evaluate Multiple Claims",
    tags=["Scoring"],
)
def predict_batch_claims(payload: BatchClaimInput):
    """
    High-throughput batch endpoint: evaluates multiple claims concurrently.
    """
    if predictor is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model service not ready.")
        
    try:
        claims_list = [item.model_dump() for item in payload.claims]
        results = predictor.predict_batch(claims_list)
        high_risk_count = sum(1 for r in results if r["risk_level"] in ["HIGH", "CRITICAL"])
        
        return BatchPredictionResponse(
            total_claims_processed=len(results),
            high_risk_count=high_risk_count,
            results=results,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
