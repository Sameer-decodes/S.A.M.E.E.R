"""
Pydantic schemas for FastAPI Insurance Claim Risk & Fraud API.
Validates input claim payloads and structures prediction responses.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ClaimInput(BaseModel):
    """Input schema representing an individual insurance claim."""
    claim_id: Optional[str] = Field(default="CLM-AUTO-001", description="Unique claim identifier")
    customer_age: int = Field(..., ge=18, le=100, description="Age of the primary insured policyholder", examples=[42])
    customer_gender: str = Field(..., description="Gender: Male, Female, Other", examples=["Female"])
    customer_income: Optional[float] = Field(default=55000.0, ge=0, description="Annual income of the customer", examples=[62000.0])
    policy_type: str = Field(..., description="Coverage tier: Comprehensive, Collision, Third-Party", examples=["Comprehensive"])
    policy_tenure: float = Field(..., ge=0.0, le=40.0, description="Years policy has been continuously active", examples=[4.5])
    premium_amount: float = Field(..., gt=0.0, description="Annual premium paid ($)", examples=[1450.0])
    claim_amount: float = Field(..., gt=0.0, description="Total amount claimed by policyholder ($)", examples=[18500.0])
    vehicle_age: int = Field(..., ge=0, le=40, description="Age of vehicle in years", examples=[3])
    vehicle_type: str = Field(..., description="Sedan, SUV, Truck, Luxury, Sports", examples=["Luxury"])
    accident_type: str = Field(..., description="Multi-Vehicle, Single-Vehicle, Theft, Vandalism, Animal Collision", examples=["Single-Vehicle"])
    accident_severity: str = Field(..., description="Minor, Moderate, Major, Total Loss", examples=["Major"])
    claim_delay_days: int = Field(..., ge=0, le=365, description="Days elapsed between incident and claim notice", examples=[28])
    police_report: str = Field(..., description="Yes or No", examples=["No"])
    witness_available: Optional[str] = Field(default="No", description="Yes or No", examples=["No"])
    repair_estimate: Optional[float] = Field(default=None, description="Certified repair shop estimate ($)", examples=[11200.0])
    number_of_injuries: int = Field(default=0, ge=0, le=20, description="Number of injured individuals", examples=[0])
    hospital_expense: Optional[float] = Field(default=0.0, ge=0.0, description="Hospitalization and medical expenses ($)", examples=[0.0])
    previous_claims: int = Field(default=0, ge=0, le=30, description="Number of prior claims filed", examples=[3])
    previous_fraud_flags: int = Field(default=0, ge=0, le=10, description="Number of prior fraud investigations", examples=[1])
    claim_history: Optional[str] = Field(default="Clean", description="Clean, Standard, Frequent, Suspicious", examples=["Suspicious"])
    location_risk_score: float = Field(default=0.50, ge=0.0, le=1.0, description="Geographic fraud index (0.0 to 1.0)", examples=[0.82])
    policy_risk_score: float = Field(default=0.50, ge=0.0, le=1.0, description="Actuarial risk rating (0.0 to 1.0)", examples=[0.74])


class BatchClaimInput(BaseModel):
    """Batch container for scoring multiple claims simultaneously."""
    claims: List[ClaimInput]


class RiskDriver(BaseModel):
    """Explaining factor contributing to the risk score."""
    factor: str
    impact: str
    description: str
    importance_score: float


class PredictionResponse(BaseModel):
    """Standardized response payload for risk and fraud scoring."""
    claim_id: str
    fraud_probability: float
    fraud_probability_pct: str
    risk_score: int
    risk_level: str
    prediction: str
    decision_threshold_used: float
    recommended_action: str
    top_contributing_factors: List[RiskDriver]


class BatchPredictionResponse(BaseModel):
    """Response payload for batch claim evaluations."""
    total_claims_processed: int
    high_risk_count: int
    results: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Service health and readiness check."""
    status: str
    model_loaded: bool
    model_architecture: Optional[str]
    system_version: str
    api_environment: str


class ModelInfoResponse(BaseModel):
    """Model governance metadata, validation metrics, and configuration."""
    model_name: str
    version: str
    architecture: str
    decision_threshold: float
    dataset_summary: Dict[str, Any]
    evaluation_metrics: Dict[str, Any]
    feature_engineering_lift: Dict[str, Any]
    top_feature_importances: Dict[str, float]
