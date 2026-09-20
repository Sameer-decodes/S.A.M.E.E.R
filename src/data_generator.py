"""
Synthetic Insurance Claims Data Generator.
Generates 20,000+ realistic claims with genuine domain distributions,
subtle missingness, and realistic class imbalance (~12% fraud).
"""
import numpy as np
import pandas as pd
from typing import Optional
from pathlib import Path

from src.config import RANDOM_SEED, RAW_DATA_FILE, RAW_DATA_DIR


def generate_insurance_claims_dataset(
    n_samples: int = 20000,
    random_state: int = RANDOM_SEED,
    target_fraud_rate: float = 0.12,
    introduce_missingness: bool = True,
    introduce_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Generate synthetic insurance claims with realistic non-linear risk drivers.
    
    Args:
        n_samples: Total number of records to generate (>= 20,000).
        random_state: Seed for reproducibility.
        target_fraud_rate: Target fraction of fraudulent claims (default ~12%).
        introduce_missingness: Whether to inject realistic missing values.
        introduce_duplicates: Whether to inject minor duplicate records.
        
    Returns:
        pd.DataFrame containing the generated raw claims data.
    """
    rng = np.random.default_rng(random_state)
    
    # 1. Identifiers and Customer Attributes
    claim_ids = [f"CLM-2024-{i+10001:05d}" for i in range(n_samples)]
    customer_age = np.clip(rng.normal(loc=42, scale=14, size=n_samples).astype(int), 18, 85)
    customer_gender = rng.choice(["Male", "Female", "Other"], size=n_samples, p=[0.49, 0.49, 0.02])
    
    # Log-normal customer income (approx $25k to $200k)
    customer_income = np.round(
        np.clip(rng.lognormal(mean=10.9, sigma=0.45, size=n_samples), 22000, 250000), 2
    )
    
    # 2. Policy Attributes
    policy_types = ["Comprehensive", "Collision", "Third-Party"]
    policy_type = rng.choice(policy_types, size=n_samples, p=[0.55, 0.30, 0.15])
    policy_tenure = np.round(np.clip(rng.exponential(scale=4.5, size=n_samples), 0.2, 28.0), 1)
    
    # Base annual premium based on policy type and tenure
    premium_base = np.where(
        policy_type == "Comprehensive", 1800,
        np.where(policy_type == "Collision", 1200, 750)
    )
    premium_amount = np.round(
        np.clip(premium_base + rng.normal(loc=0, scale=280, size=n_samples) - (policy_tenure * 15), 380, 4800), 2
    )
    
    # 3. Vehicle Attributes
    vehicle_types = ["Sedan", "SUV", "Truck", "Luxury", "Sports"]
    vehicle_type = rng.choice(vehicle_types, size=n_samples, p=[0.42, 0.33, 0.12, 0.08, 0.05])
    vehicle_age = np.clip(rng.poisson(lam=5.5, size=n_samples), 0, 22)
    
    # 4. Incident Attributes
    accident_types = ["Multi-Vehicle", "Single-Vehicle", "Theft", "Vandalism", "Animal Collision"]
    accident_type = rng.choice(accident_types, size=n_samples, p=[0.48, 0.28, 0.10, 0.08, 0.06])
    
    accident_severities = ["Minor", "Moderate", "Major", "Total Loss"]
    accident_severity = rng.choice(accident_severities, size=n_samples, p=[0.38, 0.34, 0.19, 0.09])
    
    # Severity multiplier for damages
    severity_multiplier = np.where(
        accident_severity == "Minor", 1.0,
        np.where(accident_severity == "Moderate", 2.8,
        np.where(accident_severity == "Major", 6.5, 12.0))
    )
    
    vehicle_multiplier = np.where(
        vehicle_type == "Luxury", 2.4,
        np.where(vehicle_type == "Sports", 2.0,
        np.where(vehicle_type == "Truck", 1.3,
        np.where(vehicle_type == "SUV", 1.2, 1.0)))
    )
    
    # Baseline repair estimate
    base_repair = rng.gamma(shape=3.0, scale=1200, size=n_samples)
    repair_estimate = np.round(
        np.clip(base_repair * severity_multiplier * vehicle_multiplier, 350, 58000), 2
    )
    
    # Claim delay in days: right-skewed
    claim_delay_days = rng.geometric(p=0.18, size=n_samples) - 1
    # Add long-tail delays
    long_delay_mask = rng.random(size=n_samples) < 0.08
    claim_delay_days[long_delay_mask] = rng.integers(18, 65, size=np.sum(long_delay_mask))
    
    # Police report and witnesses
    police_report_prob = np.where(
        np.isin(accident_severity, ["Major", "Total Loss"]), 0.88,
        np.where(accident_severity == "Moderate", 0.62, 0.35)
    )
    police_report = np.where(rng.random(size=n_samples) < police_report_prob, "Yes", "No")
    witness_available = rng.choice(["Yes", "No"], size=n_samples, p=[0.45, 0.55])
    
    # Injuries and hospital expenses
    injury_prob = np.where(
        accident_severity == "Total Loss", 0.65,
        np.where(accident_severity == "Major", 0.45,
        np.where(accident_severity == "Moderate", 0.18, 0.04))
    )
    has_injury = rng.random(size=n_samples) < injury_prob
    number_of_injuries = np.where(has_injury, rng.integers(1, 5, size=n_samples), 0)
    hospital_expense = np.where(
        number_of_injuries > 0,
        np.round(number_of_injuries * rng.uniform(1500, 8500, size=n_samples), 2),
        0.0
    )
    
    # History & Risk Ratings
    previous_claims = rng.poisson(lam=0.75, size=n_samples)
    previous_claims = np.clip(previous_claims, 0, 7)
    
    # Previous fraud flags (rare)
    fraud_flag_prob = np.where(previous_claims >= 3, 0.12, np.where(previous_claims >= 1, 0.025, 0.005))
    previous_fraud_flags = np.where(rng.random(size=n_samples) < fraud_flag_prob, rng.choice([1, 2], size=n_samples, p=[0.85, 0.15]), 0)
    
    claim_history = np.where(
        previous_fraud_flags > 0, "Suspicious",
        np.where(previous_claims >= 3, "Frequent",
        np.where(previous_claims >= 1, "Standard", "Clean"))
    )
    
    location_risk_score = np.round(np.clip(rng.beta(a=2.8, b=3.5, size=n_samples), 0.05, 0.95), 3)
    policy_risk_score = np.round(np.clip(rng.beta(a=2.5, b=3.2, size=n_samples), 0.05, 0.95), 3)
    
    # Claim amount initially closely tracks repair_estimate + hospital_expense
    claim_amount = np.round(
        repair_estimate + hospital_expense + rng.normal(0, 150, size=n_samples), 2
    )
    claim_amount = np.clip(claim_amount, 400, 75000)
    
    # 5. Latent Fraud Risk Formulation (Domain-Driven Non-Linear Logic)
    claim_to_prem_raw = claim_amount / (premium_amount + 1e-5)
    unwitnessed_no_police = (np.isin(accident_severity, ["Major", "Total Loss"])) & (police_report == "No") & (witness_available == "No")
    
    latent_score = (
        -3.60  # Base intercept
        + 0.28 * np.log1p(claim_to_prem_raw)
        + 0.038 * claim_delay_days
        + 0.75 * (previous_fraud_flags > 0).astype(float)
        + 0.32 * (previous_claims >= 2).astype(float)
        + 0.85 * unwitnessed_no_police.astype(float)
        + 1.10 * (location_risk_score > 0.65).astype(float)
        + 0.50 * (policy_risk_score > 0.65).astype(float)
        + 0.40 * (accident_type == "Theft").astype(float)
        + 0.35 * (np.isin(vehicle_type, ["Luxury", "Sports"])).astype(float)
        + rng.normal(0, 0.45, size=n_samples)  # Realistic stochastic noise
    )
    
    # Map through sigmoid to get fraud probability
    fraud_prob = 1.0 / (1.0 + np.exp(-latent_score))
    
    # Calibrate probability threshold to achieve desired ~12% fraud rate
    threshold = np.quantile(fraud_prob, 1.0 - target_fraud_rate)
    fraud_flag = (fraud_prob >= threshold).astype(int)
    
    # Inflate claim_amount and create repair discrepancy for fraudulent subset
    fraud_mask = fraud_flag == 1
    inflation_factor = rng.uniform(1.25, 2.2, size=np.sum(fraud_mask))
    claim_amount[fraud_mask] = np.round(claim_amount[fraud_mask] * inflation_factor, 2)
    
    # Assemble DataFrame
    df = pd.DataFrame({
        "claim_id": claim_ids,
        "customer_age": customer_age,
        "customer_gender": customer_gender,
        "customer_income": customer_income,
        "policy_type": policy_type,
        "policy_tenure": policy_tenure,
        "premium_amount": premium_amount,
        "claim_amount": claim_amount,
        "vehicle_age": vehicle_age,
        "vehicle_type": vehicle_type,
        "accident_type": accident_type,
        "accident_severity": accident_severity,
        "claim_delay_days": claim_delay_days,
        "police_report": police_report,
        "witness_available": witness_available,
        "repair_estimate": repair_estimate,
        "number_of_injuries": number_of_injuries,
        "hospital_expense": hospital_expense,
        "previous_claims": previous_claims,
        "previous_fraud_flags": previous_fraud_flags,
        "claim_history": claim_history,
        "location_risk_score": location_risk_score,
        "policy_risk_score": policy_risk_score,
        "fraud_flag": fraud_flag,
    })
    
    # 6. Introduce realistic missing values (2% - 4% in non-critical columns)
    if introduce_missingness:
        income_missing_idx = rng.choice(n_samples, size=int(0.035 * n_samples), replace=False)
        df.loc[income_missing_idx, "customer_income"] = np.nan
        
        witness_missing_idx = rng.choice(n_samples, size=int(0.025 * n_samples), replace=False)
        df.loc[witness_missing_idx, "witness_available"] = np.nan
        
        repair_missing_idx = rng.choice(n_samples, size=int(0.020 * n_samples), replace=False)
        df.loc[repair_missing_idx, "repair_estimate"] = np.nan
    
    # 7. Introduce realistic slight duplicate records (~0.2%)
    if introduce_duplicates:
        dup_count = int(0.002 * n_samples)
        dup_indices = rng.choice(n_samples, size=dup_count, replace=False)
        duplicates = df.iloc[dup_indices].copy()
        df = pd.concat([df, duplicates], ignore_index=True)
        # Shuffle back
        df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        
    return df


def generate_and_save_data(output_path: Optional[Path] = None, n_samples: int = 20000) -> pd.DataFrame:
    """Generate and persist the raw dataset to disk."""
    if output_path is None:
        output_path = RAW_DATA_FILE
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_insurance_claims_dataset(n_samples=n_samples)
    df.to_csv(output_path, index=False)
    
    fraud_count = df["fraud_flag"].sum()
    total_records = len(df)
    fraud_pct = (fraud_count / total_records) * 100
    
    print(f"[Data Generator] Successfully generated {total_records:,} insurance claims.")
    print(f"[Data Generator] Features: {df.shape[1]}")
    print(f"[Data Generator] Fraudulent claims: {fraud_count:,} ({fraud_pct:.2f}%)")
    print(f"[Data Generator] Saved to: {output_path}")
    return df


if __name__ == "__main__":
    generate_and_save_data()
