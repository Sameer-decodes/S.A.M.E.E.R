"""
Inference Latency & Throughput Benchmark Suite.
Runs 100+ sequential inference requests to calculate mean, median,
P95, P99 latency and throughput (requests per second).
"""
import time
import json
import numpy as np
from pathlib import Path

from src.predict import ClaimPredictor
from src.config import BENCHMARKS_DIR


def run_latency_benchmark(n_requests: int = 150) -> dict:
    """Execute n_requests inference calls and measure statistical latency percentiles."""
    print(f"\n=======================================================")
    print(f" RUNNING INFERENCE LATENCY BENCHMARK ({n_requests} requests)")
    print(f"=======================================================")
    
    predictor = ClaimPredictor.get_instance()
    
    sample_claim = {
        "claim_id": "CLM-BENCH-001",
        "customer_age": 38,
        "customer_gender": "Male",
        "customer_income": 72000.0,
        "policy_type": "Comprehensive",
        "policy_tenure": 4.2,
        "premium_amount": 1420.0,
        "claim_amount": 16500.0,
        "vehicle_age": 3,
        "vehicle_type": "SUV",
        "accident_type": "Multi-Vehicle",
        "accident_severity": "Major",
        "claim_delay_days": 14,
        "police_report": "Yes",
        "witness_available": "No",
        "repair_estimate": 15800.0,
        "number_of_injuries": 1,
        "hospital_expense": 2400.0,
        "previous_claims": 2,
        "previous_fraud_flags": 0,
        "location_risk_score": 0.62,
        "policy_risk_score": 0.58,
    }
    
    # Warm-up (5 requests)
    for _ in range(5):
        predictor.predict_single(sample_claim)
        
    latencies_ms = []
    start_total = time.time()
    
    for i in range(n_requests):
        t0 = time.perf_counter()
        res = predictor.predict_single(sample_claim)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)
        
    total_elapsed = time.time() - start_total
    rps = round(n_requests / total_elapsed, 2)
    
    latencies_arr = np.array(latencies_ms)
    avg_latency = round(float(np.mean(latencies_arr)), 2)
    median_latency = round(float(np.median(latencies_arr)), 2)
    p90_latency = round(float(np.percentile(latencies_arr, 90)), 2)
    p95_latency = round(float(np.percentile(latencies_arr, 95)), 2)
    p99_latency = round(float(np.percentile(latencies_arr, 99)), 2)
    min_latency = round(float(np.min(latencies_arr)), 2)
    max_latency = round(float(np.max(latencies_arr)), 2)
    
    benchmark_results = {
        "requests_evaluated": n_requests,
        "total_elapsed_sec": round(total_elapsed, 3),
        "requests_per_second": rps,
        "average_latency_ms": avg_latency,
        "median_latency_ms": median_latency,
        "p90_latency_ms": p90_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
    }
    
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = BENCHMARKS_DIR / "latency_results.json"
    with open(out_file, "w") as f:
        json.dump(benchmark_results, f, indent=2)
        
    print(f"Total Requests:      {n_requests}")
    print(f"Requests / Sec:      {rps} req/s")
    print(f"Average Latency:     {avg_latency} ms")
    print(f"Median Latency:      {median_latency} ms")
    print(f"P95 Latency:         {p95_latency} ms")
    print(f"P99 Latency:         {p99_latency} ms")
    print(f"Saved results to:    {out_file}")
    print("=======================================================\n")
    
    return benchmark_results


if __name__ == "__main__":
    run_latency_benchmark(150)
