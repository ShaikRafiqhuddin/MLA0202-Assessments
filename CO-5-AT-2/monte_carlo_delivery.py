"""Q1 - Monte Carlo Sampling for Probability Estimation."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_delivery_trials(n_trials=10000, seed=42):
    rng = np.random.default_rng(seed)
    traffic = rng.choice(["Light", "Moderate", "Heavy"], n_trials, p=[.35, .45, .20])
    probs = {"Light": .95, "Moderate": .85, "Heavy": .65}
    p = np.array([probs[x] for x in traffic])
    on_time = (rng.random(n_trials) < p).astype(int)
    return pd.DataFrame({"trial": np.arange(1, n_trials+1), "traffic": traffic,
                         "on_time": on_time, "on_time_probability": p})

def estimate_probability(data):
    return float(data["on_time"].mean())

def main():
    out = Path(__file__).parent / "outputs"; out.mkdir(exist_ok=True)
    data = simulate_delivery_trials()
    estimate = estimate_probability(data)
    data.to_csv(out/"delivery_trials.csv", index=False)
    cumulative = data.on_time.cumsum() / np.arange(1, len(data)+1)
    plt.figure(figsize=(9,5)); plt.plot(data.trial, cumulative)
    plt.axhline(estimate, linestyle="--", label=f"Final estimate = {estimate:.4f}")
    plt.xlabel("Simulation trial"); plt.ylabel("Estimated P(on-time)")
    plt.title("Monte Carlo Estimate of On-Time Delivery Probability"); plt.legend()
    plt.tight_layout(); plt.savefig(out/"monte_carlo_convergence.png", dpi=150); plt.close()
    summary = data.groupby("traffic").on_time.agg(["count","mean"]).rename(columns={"mean":"observed_probability"})
    summary.to_csv(out/"traffic_summary.csv")
    print(f"Estimated probability of on-time delivery: {estimate:.4f}")
    print(summary)

if __name__ == "__main__":
    main()
