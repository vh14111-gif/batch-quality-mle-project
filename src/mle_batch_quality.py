"""
mle_batch_quality.py
---------------------
Maximum Likelihood Estimation (MLE) for batch quality data.

Problem:
    A food-processing lab records whether each tested batch meets a
    quality standard (Pass/Fail). This is modelled as a sequence of
    independent Bernoulli trials with an unknown success probability p
    (probability that a batch meets the quality standard).

Goal:
    Given three CANDIDATE probability estimates (p1, p2, p3), compute
    the likelihood and log-likelihood of the observed data under each
    candidate, compare them, and determine which candidate best
    explains the observed data. Additionally compute the analytical
    MLE (p_hat = k/n) and show that it maximises the likelihood.

Output:
    - Console summary table
    - results/likelihood_curve.png   (likelihood function over p, candidates marked)
    - results/loglikelihood_comparison.png (bar chart comparing candidates)
    - results/mle_summary.csv        (numeric summary table)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import comb

# ----------------------------------------------------------------------
# 1. Load observed data
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "dataset", "batch_quality_data.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

n = len(df)                                   # total number of batches
k = int(df["Result_Binary"].sum())            # number of batches that PASS (meet standard)

print("=" * 60)
print("OBSERVED DATA SUMMARY")
print("=" * 60)
print(f"Total batches tested (n)      : {n}")
print(f"Batches meeting standard (k)  : {k}")
print(f"Batches failing standard      : {n - k}")
print(f"Observed proportion (k/n)     : {k/n:.4f}")
print()


# ----------------------------------------------------------------------
# 2. Likelihood function for Bernoulli/Binomial data
#    L(p) = C(n,k) * p^k * (1-p)^(n-k)
#    log L(p) = log C(n,k) + k*log(p) + (n-k)*log(1-p)
# ----------------------------------------------------------------------
def likelihood(p, n, k):
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def log_likelihood(p, n, k):
    # avoid log(0)
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    return np.log(comb(n, k)) + k * np.log(p) + (n - k) * np.log(1 - p)


# ----------------------------------------------------------------------
# 3. Candidate probability estimates proposed from the observed data
#    (e.g. proposed by three different lab technicians / methods)
# ----------------------------------------------------------------------
candidates = {
    "Candidate A (p = 0.50, naive guess)": 0.50,
    "Candidate B (p = 0.60, conservative estimate)": 0.60,
    "Candidate C (p = 0.725, sample proportion k/n)": k / n,
}

results = []
for name, p in candidates.items():
    L = likelihood(p, n, k)
    logL = log_likelihood(p, n, k)
    results.append({"Candidate": name, "p": p, "Likelihood": L, "Log-Likelihood": logL})

results_df = pd.DataFrame(results)

# Analytical MLE for Bernoulli/Binomial: p_hat = k / n
p_mle = k / n
L_mle = likelihood(p_mle, n, k)
logL_mle = log_likelihood(p_mle, n, k)

print("=" * 60)
print("CANDIDATE PROBABILITY ESTIMATES - LIKELIHOOD COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False, formatters={
    "p": "{:.4f}".format,
    "Likelihood": "{:.6e}".format,
    "Log-Likelihood": "{:.4f}".format
}))
print()
print(f"Analytical MLE estimate (p_hat = k/n) : {p_mle:.4f}")
print(f"Likelihood at MLE                     : {L_mle:.6e}")
print(f"Log-Likelihood at MLE                 : {logL_mle:.4f}")
print()

best_row = results_df.loc[results_df["Log-Likelihood"].idxmax()]
print(f">> Candidate with HIGHEST likelihood: {best_row['Candidate']}")
print(f">> This matches the analytical MLE p_hat = k/n = {p_mle:.4f}")
print()

# Save numeric summary
summary_out = results_df.copy()
summary_out.loc[len(summary_out)] = ["MLE (analytical, p_hat = k/n)", p_mle, L_mle, logL_mle]
summary_out.to_csv(os.path.join(RESULTS_DIR, "mle_summary.csv"), index=False)


# ----------------------------------------------------------------------
# 4. Plot 1: Likelihood curve over the full range of p, with each
#    candidate and the MLE marked
# ----------------------------------------------------------------------
p_range = np.linspace(0.001, 0.999, 500)
L_values = likelihood(p_range, n, k)

plt.figure(figsize=(9, 6))
plt.plot(p_range, L_values, color="#2b6cb0", linewidth=2, label="Likelihood function L(p)")

colors = ["#e53e3e", "#dd6b20", "#38a169"]
for (name, p), color in zip(candidates.items(), colors):
    plt.axvline(p, color=color, linestyle="--", alpha=0.7)
    plt.plot(p, likelihood(p, n, k), "o", color=color, markersize=9,
              label=f"{name.split('(')[0].strip()} (p={p:.3f})")

plt.plot(p_mle, L_mle, "*", color="black", markersize=18, label=f"MLE estimate (p={p_mle:.3f})")

plt.title(f"Likelihood Function for Batch Quality Data (n={n}, k={k})", fontsize=13)
plt.xlabel("Probability of meeting quality standard (p)")
plt.ylabel("Likelihood L(p)")
plt.legend(fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "likelihood_curve.png"), dpi=150)
plt.close()


# ----------------------------------------------------------------------
# 5. Plot 2: Bar chart comparing log-likelihood of each candidate vs MLE
# ----------------------------------------------------------------------
labels = [c.split("(")[0].strip() for c in candidates.keys()] + ["MLE (p_hat)"]
logL_vals = list(results_df["Log-Likelihood"]) + [logL_mle]
bar_colors = colors + ["black"]

plt.figure(figsize=(9, 6))
bars = plt.bar(labels, logL_vals, color=bar_colors, alpha=0.85)
for bar, val in zip(bars, logL_vals):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.05, f"{val:.3f}",
              ha="center", va="bottom", fontsize=9)

plt.title("Log-Likelihood Comparison Across Candidate Estimates", fontsize=13)
plt.ylabel("Log-Likelihood  log L(p)")
plt.xticks(rotation=15, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "loglikelihood_comparison.png"), dpi=150)
plt.close()

print("Plots saved to results/ :")
print(" - likelihood_curve.png")
print(" - loglikelihood_comparison.png")
print(" - mle_summary.csv")
