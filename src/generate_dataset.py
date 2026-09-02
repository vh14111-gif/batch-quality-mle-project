"""
generate_dataset.py
--------------------
Generates a synthetic dataset representing quality-check outcomes of
food-processing batches. Each record represents one tested batch:
    1 -> batch MEETS the quality standard (Pass)
    0 -> batch FAILS the quality standard (Fail)

The dataset is saved to ../dataset/batch_quality_data.csv
"""

import numpy as np
import pandas as pd
import os

# Reproducibility
np.random.seed(42)

# True (unknown, hidden) probability of a batch meeting the standard.
# In real life this would not be known - it is only used here to
# simulate realistic data.
TRUE_P = 0.65

N_BATCHES = 40

# Simulate batch quality outcomes: 1 = Pass (meets standard), 0 = Fail
outcomes = np.random.binomial(1, TRUE_P, size=N_BATCHES)

df = pd.DataFrame({
    "Batch_ID": [f"B{str(i+1).zfill(3)}" for i in range(N_BATCHES)],
    "Quality_Result": np.where(outcomes == 1, "Pass", "Fail"),
    "Result_Binary": outcomes
})

out_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "batch_quality_data.csv")
df.to_csv(out_path, index=False)

print(f"Dataset generated with {N_BATCHES} batches.")
print(f"Number of Pass (meets standard): {outcomes.sum()}")
print(f"Number of Fail: {N_BATCHES - outcomes.sum()}")
print(f"Saved to: {out_path}")
