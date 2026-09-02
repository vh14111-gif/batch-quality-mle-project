# Batch Quality MLE Project

Maximum Likelihood Estimation (MLE) applied to food-processing batch-quality
data. Each tested batch is recorded as **Pass** (meets the quality standard)
or **Fail**. Three candidate probability estimates are compared using their
likelihood values to determine which one best represents the observed data.

## Problem Statement

A food-processing laboratory records whether each tested batch meets a
quality standard. Three candidate probability estimates are proposed from
the observed data. This project implements Maximum Likelihood Estimation on
the observed batch-quality data, analyzes the likelihood obtained for each
candidate estimate, compares the likelihood values, and determines which
probability estimate best represents the observed batch-quality data.

## Project Structure

```
batch-quality-mle-project/
├── README.md
├── requirements.txt
├── dataset/
│   └── batch_quality_data.csv
├── src/
│   ├── generate_dataset.py
│   └── mle_batch_quality.py
├── notebooks/
│   └── MLE_Analysis.ipynb
├── results/
│   ├── likelihood_curve.png
│   ├── loglikelihood_comparison.png
│   └── mle_summary.csv
└── screenshots/
    └── output.png
```

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional) Regenerate the synthetic dataset:
   ```bash
   python src/generate_dataset.py
   ```
3. Run the MLE analysis:
   ```bash
   python src/mle_batch_quality.py
   ```
4. Or open the interactive notebook:
   ```bash
   jupyter notebook notebooks/MLE_Analysis.ipynb
   ```

## Method Summary

- Each batch outcome is modeled as an independent **Bernoulli trial** with
  unknown probability `p` of meeting the quality standard.
- The **likelihood function** for `n` batches with `k` passes is:

  `L(p) = C(n,k) * p^k * (1-p)^(n-k)`

- Three candidate values of `p` are evaluated and compared using their
  likelihood and log-likelihood values.
- The **analytical MLE** (`p_hat = k/n`) is computed and shown to maximize
  the likelihood function among all possible values of `p`.

## Output

- `results/likelihood_curve.png` – Likelihood function plotted over `p`
  with each candidate and the MLE marked.
- `results/loglikelihood_comparison.png` – Bar chart comparing
  log-likelihood values of all candidates vs. the MLE.
- `results/mle_summary.csv` – Numeric summary table.

## Tools & Libraries

Python, NumPy, Pandas, Matplotlib, SciPy, Jupyter Notebook.
