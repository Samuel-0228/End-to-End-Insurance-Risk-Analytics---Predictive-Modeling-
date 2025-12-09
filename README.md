# ACIS Insurance Risk Analytics & Predictive Modeling

## Overview

Welcome to the **ACIS Insurance Risk Analytics** project! This repository implements an end-to-end analytics pipeline for AlphaCare Insurance Solutions (ACIS), analyzing historical South African car insurance data (Feb 2014–Aug 2015) to identify low-risk segments, validate risk drivers via hypothesis testing, and build predictive models for dynamic premium optimization. 

**Business Goal:** Uncover low-risk targets (e.g., Mpumalanga females with Sedans) for 10-15% premium reductions, potentially boosting client acquisition by 5-10% (est. R500k uplift).

**Key Insights:**
- **Portfolio Loss Ratio:** 3.3% (profitable; Gauteng high at 3.3%, Mpumalanga low 2.5%).
- **Risk Drivers:** Provinces/zips/gender differ (rejected H1/H2/H4, p<0.05)—regional segmentation warranted.
- **Modeling:** XGBoost predicts claim severity (R²=0.09, RMSE=906 R); top feature: SumInsured (15% importance, +R500 high-value risk).
- **Premium Framework:** Risk-based = P(claim) × Severity + R200 margin (~R426 avg; low-risk R350).

Built over 4 days (Dec 03-09, 2025) simulating financial deadlines: Git/DVC for repro, EDA for patterns, tests for validation, ML for pricing. Synthetic 10k-row sample mirrors real 100k+ structure—swap CSV for production.

**Learning Outcomes:** Data engineering (pipelines), predictive analytics (stats/ML), MLE (modular code/SHAP).

## Quick Start & Reproduction

1. **Clone & Install:**
   ```
   git clone https://github.com/YOUR_USERNAME/acis-insurance-analytics.git
   cd acis-insurance-analytics
   pip install -r requirements.txt  # Pandas, Scikit-learn, XGBoost, SHAP, DVC
   ```

2. **DVC Setup (Data Versioning):**
   ```
   dvc pull  # Restore tracked data (v1.2: raw + processed)
   # If first-time: dvc init && dvc remote add -d localstorage ../dvc-storage (local demo)
   ```

3. **Run Analyses:**
   - **EDA (Task 1):** `python src/eda.py` – Generates plots/stats in `reports/` (e.g., loss heatmap).
   - **Hypothesis Tests (Task 3):** `python src/hypothesis_testing.py` – Outputs CSV/table (e.g., reject H1 p=0.023).
   - **Modeling (Task 4):** `python src/modeling.py` – Trains models, saves metrics/SHAP plot (R²=0.09).
   - **Full Pipeline:** `dvc repro` (future-proof; add dvc.yaml for deps).

4. **View Outputs:**
   - Reports: `reports/final_report.md` (Medium-style summary).
   - Plots: `reports/shap_summary.png` (feature impacts), `reports/loss_ratio_heatmap.png`.

**Expected Runtime:** 2-5 mins (synthetic data); scales to full dataset.

## Project Structure

```
acis-insurance-analytics/
├── .dvc/                          # DVC config/metadata (committed)
│   └── config                     # Remotes/cache
├── .github/workflows/
│   └── ci.yml                     # CI/CD: Tests scripts on push/PR
├── data/raw/                      # Tracked via DVC (git-ignored)
│   ├── insurance_data.csv         # Synthetic/real policies (10k rows)
│   └── insurance_data.csv.dvc     # Metadata (committed)
├── notebooks/
│   └── eda_exploration.ipynb      # Interactive EDA (Jupyter)
├── reports/                       # Outputs & docs
│   ├── correlation_heatmap.png    # EDA plots
│   ├── loss_ratio_heatmap.png     # Creative: Province-Gender risks
│   ├── claims_by_make_trends.png  # Creative: Temporal by make
│   ├── claims_violin_vehicletype.png # Creative: Distributions
│   ├── outliers_boxplot.png
│   ├── hypothesis_results.csv     # Task 3: p-values/decisions
│   ├── model_results.csv          # Task 4: RMSE/R² table
│   ├── feature_importance.csv     # Top feats (XGBoost)
│   ├── shap_summary.png           # Interpretability
│   ├── interim_report.md          # Dec 07 summary
│   └── final_report.md            # Full Medium-style post
├── src/                           # Core scripts (modular Python)
│   ├── eda.py                     # Task 1: Summaries/viz/outliers
│   ├── hypothesis_testing.py      # Task 3: Chi2/ANOVA/t-tests
│   └──
