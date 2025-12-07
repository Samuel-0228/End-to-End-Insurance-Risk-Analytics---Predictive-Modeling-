# ACIS Insurance Risk Analytics Project

## Overview
Analyzing historical South African car insurance data (Feb 2014–Aug 2015) to optimize marketing, identify low-risk segments, and build predictive models for premiums. Focus: EDA, hypothesis testing, ML modeling.

## Setup
1. Clone repo: `git clone https://github.com/YOUR_USERNAME/acis-insurance-analytics.git`
2. Install deps: `pip install -r requirements.txt`
3. Run EDA: `python src/eda.py`

## Structure
- `data/`: Raw/processed data (tracked via DVC).
- `src/`: Scripts (e.g., `eda.py`, `dvc_setup.py`).
- `notebooks/`: Jupyter for exploration.
- `reports/`: Interim/final reports.

## Branches
- `main`: Stable, merged work.
- `task-1`: EDA & Git setup.
- `task-2`: DVC pipeline.

## CI/CD
Automated tests via GitHub Actions (linting, EDA smoke tests).

## Data Source
Historical claims data from ACIS (columns: UnderwrittenCoverID, TotalClaims, etc.).

## Learning Outcomes
- EDA insights on loss ratios, outliers.
- Reproducible pipelines with DVC.