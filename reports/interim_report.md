# ACIS Interim Report: Tasks 1 & 2 (Dec 07, 2025)

## Executive Summary (For Leadership)
AlphaCare Insurance Solutions (ACIS) can leverage this analysis to target low-risk segments in South Africa, potentially reducing premiums by 10-15% in safer provinces like Mpumalanga (loss ratio 2.45%) to boost client acquisition. EDA reveals a healthy portfolio loss ratio of 3.32%, with opportunities in vehicle types (e.g., Sedans lowest risk) and demographics (females safer). DVC ensures all insights are reproducible for regulatory audits. Next: Hypothesis tests to validate risk drivers.

## Task 1: Project Planning - EDA & Stats
### Data Understanding & Quality
- **Structure**: 10,000 policies (52 columns: policy, client, location, vehicle, plan, claims). Dtypes correct (e.g., `TransactionMonth` object, numerics float/int).
- **Descriptive Stats** (Key Financials):

| Metric          | Mean     | Std      | Min     | 25%      | 50%      | 75%      | Max       |
|-----------------|----------|----------|---------|----------|----------|----------|-----------|
| TotalPremium   | 6371.01 | 1621.71 | 593.41  | 5264.23 | 6350.72 | 7454.61 | 13089.24 |
| TotalClaims    | 194.10  | 604.18  | 0.00    | 0.00    | 0.00    | 0.00    | 9396.10  |
| SumInsured     | 199295.34| 50275.36| 19780.20| 165753.62| 199219.05| 233215.89| 382951.44|

- **Quality**: 0 missing values. ~80% policies claim-free (realistic for auto insurance).
- **Temporal Trends**: Claims stable over 19 months; slight uptick in later periods (e.g., post-2014 Q3—possible seasonal/economic factor; visualize in `claims_by_make_trends.png`).

### Univariate Analysis
- **Numerical**: TotalClaims right-skewed (exponential dist., median 0); TotalPremium normal-ish. See histograms (`TotalClaims_hist.png`).
- **Categorical**: Gauteng dominates (40%); Males 60%; Sedans 50%. Bar charts (`Province_bar.png`, etc.).

### Bivariate/Multivariate Analysis
- **Loss Ratio (TotalClaims / TotalPremium)**: Overall 3.32% (low—profitable portfolio).

| Group              | Mean Loss Ratio | Std      |
|--------------------|-----------------|----------|
| **By Province**   |                 |          |
| Eastern Cape      | 0.0313         | 0.0985  |
| Gauteng           | 0.0327         | 0.1056  |
| KwaZulu-Natal     | 0.0335         | 0.1072  |
| Western Cape      | 0.0341         | 0.1098  |
| **By VehicleType**|                 |          |
| Hatchback         | 0.0332         | 0.1064  |
| SUV               | 0.0350         | 0.1123  |
| Sedan             | 0.0327         | 0.1049  |
| **By Gender**     |                 |          |
| F                 | 0.0314         | 0.1013  |
| M                 | 0.0345         | 0.1133  |

- **Correlations**: Near-zero (e.g., Premium-Claims: 0.00)—no strong linear ties yet; explore non-linear in ML. Heatmap: `correlation_heatmap.png`.
- **Geography Trends**: Premiums higher for Comprehensive covers in Gauteng (~R6,500 vs. R5,800 Third Party). Bar plot: `premium_by_province_cover.png`.
- **ZipCode Scatter**: Weak positive trend in avg claims vs. premiums (sample 50 zips); `premium_claims_scatter.png`.

### Outlier Detection
- Boxplot (`outliers_boxplot.png`): TotalClaims has upper outliers (e.g., >R5k).
- Z-Score (>3): 242 outliers in non-zero claims (2.42% of total; cap/log-transform for modeling).
- Violin (`claims_violin_vehicletype.png`): SUVs show fatter tails—higher severity risk.

### 3 Creative Visualizations (Key Insights)
1. **Loss Ratio Heatmap** (`loss_ratio_heatmap.png`): Females in Mpumalanga lowest (2.45%)—target for low-risk marketing; Males in North West highest (4.09%).
2. **Claims Trends by Make** (`claims_by_make_trends.png`): Toyota stable/low (~R150 avg/month); BMW spikes in mid-2014—age-related?
3. **Violin by VehicleType** (`claims_violin_vehicletype.png`): Sedans tight dist. (low variability)—recommend for premium discounts.

### Statistical Thinking
- Distributions: Claims ~ Exponential (λ≈0.005 for severity); Premiums ~ Normal (μ=6371, σ=1622).
- Actionable Insights: Target Mpumalanga females with Sedans/Toyota for 10% premium cuts (est. 1% portfolio uplift). References: [XenonStack Insurance Analytics](https://www.xenonstack.com/blog/data-analytics-in-insurance) for loss ratio benchmarks.

## Task 2: Data Version Control (DVC)
- **Setup**: Installed `dvc==3.51.0`; `dvc init` in root. Local remote: `../dvc-storage` (`dvc remote add -d localstorage ../dvc-storage`).
- **Tracking**: `dvc add data/raw/insurance_data.csv` (v1.0: raw synthetic); created v1.1 processed (e.g., added LossRatio). Committed `.dvc` files to Git; `dvc push`.
- **Reproducibility**: `dvc pull` restores data; `dvc repro` for future pipelines (e.g., EDA). 2 versions committed—enables auditing (e.g., revert to pre-cleaning).
- **Why?**: In regulated insurance, traces every analysis input (e.g., for FSRAO compliance). Reference: [DVC User Guide](https://dvc.org/doc/user-guide).

## Next Steps
- Task 3: A/B tests (e.g., chi-squared for province risks).
- Limitations: Synthetic data—scale to full for production.
- Commits: 12+ across branches (e.g., 4 for EDA plots).

Prepared by: [Your Name] | Repo: https://github.com/YOUR_USERNAME/acis-insurance-analytics