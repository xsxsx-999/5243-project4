# Mortgage Approval Prediction — STAT GR5243 Project 4

**Team 23:** Veronica Joe (jj3470), Crystal Guo (lg3434), Shuxuan Xu (sx2412), Fangyi Lin (fl2748)

## Introduction

Designed for STAT GR5243, this project establishes an end-to-end machine learning framework to decode mortgage application outcomes using the Home Mortgage Disclosure Act (HMDA) database. We developed a robust analytical pipeline — spanning from sophisticated feature engineering to predictive modeling — to isolate the critical drivers of lending decisions. The project culminates in a comprehensive analysis of applicant demographics and loan characteristics, providing data-driven insights into the complex factors that influence mortgage approval and denial in the U.S. housing market.

The final calibrated LightGBM model is deployed as an interactive R Shiny app:
**[Mortgage Approval Estimator](https://crystalguo.shinyapps.io/mortgage-approval-estimator/)**

## How to Run the Code

1. Clone the repository:
   ```bash
   git clone https://github.com/xsxsx-999/5243-project4.git
   cd 5243-project4
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate     # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. Launch the notebooks:
   ```bash
   jupyter notebook
   ```

### Recommended notebook execution order

1. `data/data_prep.ipynb` — cleans raw HMDA data and produces the encoded train/test CSVs
2. `eda_insights/eda_insights.ipynb` — feature signal analysis and SVD diagnostics
3. `model/modeling.ipynb` — model training, calibration, SHAP, and threshold analysis
4. `report.ipynb` — final write-up integrating all sections
5. `appendix.ipynb` — supplementary analyses


## Project Structure

```
5243-project4/
├── README.md
├── requirements.txt
├── report.ipynb              # Main report notebook
├── appendix.ipynb            # Supplementary analyses
│
├── data/                     # Raw data + preprocessing pipeline + encoded CSVs
├── eda_insights/             # Feature signal, redundancy, and SVD analysis
├── model/                    # Modeling notebook + saved metrics in results/
└── app/                      # Shiny for Python deployment + model artifacts
```

## Key Results

| Metric | Value |
|---|---|
| Final model | Calibrated LightGBM (93 features, post-leakage adjustment) |
| Test ROC-AUC | 0.858 |
| Test PR-AUC | 0.946 |
| Test Brier score | 0.095 |
| Recommended operating threshold | 0.6 (precision 0.75, recall 0.58, flagged rate 15.6%) |

Top predictors by SHAP: `debt_to_income_ratio_ord`, `aus_grouped_Standard_AUS`, `loan_to_value_ratio`, `loan_amount`, `property_value`.

## Team

| Member | Contribution |
|---|---|
| Shuxuan Xu (sx2412) | Data preprocessing, cleaning, feature engineering |
| Fangyi Lin (fl2748) | Exploratory data analysis and visualization |
| Veronica Joe (jj3470) | Modeling-stage EDA and model selection support |
| Crystal Guo (lg3434) | Model building, evaluation, and deployment |

## References

- Federal Financial Institutions Examination Council. (2025). *HMDA Data Browser* [Data set]. CFPB. https://ffiec.cfpb.gov/data-browser/
- U.S. Census Bureau. (2025). *State Population Totals: 2020–2025*. https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-total.html
