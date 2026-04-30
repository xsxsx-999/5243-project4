## Introduction

Designed for STATGR5243, this project establishes an end-to-end machine learning framework to decode mortgage application outcomes using the Home Mortgage Disclosure Act (HMDA) database. We developed a robust analytical pipeline—spanning from sophisticated feature engineering to predictive modeling—to isolate the critical drivers of lending decisions. The project culminates in a comprehensive analysis of applicant demographics and loan characteristics, providing data-driven insights into the complex factors that influence mortgage approval and denial in the U.S. housing market.

## How to Run the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/xsxsx-999/5243-project4.git
   cd 5243-project4
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Launch the files:
   ```bash
   jupyter notebook
   ```

## Project Structure
```
5243-project4/
├── README.md                          # Project documentation
├── requirements.txt                    # Notebook/analysis dependencies
├── report.ipynb                        # Main report notebook
├── report.pdf                          # Rendered report
├── appendix.ipynb                      # Appendix / extra analyses
├── app/                                # Shiny for Python web app
│   ├── app.py                          # App entry point
│   ├── helpers.py                      # UI + explanation helpers
│   ├── requirements.txt                # App runtime dependencies
│   ├── rsconnect-python/               # Deployment metadata
│   └── artifacts/                      # Exported artifacts used by the app
├── data/                               # Data prep inputs/outputs
│   ├── data_prep.ipynb                 # Cleaning + preprocessing notebook
│   ├── HMDA_CA_TX_2019_2024_Master.csv # Raw master dataset
│   ├── X_train_ohe.csv                 # Encoded training features (derived)
│   ├── X_test_ohe.csv                  # Encoded test features (derived)
│   ├── y_train.csv                     # Training labels (derived)
│   ├── y_test.csv                      # Test labels (derived)
│   └── data_prep_plots/                # Figures generated during preprocessing
├── eda_insights/                       # EDA notebook + figures
│   ├── eda_insights.ipynb
│   └── eda_insights_plots/
└── model/                              # Modeling notebook + artifacts/metrics/plots
    ├── modeling.ipynb
    ├── final_model.pkl                 # Trained model artifact
    ├── feature_columns.json            # Feature schema used for inference
    ├── test_metrics.csv                # Evaluation metrics
    ├── cv_summary.csv                  # Cross-validation summary
    ├── subgroup_rates.csv              # Group-level fairness/approval rates
    └── (plots + SHAP outputs)          # e.g., calibration curves, SHAP summary, PDP
```

