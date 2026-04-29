"""
Mortgage Approval Estimator — Shiny for Python app.
Two tabs: Batch scoring + Individual estimate.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import shap
from shiny import App, ui, render, reactive
from shinywidgets import output_widget, render_widget
import plotly.graph_objects as go

from helpers import (
    humanize_feature,
    get_top_reasons,
    get_verdict,
    reasons_to_html,
    FEATURE_LABELS,
)

# -----------------------------------------------------------------------------
# Load artifacts
# -----------------------------------------------------------------------------
ART = Path(__file__).parent / "artifacts"

final_model     = joblib.load(ART / "final_model.pkl")
final_lgbm_raw  = joblib.load(ART / "final_lgbm_raw.pkl")
feature_columns = json.load(open(ART / "feature_columns.json"))
shap_values     = np.load(ART / "shap_values_test.npy")
test_df         = pd.read_parquet(ART / "test_predictions.parquet")

# Feature matrix only (drop the meta columns we added during export)
META_COLS = ["actual", "predicted_proba", "predicted_class", "ethnicity", "sex", "state"]
X_test = test_df[feature_columns].copy()

explainer = shap.TreeExplainer(final_lgbm_raw)


# -----------------------------------------------------------------------------
# Gauge plot helper
# -----------------------------------------------------------------------------
def make_gauge(prob):
    """Plotly gauge for approval probability."""
    color = "#2e7d32" if prob >= 0.70 else ("#f57c00" if prob >= 0.40 else "#c62828")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 48}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.8},
            "steps": [
                {"range": [0, 40],   "color": "#ffebee"},
                {"range": [40, 70],  "color": "#fff3e0"},
                {"range": [70, 100], "color": "#e8f5e9"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
        },
        title={"text": "Approval Probability", "font": {"size": 18}},
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
    return fig


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
app_ui = ui.page_navbar(
    
    # =========================================================================
    # TAB 1 — Batch scoring
    # =========================================================================
    ui.nav_panel(
        "Score Applications",
        ui.div(
            ui.h3("Batch Application Scoring"),
            ui.markdown(
                """
                **What this tab does:** Loads a sample of mortgage applications and 
                scores each one using our model. You'll see the original application 
                data plus three new columns: predicted approval probability, a 
                plain-language verdict, and the top reasons driving each prediction.
                
                *Click any row to see a detailed explanation for that application.*
                """
            ),
            ui.input_slider("n_rows", "Number of applications to display:", min=10, max=500, value=50, step=10),
            ui.output_data_frame("batch_table"),
            ui.hr(),
            ui.h4("Selected Application — Detail"),
            ui.output_ui("selected_detail"),
            style="padding: 20px;",
        ),
    ),
    
    # =========================================================================
    # TAB 2 — Individual estimator
    # =========================================================================
    ui.nav_panel(
        "Estimate My Approval",
        ui.div(
            ui.h3("Personal Mortgage Approval Estimator"),
            ui.markdown(
                "Fill in your application details below to see your estimated approval probability."
            ),
            ui.layout_columns(
                # ----- LEFT: Form -----
                ui.div(
                    ui.h4("Required Information"),
                    ui.input_numeric("loan_amount",    "Loan amount ($)",     value=400_000, min=10_000, max=5_000_000, step=10_000),
                    ui.input_numeric("property_value", "Property value ($)",  value=500_000, min=20_000, max=10_000_000, step=10_000),
                    ui.input_numeric("income",         "Annual income ($)",   value=120_000, min=10_000, max=5_000_000, step=5_000),
                    ui.input_slider("dti",             "Debt-to-income ratio (%)", min=0, max=70, value=30, step=1),
                    ui.input_slider("loan_term",       "Loan term (months)",  min=120, max=360, value=360, step=60),
                    ui.input_select("loan_purpose", "Loan purpose", choices={
                        "Standard": "Home purchase / Refinance",
                        "High_Risk": "Home improvement / Other",
                    }, selected="Standard"),
                    ui.input_select("loan_type", "Loan type", choices={
                        "1": "Conventional",
                        "2": "FHA",
                        "3": "VA",
                        "4": "USDA",
                    }, selected="1"),
                    ui.input_select("aus", "Underwriting system", choices={
                        "1": "Standard (Fannie/Freddie/FHA engine)",
                        "0": "Non-standard or exempt",
                    }, selected="1"),
                    
                    ui.hr(),
                    ui.tags.details(
                        ui.tags.summary(ui.h4("Optional Details", style="display: inline-block;")),
                        ui.input_select("joint", "Joint application?", choices={"1": "Yes", "0": "No"}, selected="0"),
                        ui.input_select("age", "Applicant age", choices={
                            "25_44": "25-44",
                            "_25":   "Under 25",
                            "_44":   "Over 44",
                        }, selected="25_44"),
                        ui.markdown(
                            """
                            ---
                            **Demographic information** (optional)
                            
                            *Disclaimer: We collect demographic data only to provide a 
                            comparison to similar applications and to support fairness 
                            auditing. The model was trained on historical data that 
                            includes these features, so they may influence predictions. 
                            You may leave these as default if you prefer.*
                            """
                        ),
                        ui.input_select("ethnicity", "Ethnicity", choices={
                            "Not_Hispanic_or_Latino":   "Not Hispanic or Latino",
                            "Hispanic_or_Latino":       "Hispanic or Latino",
                            "Joint":                    "Joint",
                            "Ethnicity_Not_Available":  "Prefer not to say",
                            "Free_Form_Text_Only":      "Other",
                        }, selected="Not_Hispanic_or_Latino"),
                        ui.input_select("sex", "Sex", choices={
                            "Male":              "Male",
                            "Female":            "Female",
                            "Joint":             "Joint",
                            "Sex_Not_Available": "Prefer not to say",
                        }, selected="Male"),
                    ),
                    ui.br(),
                    ui.input_action_button("estimate", "Estimate My Approval", class_="btn-primary btn-lg"),
                ),
                
                # ----- RIGHT: Results -----
                ui.div(
                    ui.output_ui("individual_result"),
                ),
                col_widths=[5, 7],
            ),
            style="padding: 20px;",
        ),
    ),
    
    # =========================================================================
    # TAB 3 — About
    # =========================================================================
    ui.nav_panel(
        "About This Tool",
        ui.div(
            ui.h3("About This Tool"),
            ui.markdown(
                """
                ### What this tool does
                
                This tool estimates the likelihood that a mortgage application 
                will be approved, based on patterns learned from historical lending data.
                
                ### How it was built
                
                We trained a gradient-boosting model (LightGBM) on **140,000 mortgage 
                applications** from California and Texas, submitted between 2019 and 
                2024. The model was carefully calibrated so that its predicted 
                probabilities reflect real-world approval rates.
                
                **Performance on held-out data:**
                - Correctly ranks applications about **86%** of the time (ROC-AUC = 0.86)
                - Well-calibrated probabilities (Brier score = 0.10)
                
                ### Important caveats
                
                - **This is a predictive tool, not a guarantee.** Lenders make the 
                  final decision based on a complete application review, including 
                  documents and considerations not captured in this model.
                - The model reflects historical patterns. If those patterns include 
                  biases, the model may reproduce them.
                - The model was trained on California and Texas data only. Predictions 
                  for other states may be less accurate.
                
                ### Fairness
                
                We audited this model across demographic groups (ethnicity, sex, age). 
                The model's predicted approval rates closely match the actual approval 
                rates in each subgroup, indicating that the model **mirrors existing 
                approval patterns rather than amplifying disparities**. The disparate 
                impact ratio across ethnicity groups remained above the 80% EEOC 
                fairness threshold.
                
                That said, real-world disparities exist in the underlying data, and 
                this model reflects them. We recommend using this tool as a starting 
                point for conversation, not as a final word.
                
                ### Data source
                
                Home Mortgage Disclosure Act (HMDA) public data, via the Consumer 
                Financial Protection Bureau (CFPB). [Learn more →](https://www.consumerfinance.gov/data-research/hmda/)
                """
            ),
            style="padding: 20px; max-width: 900px;",
        ),
    ),
    
    title="Mortgage Approval Estimator",
    id="navbar",
)


# -----------------------------------------------------------------------------
# Server
# -----------------------------------------------------------------------------
def server(input, output, session):
    
    # ----- Tab 1: Batch table -----
    @reactive.calc
    def batch_data():
        n = input.n_rows()
        sample = test_df.head(n).copy().reset_index(drop=True)
        
        # Add the three new columns
        sample["Approval Probability"] = (sample["predicted_proba"] * 100).round(1).astype(str) + "%"
        
        verdicts = sample["predicted_proba"].apply(lambda p: get_verdict(p)[0])
        sample["Verdict"] = verdicts
        
        # Top reason (just the strongest one for the table)
        sample_idx = sample.index.tolist()
        top_reasons_short = []
        for i in sample_idx:
            shap_row = shap_values[i]
            feat_vals = X_test.iloc[i].values
            reasons = get_top_reasons(shap_row, feature_columns, feat_vals, top_n=1)
            top_reasons_short.append(reasons[0]["text"])
        sample["Top Reason"] = top_reasons_short
        
        # Display columns
        display_cols = [
            "loan_amount", "property_value", "income",
            "debt_to_income_ratio_ord", "loan_to_value_ratio",
            "Approval Probability", "Verdict", "Top Reason",
        ]
        display_cols = [c for c in display_cols if c in sample.columns]
        return sample[display_cols]
    
    @output
    @render.data_frame
    def batch_table():
        return render.DataGrid(batch_data(), selection_mode="row", height="400px")
    
    @output
    @render.ui
    def selected_detail():
        sel = batch_table.cell_selection()
        if not sel or not sel["rows"]:
            return ui.markdown("*Select a row above to see detailed reasons.*")
        
        row_idx = list(sel["rows"])[0]
        prob = test_df.iloc[row_idx]["predicted_proba"]
        verdict, color = get_verdict(prob)
        
        shap_row = shap_values[row_idx]
        feat_vals = X_test.iloc[row_idx].values
        reasons = get_top_reasons(shap_row, feature_columns, feat_vals, top_n=5)
        
        return ui.div(
            ui.h5(f"Approval probability: {prob*100:.1f}% — ", 
                  ui.span(verdict, style=f"color: {color}; font-weight: bold;")),
            ui.h6("Top factors driving this prediction:"),
            ui.HTML(reasons_to_html(reasons)),
        )
    
    # ----- Tab 2: Individual estimator -----
    @reactive.calc
    @reactive.event(input.estimate)
    def individual_prediction():
        # Build a row matching the model's feature schema
        row = pd.Series(0.0, index=feature_columns)
        
        # Numeric features (note: loan_amount, property_value, income are log-scaled in training)
        row["loan_amount"]              = np.log(input.loan_amount())
        row["property_value"]           = np.log(input.property_value())
        row["income"]                   = np.log(input.income())
        row["loan_to_value_ratio"]      = (input.loan_amount() / input.property_value()) * 100
        row["debt_to_income_ratio_ord"] = input.dti()
        row["loan_term"]                = input.loan_term()
        row["dti_missing_flag"]         = 0
        
        # AUS
        row["aus_grouped_Standard_AUS"] = float(input.aus())
        
        # Loan type one-hot
        for t in ["1", "2", "3", "4"]:
            col = f"loan_type_{t}"
            if col in row.index:
                row[col] = 1.0 if input.loan_type() == t else 0.0
        
        # Loan purpose
        if "loan_purpose_grouped_Standard_Purpose" in row.index:
            row["loan_purpose_grouped_Standard_Purpose"] = 1.0 if input.loan_purpose() == "Standard" else 0.0
        
        # Joint
        if "is_joint_application_1" in row.index:
            row["is_joint_application_1"] = float(input.joint())
        
        # Age
        for age_cat in ["25_44", "_25", "_44"]:
            col = f"applicant_age_grouped_{age_cat}"
            if col in row.index:
                row[col] = 1.0 if input.age() == age_cat else 0.0
        
        # Ethnicity
        for eth in ["Hispanic_or_Latino", "Not_Hispanic_or_Latino", "Joint",
                    "Ethnicity_Not_Available", "Free_Form_Text_Only"]:
            col = f"derived_ethnicity_{eth}"
            if col in row.index:
                row[col] = 1.0 if input.ethnicity() == eth else 0.0
        
        # Sex
        for s in ["Male", "Female", "Joint", "Sex_Not_Available"]:
            col = f"derived_sex_{s}"
            if col in row.index:
                row[col] = 1.0 if input.sex() == s else 0.0
        
        # Reasonable defaults for tract-level features (medians from training data)
        defaults = {
            "tract_population":                  4500,
            "tract_minority_population_percent": 50,
            "ffiec_msa_md_median_family_income": 90000,
            "tract_to_msa_income_percentage":    100,
            "tract_owner_occupied_units":        1500,
            "tract_one_to_four_family_homes":    2000,
            "tract_median_age_of_housing_units": 30,
            "state_code_bin":                    0,  # CA default
            "covid_phase_Post_Pandemic":         1,
        }
        for k, v in defaults.items():
            if k in row.index:
                row[k] = v
        
        X = row.to_frame().T[feature_columns]
        prob = float(final_model.predict_proba(X)[:, 1][0])
        
        # SHAP for this prediction
        sv = explainer.shap_values(X)
        if isinstance(sv, list):  # binary classifier sometimes returns list
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv_row = sv[0] if sv.ndim > 1 else sv
        
        reasons = get_top_reasons(sv_row, feature_columns, X.iloc[0].values, top_n=3)
        return prob, reasons
    
    @output
    @render.ui
    def individual_result():
        if input.estimate() == 0:
            return ui.div(
                ui.markdown("*Fill in the form on the left and click **Estimate My Approval** to see your result.*"),
                style="padding-top: 100px; text-align: center; color: #666;",
            )
        
        prob, reasons = individual_prediction()
        verdict, color = get_verdict(prob)
        
        # Find similar applications in test set for the comparison line
        # (rough match: similar DTI bin and loan purpose)
        similar = test_df[
            (test_df["debt_to_income_ratio_ord"].between(input.dti() - 5, input.dti() + 5))
        ]
        similar_rate = similar["actual"].mean() if len(similar) > 50 else None
        
        comparison = ""
        if similar_rate is not None:
            comparison = f"Applications with a similar debt-to-income ratio were approved **{similar_rate*100:.0f}%** of the time in our data."
        
        return ui.div(
            output_widget("gauge_plot"),
            ui.h4(verdict, style=f"color: {color}; text-align: center;"),
            ui.hr(),
            ui.h5("Top 3 reasons:"),
            ui.HTML(reasons_to_html(reasons)),
            ui.hr(),
            ui.markdown(comparison) if comparison else ui.div(),
        )
    
    @output
    @render_widget
    def gauge_plot():
        if input.estimate() == 0:
            return go.Figure()
        prob, _ = individual_prediction()
        return make_gauge(prob)


app = App(app_ui, server)