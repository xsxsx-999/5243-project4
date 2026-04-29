"""Helper functions for the mortgage approval app."""

import numpy as np
import pandas as pd
import shap

# Plain-language names for top features (shown in reasons)
FEATURE_LABELS = {
    "debt_to_income_ratio_ord":              "Debt-to-income ratio",
    "aus_grouped_Standard_AUS":              "Underwriting system",
    "loan_to_value_ratio":                   "Loan-to-value ratio",
    "loan_amount":                           "Loan amount",
    "property_value":                        "Property value",
    "income":                                "Annual income",
    "loan_term":                             "Loan term",
    "dti_missing_flag":                      "DTI information availability",
    "loan_purpose_grouped_Standard_Purpose": "Loan purpose",
    "loan_type_1":                           "Loan type (Conventional)",
    "loan_type_2":                           "Loan type (FHA)",
    "loan_type_3":                           "Loan type (VA)",
    "loan_type_4":                           "Loan type (USDA)",
    "submission_of_application_1":           "Application submission method",
    "applicant_age_grouped__44":             "Applicant age (over 44)",
    "applicant_age_grouped_25_44":           "Applicant age (25-44)",
    "applicant_age_grouped__25":             "Applicant age (under 25)",
    "is_joint_application_1":                "Joint application",
    "derived_sex_Joint":                     "Joint applicants",
    "derived_sex_Female":                    "Female applicant",
    "derived_sex_Male":                      "Male applicant",
    "derived_ethnicity_Hispanic_or_Latino":  "Hispanic or Latino background",
    "derived_ethnicity_Not_Hispanic_or_Latino": "Non-Hispanic background",
    "tract_minority_population_percent":     "Neighborhood minority composition",
    "tract_to_msa_income_percentage":        "Neighborhood income level",
    "covid_phase_Peak_Pandemic":             "Application during peak pandemic",
    "covid_phase_Post_Pandemic":             "Application after pandemic",
}


def humanize_feature(name):
    """Get a plain-language label for a feature, falling back to the name."""
    return FEATURE_LABELS.get(name, name.replace("_", " ").title())


def get_top_reasons(shap_row, feature_names, feature_values, top_n=3):
    """
    Translate one row of SHAP values into top N plain-language reasons.
    
    Returns a list of dicts: [{direction: 'positive'/'negative', text: '...'}]
    """
    # Sort by absolute SHAP value
    abs_shap = np.abs(shap_row)
    top_idx = np.argsort(-abs_shap)[:top_n]
    
    reasons = []
    for i in top_idx:
        feat = feature_names[i]
        val = feature_values[i]
        shap_val = shap_row[i]
        label = humanize_feature(feat)
        
        direction = "positive" if shap_val > 0 else "negative"
        impact = "supporting approval" if shap_val > 0 else "reducing approval chances"
        
        # Format the value contextually
        if "ratio" in feat.lower() and val < 100:
            val_str = f"{val:.0f}%" if val > 1 else f"{val:.0%}"
        elif feat in ("loan_amount", "property_value", "income"):
            # These are log-scaled in the data; show in dollars
            val_str = f"${np.exp(val):,.0f}"
        elif val == 0 or val == 1:
            val_str = "yes" if val == 1 else "no"
        else:
            val_str = f"{val:.1f}" if isinstance(val, float) else str(val)
        
        reasons.append({
            "direction": direction,
            "text": f"{label} ({val_str}) is {impact}",
        })
    
    return reasons


def get_verdict(prob, low=0.40, high=0.70):
    """Convert probability to plain-language verdict."""
    if prob >= high:
        return "Likely approved", "green"
    elif prob >= low:
        return "On the fence", "orange"
    else:
        return "Likely denied", "red"


def reasons_to_html(reasons):
    """Format reasons as HTML for display."""
    html = "<ul style='list-style: none; padding-left: 0;'>"
    for r in reasons:
        icon = "✓" if r["direction"] == "positive" else "✗"
        color = "#2e7d32" if r["direction"] == "positive" else "#c62828"
        html += f"<li style='margin: 8px 0;'><span style='color: {color}; font-weight: bold;'>{icon}</span> {r['text']}</li>"
    html += "</ul>"
    return html