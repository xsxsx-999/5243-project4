
import re
def sanitize_columns(df):
    """Replace special characters in column names for LightGBM compatibility."""
    df = df.copy()
    df.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", col).strip("_") for col in df.columns]
    return df
