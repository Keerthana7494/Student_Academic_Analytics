import pandas as pd
from src.config import DATA_PATH, FEATURES, TARGETS

REQUIRED_COLUMNS = ["Student_ID"] + FEATURES + list(TARGETS.values())


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Dataset has no rows.")
    return df


def validate_data(df):
    issues = []
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        return [f"Missing columns: {missing_columns}"]
    if df["Student_ID"].duplicated().any():
        issues.append("Student_ID contains duplicate values.")
    if df[TARGETS.values()].isna().any().any():
        issues.append("One or more target columns contain missing values.")
    if not df["Academic_Failure_Risk"].dropna().isin([0, 1]).all():
        issues.append("Academic_Failure_Risk should contain only 0 and 1.")
    if not df["Academic_Performance_Score"].dropna().between(0, 100).all():
        issues.append("Academic_Performance_Score has values outside 0-100.")
    return issues
