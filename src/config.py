from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "student_health.csv"
MODEL_DIR = ROOT_DIR / "artifacts" / "models"
REPORT_DIR = ROOT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.20

NUMERIC_FEATURES = [
    "Age",
    "Daily_Social_Media_Hours",
    "Daily_AI_Tool_Usage_Hours",
    "Sleep_Hours",
    "Physical_Activity_Hours",
    "Mental_Health_Score",
    "Physical_Health_Score",
    "Social_Isolation_Score",
]
CATEGORICAL_FEATURES = ["Gender", "Education_Level"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGETS = {
    "performance": "Academic_Performance_Score",
    "failure_risk": "Academic_Failure_Risk",
    "burnout": "Burnout_Level",
}

for folder in [MODEL_DIR, REPORT_DIR, FIGURE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
