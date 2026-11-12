import json
import zipfile
from pathlib import Path
from io import BytesIO

import joblib
import pandas as pd
import streamlit as st

from src.config import DATA_PATH, FEATURES, TARGETS, MODEL_DIR, REPORT_DIR, FIGURE_DIR
from src.data_utils import load_data, validate_data
from src.modeling import run_training

st.set_page_config(page_title="Student Health ML", page_icon="🎓", layout="wide")

st.title("🎓 Student Health & Academic Performance Analyzer")
#st.caption("Beginner-friendly machine learning application | Kaggle student-health dataset")

@st.cache_data
def get_data():
    return load_data()

try:
    df = get_data()
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

page = st.sidebar.radio(
    "Navigation",
    ["Project Overview", "Explore Dataset", "Train Models", "Predict Student", "Reports"]
)

if page == "Project Overview":
    st.subheader("Project objectives")
    st.write("This application predicts three student outcomes using three separate machine learning models.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Academic score", "Regression")
    c2.metric("Failure risk", "Binary classification")
    c3.metric("Burnout level", "Multiclass classification")
    st.divider()
    a, b, c = st.columns(3)
    a.metric("Dataset rows", f"{len(df):,}")
    b.metric("Dataset columns", len(df.columns))
    c.metric("Input features", len(FEATURES))
    st.subheader("Six concepts demonstrated")
    concepts = [
        "1. Data loading and exploratory data analysis",
        "2. Preprocessing: imputation, encoding and scaling",
        "3. Feature and target selection",
        "4. Train-test split and cross-validation",
        "5. Model training and comparison",
        "6. Model evaluation and leakage screening",
    ]
    for item in concepts:
        st.write("✓ " + item)
    st.info("The original Kaggle target-generation methodology is not yet verified. The leakage screen is a warning tool, not proof of how labels were created.")

elif page == "Explore Dataset":
    st.subheader("Dataset preview")
    st.dataframe(df.head(20),width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Duplicate rows", int(df.duplicated().sum()))
    st.subheader("Column information")
    info = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[x].dtype) for x in df.columns],
        "Missing values": [int(df[x].isna().sum()) for x in df.columns],
        "Unique values": [int(df[x].nunique()) for x in df.columns]
    })
    st.dataframe(info, width="stretch")
    st.subheader("Target distributions")
    selected_target = st.selectbox("Choose target", list(TARGETS.values()))
    if selected_target == TARGETS["performance"]:
        st.line_chart(df[selected_target].value_counts().sort_index())
        st.write(df[selected_target].describe())
    else:
        st.bar_chart(df[selected_target].value_counts())
        st.dataframe(
            (df[selected_target].value_counts(normalize=True) * 100)
            .rename("percentage").to_frame(),
            width="stretch"
        )

elif page == "Train Models":
    st.subheader("Train and compare three models")
    st.write("For each target, the app compares a simple baseline with two standard models using 3-fold cross-validation. The best model is evaluated on a separate test set.")
    issues = validate_data(df)
    if issues:
        st.error("Dataset validation found issues:")
        for issue in issues:
            st.write("- " + issue)
    else:
        st.success("Basic dataset validation passed.")
    if st.button("Train all three models and generate reports", type="primary"):
        with st.spinner("Training models and generating reports. This may take a few minutes..."):
            try:
                results = run_training(df)
                st.success("Training completed. Models and reports have been saved.")
                for key, result in results.items():
                    st.markdown(f"**{result['target']}** — {result['selected_model']}")
                    st.json(result["test_metrics"])
            except Exception as exc:
                st.exception(exc)

elif page == "Predict Student":
    st.subheader("Enter student information")
    st.caption("The three predictions are produced independently from the same input features.")
    left, right = st.columns(2)
    with left:
        age = st.number_input("Age", min_value=10, max_value=100, value=20)
        gender = st.selectbox("Gender", sorted(df["Gender"].dropna().unique()))
        education = st.selectbox("Education level", sorted(df["Education_Level"].dropna().unique()))
        social = st.number_input("Daily social media hours", min_value=0.0, max_value=24.0, value=3.0, step=0.1)
        ai = st.number_input("Daily AI tool usage hours", min_value=0.0, max_value=24.0, value=1.0, step=0.1)
        sleep = st.number_input("Sleep hours", min_value=0.0, max_value=24.0, value=7.0, step=0.1)
    with right:
        activity = st.number_input("Physical activity hours", min_value=0.0, max_value=24.0, value=1.0, step=0.1)
        mental = st.slider("Mental health score", 0.0, 100.0, 70.0)
        physical = st.slider("Physical health score", 0.0, 100.0, 75.0)
        isolation = st.slider("Social isolation score", 0.0, 10.0, 3.0)

    if st.button("Predict student outcomes", type="primary"):
        model_paths = {
            "regression": MODEL_DIR / "regression_model.joblib",
            "binary": MODEL_DIR / "binary_model.joblib",
            "multiclass": MODEL_DIR / "multiclass_model.joblib"
        }
        missing = [str(p.name) for p in model_paths.values() if not p.exists()]
        if missing:
            st.warning("Train the models first. Missing: " + ", ".join(missing))
        else:
            row = pd.DataFrame([{
                "Age": age, "Gender": gender, "Education_Level": education,
                "Daily_Social_Media_Hours": social,
                "Daily_AI_Tool_Usage_Hours": ai, "Sleep_Hours": sleep,
                "Physical_Activity_Hours": activity,
                "Mental_Health_Score": mental, "Physical_Health_Score": physical,
                "Social_Isolation_Score": isolation
            }], columns=FEATURES)
            performance = float(joblib.load(model_paths["regression"]).predict(row)[0])
            failure_model = joblib.load(model_paths["binary"])
            failure = int(failure_model.predict(row)[0])
            burnout = str(joblib.load(model_paths["multiclass"]).predict(row)[0])
            x, y, z = st.columns(3)
            x.metric("Predicted academic score", f"{max(0, min(100, performance)):.2f} / 100")
            y.metric("Academic failure risk", "At Risk" if failure == 1 else "No Risk")
            z.metric("Predicted burnout level", burnout)
            if hasattr(failure_model, "predict_proba"):
                class_order = list(failure_model.classes_)
                if 1 in class_order:
                    probability = failure_model.predict_proba(row)[0][class_order.index(1)]
                    st.write(f"Estimated failure-risk probability: **{probability:.1%}**")

elif page == "Reports":
    st.subheader("Generated reports")
    st.write("Reports are created after you train the models.")
    report_files = sorted([p for p in REPORT_DIR.rglob("*") if p.is_file()])
    if not report_files:
        st.info("No reports yet. Open Train Models and train all three models.")
    else:
        for path in report_files:
            relative = path.relative_to(REPORT_DIR)
            st.write(f"📄 `{relative}`")
            if path.suffix == ".csv":
                try:
                    st.dataframe(pd.read_csv(path), width="stretch")
                except Exception:
                    pass
            elif path.suffix == ".txt":
                st.text(path.read_text(encoding="utf-8"))
            elif path.suffix == ".json" and path.name == "model_metrics.json":
                st.json(json.loads(path.read_text(encoding="utf-8")))
            elif path.suffix == ".png":
                st.image(str(path), caption=path.name, width="stretch")

        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in report_files:
                archive.write(path, arcname=str(path.relative_to(REPORT_DIR)))
        st.download_button(
            "Download all reports (ZIP)",
            data=buffer.getvalue(),
            file_name="student_health_reports.zip",
            mime="application/zip"
        )
