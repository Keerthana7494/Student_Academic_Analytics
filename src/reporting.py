import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

from src.config import (
    FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    TARGETS, RANDOM_STATE, REPORT_DIR, FIGURE_DIR
)


def create_dataset_reports(df):
    issues = []
    required = ["Student_ID"] + FEATURES + list(TARGETS.values())
    missing = sorted(set(required) - set(df.columns))
    if missing:
        return [f"Missing required columns: {missing}"]

    summary = pd.DataFrame({
        "column": df.columns,
        "data_type": [str(df[c].dtype) for c in df.columns],
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns]
    })
    summary.to_csv(REPORT_DIR / "dataset_summary.csv", index=False)
    df.isna().sum().rename("missing_count").to_csv(REPORT_DIR / "missing_values.csv")
    df.describe(include="all").transpose().to_csv(REPORT_DIR / "descriptive_statistics.csv")

    distributions = []
    for target in TARGETS.values():
        counts = df[target].value_counts(dropna=False)
        for label, count in counts.items():
            distributions.append({
                "target": target, "class_or_value": str(label),
                "count": int(count), "percentage": float(count / len(df) * 100)
            })
    pd.DataFrame(distributions).to_csv(
        REPORT_DIR / "target_distributions.csv", index=False
    )

    numeric = df.select_dtypes(include="number")
    numeric.corr(method="spearman").to_csv(
        REPORT_DIR / "numeric_correlations_spearman.csv"
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(numeric.corr(method="spearman"), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Spearman correlation (numeric columns)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=140)
    plt.close(fig)

    for target in [TARGETS["failure_risk"], TARGETS["burnout"]]:
        fig, ax = plt.subplots(figsize=(7, 4))
        df[target].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"Target distribution: {target}")
        ax.set_xlabel(target)
        ax.set_ylabel("Number of students")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"{target.lower()}_distribution.png", dpi=140)
        plt.close(fig)

    # Leakage screen: association and shallow-tree reconstruction are warning signals,
    # not proof that the dataset author used these features to create a target.
    rows = []
    numeric_features = [c for c in NUMERIC_FEATURES if c in df.columns]
    for target in TARGETS.values():
        if target == TARGETS["performance"]:
            y = df[target]
            task = "regression"
            corr = df[numeric_features + [target]].corr(method="spearman")[target].drop(target)
            for feature, value in corr.items():
                rows.append({
                    "target": target, "check": "Spearman correlation",
                    "feature": feature, "value": float(value),
                    "interpretation": "Association only; investigate target definition"
                })
        else:
            y = df[target]
            task = "classification"
            # For binary/multiclass targets, use per-feature group means as a simple
            # descriptive screen. The model-based reconstruction is reported below.
            for feature in numeric_features:
                grouped = df.groupby(target)[feature].mean()
                spread = float(grouped.max() - grouped.min())
                rows.append({
                    "target": target, "check": "Difference in class means",
                    "feature": feature, "value": spread,
                    "interpretation": "Class separation only; not proof of leakage"
                })

        # Shallow tree cross-validation gives a basic rule-reconstruction warning.
        X = df[FEATURES]
        if task == "regression":
            estimator = DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=RANDOM_STATE)
            cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
            scoring = "r2"
        else:
            estimator = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=RANDOM_STATE, class_weight="balanced")
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
            scoring = "f1" if target == TARGETS["failure_risk"] else "f1_macro"

        prep = ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC_FEATURES),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES)
        ])
        pipe = Pipeline([("prep", prep), ("tree", estimator)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1)
        rows.append({
            "target": target, "check": "Depth-3 decision-tree CV reconstruction",
            "feature": "ALL_INPUT_FEATURES", "value": float(scores.mean()),
            "interpretation": f"CV {scoring}; high score warrants investigation, not proof of leakage"
        })

    pd.DataFrame(rows).to_csv(REPORT_DIR / "leakage_screen.csv", index=False)
    return issues


def create_model_reports(results, df):
    lines = [
        "STUDENT HEALTH & ACADEMIC PERFORMANCE - MODEL SUMMARY",
        "=" * 58,
        f"Dataset rows: {len(df)}",
        f"Input features used: {len(FEATURES)}",
        "",
        "IMPORTANT: Kaggle target-generation methodology has not been independently verified.",
        "Statistical association or successful target reconstruction is not proof of leakage.",
        ""
    ]
    for task, result in results.items():
        lines.extend([
            f"Target: {result['target']}",
            f"Task: {result['task']}",
            f"Selected model: {result['selected_model']}",
            f"Cross-validation selection metric: {result['selection_metric']}",
            f"Cross-validation mean: {result['cv_mean']:.4f}",
            f"Test metrics: {json.dumps(result['test_metrics'], ensure_ascii=False)}",
            ""
        ])
    (REPORT_DIR / "final_summary.txt").write_text("\n".join(lines), encoding="utf-8")
