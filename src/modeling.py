import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
    confusion_matrix, average_precision_score
)

from src.config import (
    FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGETS,
    RANDOM_STATE, TEST_SIZE, MODEL_DIR, REPORT_DIR
)
from src.reporting import create_dataset_reports, create_model_reports


def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES)
    ])


def get_models(task):
    if task == "regression":
        return {
            "Dummy Baseline": DummyRegressor(strategy="mean"),
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, min_samples_leaf=2,
                random_state=RANDOM_STATE, n_jobs=-1
            )
        }

    return {
        "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=1500, class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=2,
            class_weight="balanced", random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }


def make_pipeline(model):
    return Pipeline([
        ("preprocessing", build_preprocessor()),
        ("model", model)
    ])


def train_one_target(df, task, target):
    X = df[FEATURES].copy()
    y = df[target].copy()

    stratify = y if task in ["binary", "multiclass"] else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=stratify
    )

    if task == "regression":
        cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scoring = "neg_mean_absolute_error"
    else:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scoring = "f1" if task == "binary" else "f1_macro"

    comparison = []
    best_name, best_score, best_model = None, -np.inf, None

    for name, estimator in get_models(task).items():
        pipeline = make_pipeline(estimator)
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1
        )
        mean_score = float(scores.mean())
        comparison.append({
            "target": target,
            "model": name,
            "selection_metric": scoring,
            "cv_mean": mean_score,
            "cv_std": float(scores.std())
        })
        if mean_score > best_score:
            best_name, best_score, best_model = name, mean_score, pipeline

    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)

    result = {
        "target": target,
        "task": task,
        "selected_model": best_name,
        "selection_metric": scoring,
        "cv_mean": best_score,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": FEATURES
    }

    if task == "regression":
        result["test_metrics"] = {
            "MAE": float(mean_absolute_error(y_test, predictions)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "R2": float(r2_score(y_test, predictions))
        }
    else:
        result["test_metrics"] = {
            "Accuracy": float(accuracy_score(y_test, predictions)),
            "Balanced_Accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "Precision_Macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
            "Recall_Macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
            "F1_Macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "F1_Weighted": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
            "Labels": [str(x) for x in sorted(pd.Series(y).unique())],
            "Confusion_Matrix": confusion_matrix(
                y_test, predictions, labels=sorted(pd.Series(y).unique())
            ).tolist(),
            "Classification_Report": classification_report(
                y_test, predictions, output_dict=True, zero_division=0
            )
        }
        if task == "binary" and hasattr(best_model, "predict_proba"):
            class_list = list(best_model.classes_)
            if 1 in class_list:
                risk_probability = best_model.predict_proba(X_test)[:, class_list.index(1)]
                result["test_metrics"]["PR_AUC_Average_Precision"] = float(
                    average_precision_score(y_test, risk_probability)
                )

    joblib.dump(best_model, MODEL_DIR / f"{task}_model.joblib")
    return result, comparison, best_model


def run_training(df):
    issues = create_dataset_reports(df)
    if issues:
        raise ValueError("\n".join(issues))

    all_results = {}
    all_comparisons = []
    model_specs = [
        ("regression", TARGETS["performance"]),
        ("binary", TARGETS["failure_risk"]),
        ("multiclass", TARGETS["burnout"])
    ]

    for task, target in model_specs:
        result, comparison, model = train_one_target(df, task, target)
        all_results[task] = result
        all_comparisons.extend(comparison)

    pd.DataFrame(all_comparisons).to_csv(
        REPORT_DIR / "model_comparison.csv", index=False
    )
    with open(REPORT_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    create_model_reports(all_results, df)
    return all_results
