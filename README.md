# Student Health & Academic Performance Prediction

A beginner-friendly machine learning project using a Kaggle student-health dataset.

## What the application predicts

1. **Academic Performance Score** – regression.
2. **Academic Failure Risk** – binary classification (0 = No Risk, 1 = Risk).
3. **Burnout Level** – multiclass classification (Low, Moderate, High, Severe).

## Six ML concepts demonstrated

1. Data loading and exploratory data analysis (EDA).
2. Data preprocessing: missing-value imputation, one-hot encoding and scaling.
3. Feature and target selection.
4. Train-test split and cross-validation.
5. Model training and model comparison.
6. Model evaluation and a basic target-leakage screening.

## Project structure

```text
student_health_ml_project/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── data/
│   └── raw/
│       └── student_health.csv
├── src/
│   ├── config.py
│   ├── data_utils.py
│   ├── modeling.py
│   └── reporting.py
├── artifacts/
│   └── models/             # Saved models after training
└── reports/
    ├── figures/            # Charts created during training
    ├── dataset_summary.csv
    ├── missing_values.csv
    ├── target_distributions.csv
    ├── numeric_correlations.csv
    ├── leakage_screen.csv
    ├── model_comparison.csv
    ├── model_metrics.json
    └── final_summary.txt
```

## Run on Windows (VS Code)

### 1. Open the project folder

Open `student_health_ml_project` in VS Code.

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

### 3. Install packages

```bash
pip install -r requirements.txt
```

### 4. Start the Streamlit application

```bash
streamlit run app.py
```

The app opens in your browser. Use the **Train Models** page/button to train the three models and generate reports.

You can also train from the terminal:

```bash
python train.py
```

## Reports

The `reports/` folder is populated when model training runs. The Streamlit app also lets you preview and download the generated reports.

## Important data-quality note

The target-generation methodology of the original Kaggle dataset has not yet been independently verified. The leakage screen checks statistical relationships and whether shallow decision trees can reconstruct target labels from the selected input features. These checks can identify warning signs, but they cannot prove how the Kaggle author created the labels.

Do not describe model performance on this dataset as proof of real-world student-health prediction. First verify whether the dataset contains measured observations or synthetic/simulated records.

## Interview explanation

"I built a three-task machine learning application. I first explored and validated the dataset, then separated input features from three target variables. I used Scikit-learn pipelines so preprocessing is learned from training data only. I compared a simple baseline with a machine learning model using cross-validation, evaluated the selected model on a held-out test set, and generated reports. Streamlit provides a simple interface for training, prediction and report viewing."
