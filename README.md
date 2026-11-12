# Student Health & Academic Performance Prediction System

A Machine Learning-based application that analyzes student-related factors and predicts academic performance, academic failure risk, and burnout levels.

## Project Overview

This project uses Machine Learning techniques to identify patterns in student data and generate predictions that can support early identification of academic and well-being concerns.

The system implements three predictive models:

- **Academic Performance Prediction:** Predicts students' academic performance scores.
- **Academic Failure Risk Prediction:** Classifies students based on their academic failure risk.
- **Burnout Level Prediction:** Identifies students' burnout levels.

A Streamlit web application provides an interface for entering student information and viewing prediction results.

## Key Features

- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Missing-value and data quality analysis
- Correlation analysis and visualization
- Multiple ML model comparison
- Model training and evaluation
- Regression, binary classification and multiclass classification
- Prediction through a Streamlit interface
- Trained model persistence using Joblib

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Web Application | Streamlit |
| Model Persistence | Joblib |

## Dataset

- **Total Records:** 15,000
- **Input Features:** 10
- **Prediction Targets:** 3

The dataset contains student-related information used to analyze academic performance, academic failure risk and burnout.

## Machine Learning Models

Three independent ML models are developed for different prediction tasks.

| Prediction Task | ML Approach | Selected Algorithm |
|---|---|---|
| Academic Performance Score | Regression | Random Forest Regressor |
| Academic Failure Risk | Binary Classification | Random Forest Classifier |
| Burnout Level | Multiclass Classification | Random Forest Classifier |

Multiple candidate algorithms are evaluated and compared before selecting the final models.

## Machine Learning Workflow

1. Data Collection and Loading
2. Data Validation and Quality Checks
3. Exploratory Data Analysis (EDA)
4. Missing Value Analysis
5. Descriptive Statistical Analysis
6. Correlation Analysis
7. Feature and Target Selection
8. Data Preprocessing
9. Feature Encoding and Scaling
10. Train-Test Split (80:20)
11. Cross-Validation (3-Fold)
12. Model Comparison and Selection
13. Model Training
14. Model Evaluation
15. Model Saving using Joblib
16. Prediction through Streamlit

## Model Evaluation

The trained models are evaluated using appropriate performance metrics.

| Prediction Task | Evaluation Metric | Test Result |
|---|---|---:|
| Academic Performance | R² Score | 0.285 |
| Academic Failure Risk | Accuracy | 93.43% |
| Academic Failure Risk | F1-Score (Risk Class) | 0.523 |
| Burnout Level | Accuracy | 65.37% |
| Burnout Level | Macro F1-Score | 0.505 |

**Note:** Accuracy alone does not represent overall model performance, particularly when the target classes are imbalanced. Therefore, F1-score and other classification metrics are also considered.

## Project Structure

```text
student_health_ml_project/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_utils.py
│   ├── modeling.py
│   └── reporting.py
│
├── data/
│   └── raw/
│       └── student_health.csv
│
├── artifacts/
│   └── models/
│       ├── regression_model.joblib
│       ├── binary_model.joblib
│       └── multiclass_model.joblib
│
└── reports/
    ├── dataset_summary.csv
    ├── descriptive_statistics.csv
    ├── missing_values.csv
    ├── model_comparison.csv
    ├── model_metrics.json
    ├── final_summary.txt
    └── figures/
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

### 2. Navigate to the Project Directory

```bash
cd student_health_ml_project
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

### Step 1: Train the Models

```bash
python train.py
```

This executes the training workflow and generates the trained model artifacts.

### Step 2: Launch the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

## Future Enhancements

- Improve model performance through hyperparameter tuning.
- Address class imbalance to improve minority-class predictions.
- Add explainable AI techniques for prediction interpretation.
- Enhance the user interface and prediction visualizations.
- Integrate the application with a database for student record management.

## Disclaimer

This project is developed for educational and analytical purposes. Predictions should not be treated as medical diagnoses or definitive assessments of a student's mental health or academic future.

## Author

**Keerthana D**

Machine Learning | Python | Data Analytics | AI