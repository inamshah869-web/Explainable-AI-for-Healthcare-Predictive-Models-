# Explainable AI for Healthcare Predictive Models

## 1. Project Overview

This project investigates the application of **Machine Learning (ML)** and **Explainable AI (XAI)** techniques for predicting heart disease using the **UCI Heart Disease Dataset (Cleveland)**.

The primary objective was to develop predictive models for heart disease presence and interpret their decisions using **SHAP (SHapley Additive exPlanations)** and **LIME (Local Interpretable Model-agnostic Explanations)**.

---

## 2. Dataset

The project uses the **UCI Heart Disease Dataset – Cleveland**.

### Task

Binary classification:

* `0` → No heart disease
* `1` → Presence of heart disease

The original target values were transformed into a binary classification problem, where values greater than 0 indicate the presence of heart disease.

### Features

The dataset contains clinical variables including:

* Age
* Sex
* Chest pain type (`cp`)
* Resting blood pressure (`trestbps`)
* Cholesterol (`chol`)
* Fasting blood sugar (`fbs`)
* Resting ECG (`restecg`)
* Maximum heart rate (`thalach`)
* Exercise-induced angina (`exang`)
* ST depression (`oldpeak`)
* Slope
* Number of major vessels (`ca`)
* Thalassemia (`thal`)

After preprocessing, the cleaned dataset contained **303 observations and 14 columns**, with no remaining missing values.

---

## 3. Methodology

### Data Preprocessing

The following preprocessing steps were performed:

* Missing values represented by `?` were identified and handled.
* Duplicate observations were removed.
* All variables were converted to appropriate numeric types.
* Continuous variables were imputed using the **median**.
* Categorical/ordinal variables were imputed using the **mode**.
* Rows with missing target values were removed.
* The original target variable was converted into a binary target.

Continuous variables included:

```text
age
trestbps
chol
thalach
oldpeak
```

Categorical/ordinal variables included:

```text
sex
cp
fbs
restecg
exang
slope
ca
thal
```

---

## 4. Exploratory Data Analysis

Exploratory data analysis was performed to understand the dataset and identify relationships between clinical variables.

The analysis included:

* Target class distribution
* Descriptive statistics
* Feature correlation analysis
* Correlation matrix visualisation

The target distribution was relatively balanced, with slightly more observations corresponding to patients with heart disease.

---

## 5. Machine Learning Models

Four classification algorithms were trained and evaluated:

1. **Logistic Regression**
2. **Random Forest**
3. **Support Vector Machine (SVM)**
4. **XGBoost**

The dataset was divided using a **stratified 80/20 train-test split** with `random_state=42`.

Standard scaling was incorporated into pipelines for the linear/kernel-based models to avoid data leakage.

---

## 6. Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

The trained models and evaluation results were saved for further analysis.

### Performance Summary

| Model               | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -----: | ------: |
| Random Forest       |   0.8852 |    0.9286 | 0.8814 | 0.9545 |       — |
| XGBoost             |   0.8852 |    0.9286 | 0.8814 | 0.9535 |       — |
| Logistic Regression |   0.8689 |    0.9286 | 0.8667 | 0.9513 |       — |
| SVM                 |   0.8525 |    0.8929 | 0.8475 | 0.9437 |       — |

> **Note:** The source report identifies **0.9545 as the highest ROC-AUC for Random Forest**. The extracted report formatting does not preserve the original table alignment completely, so the table above should be checked against the project's `model_performance.csv` before treating every individual metric as definitive.

---

## 7. Explainable AI

The project applies two complementary XAI techniques:

### SHAP

**SHAP (SHapley Additive exPlanations)** was used to explain the predictions of the Random Forest model.

#### Global Explanation

A SHAP summary plot was generated to identify the features with the greatest overall influence on model predictions.

#### Local Explanation

A SHAP waterfall plot was generated to explain the prediction for an individual patient.

The features consistently identified as highly influential included:

* `thal`
* `ca`
* `cp`

### LIME

**LIME (Local Interpretable Model-agnostic Explanations)** was used to generate human-interpretable explanations for individual predictions.

LIME explanations were also aggregated across multiple samples to investigate overall feature relevance.

---

## 8. SHAP vs. LIME Agreement

The project compared SHAP and LIME explanations to assess whether the two XAI methods identified similar important features.

### Results

* **Spearman rank correlation:** `0.9780`
* **Top-10 feature agreement:** `1.0000`
* **SHAP explanation stability:** `1.0000`

The high Spearman correlation indicates strong agreement between the feature rankings produced by SHAP and LIME. All of the top 10 features identified by SHAP were also among the top 10 identified by LIME.

---

## 9. SHAP Stability Analysis

SHAP explanation stability was assessed by repeatedly calculating feature importance using **bootstrapped samples**.

The resulting stability score was:

```text
SHAP Stability Score = 1.0000
```

This indicates that the feature-importance rankings remained highly consistent across the evaluated bootstrap samples.

---

## 10. Project Pipeline

```text
UCI Heart Disease Dataset
          │
          ▼
   Data Preprocessing
          │
          ├── Missing Value Handling
          ├── Duplicate Removal
          ├── Numeric Conversion
          └── Target Binarisation
          │
          ▼
        EDA
          │
          ├── Class Distribution
          ├── Descriptive Statistics
          └── Correlation Analysis
          │
          ▼
    Stratified Train/Test Split
          │
          ▼
    Machine Learning Models
          │
     ┌────┼────────┬─────────┐
     ▼    ▼        ▼         ▼
   LR    RF       SVM      XGBoost
     │    │        │         │
     └────┴────────┴─────────┘
              │
              ▼
       Model Evaluation
              │
       ┌──────┴──────┐
       ▼             ▼
    ROC-AUC      Other Metrics
              │
              ▼
       Best Model: RF
              │
        ┌─────┴─────┐
        ▼           ▼
       SHAP        LIME
        │           │
        └─────┬─────┘
              ▼
      XAI Agreement &
      Stability Analysis
```

---

## 11. Key Outputs

The project generates and stores the following outputs:

### Data

```text
heart_disease_cleveland_clean.csv
```

### Models

Trained models are saved as:

```text
models/*.joblib
```

### Evaluation Results

```text
results/
├── descriptive_statistics.csv
├── model_performance.csv
├── shap_random_forest_feature_importance.csv
└── project_summary.txt
```

### Visualisations

```text
plots/
├── class_distribution.png
├── correlation_matrix.png
├── model_performance_comparison.png
├── roc_curves.png
├── shap_random_forest_summary.png
├── shap_local_patient.png
└── lime_random_forest_patient.png
```

The report specifies that models, plots, explanations and results are saved in their respective directories for further analysis and deployment.

---

## 12. Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* SHAP
* LIME
* Matplotlib
* Seaborn
* Joblib

---

## 13. Key Findings

The project demonstrated that machine learning can be combined with explainability techniques to produce both predictive results and interpretable insights.

Key findings included:

* Random Forest was identified in the report as the best-performing model based on ROC-AUC.
* SHAP and LIME produced highly consistent feature rankings.
* SHAP achieved a stability score of `1.0000`.
* SHAP and LIME achieved a Spearman rank correlation of `0.9780`.
* `thal`, `ca`, and `cp` were consistently identified as influential features.

---

## 14. Conclusion

This project demonstrates an end-to-end **Explainable AI pipeline for healthcare predictive modelling**, covering data preprocessing, exploratory analysis, machine learning model development, evaluation, and model interpretation.

The combination of **SHAP and LIME** provided consistent explanations of model predictions, while the stability analysis demonstrated consistent feature-importance results across repeated evaluations.

The project provides practical experience in:

* Healthcare machine learning
* Binary classification
* Model evaluation
* Explainable AI
* SHAP
* LIME
* Feature-importance analysis
* Model interpretability
* Statistical stability analysis

---

## 15. Reproducibility

The complete implementation is designed to reproduce the data preprocessing, model training, evaluation, SHAP/LIME explanations, agreement analysis, and stability analysis described in this project.

A fixed random seed (`42`) was used for reproducibility, and generated models, metrics, plots, and explanations are organised into dedicated project directories.
