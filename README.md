# Loan Approval Prediction

A machine learning project that predicts whether a loan application will be **Approved** or **Rejected** based on applicant financial and personal details, using Logistic Regression.

## Problem Statement

Financial institutions need a fast, data-driven way to assess loan applications. This project builds a binary classification model to predict loan approval status using applicant details such as income, credit score (CIBIL), assets, and loan terms.

## Dataset

The dataset (`loan_approval_dataset.csv`) contains loan application records with the following features:

| Column | Description |
|---|---|
| `loan_id` | Unique identifier for each application |
| `no_of_dependents` | Number of dependents |
| `education` | Graduate / Not Graduate |
| `self_employed` | Yes / No |
| `income_annum` | Annual income |
| `loan_amount` | Requested loan amount |
| `loan_term` | Loan term (in years) |
| `cibil_score` | Credit score |
| `residential_assets_value` | Value of residential assets |
| `commercial_assets_value` | Value of commercial assets |
| `luxury_assets_value` | Value of luxury assets |
| `bank_asset_value` | Value of bank assets |
| `loan_status` | Target: Approved / Rejected |

## Approach

1. **Data cleaning** — stripped whitespace, standardized column names and categorical values, removed duplicates.
2. **Encoding** — categorical features (`education`, `self_employed`, `loan_status`) encoded using `LabelEncoder`.
3. **Train/test split** — 80/20 split.
4. **Scaling** — features scaled using `MinMaxScaler`.
5. **Model** — `LogisticRegression` from scikit-learn.
6. **Evaluation** — accuracy, precision, recall, F1-score, confusion matrix.
7. **Persistence** — trained model, scaler, and encoders saved with `joblib` for reuse.
8. **Prediction** — a simple CLI-style flow to collect new applicant details and predict loan status.

## Results

| Metric | Score |
|---|---|
| Accuracy | 91.2% |
| Precision | 88.8% |
| Recall | 87.4% |
| F1 Score | 88.1% |

**Confusion Matrix**

|  | Predicted Rejected | Predicted Approved |
|---|---|---|
| **Actual Rejected** | 501 | 35 |
| **Actual Approved** | 40 | 278 |

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/SarthakoZ/loan-approval-prediction.git
   cd loan-approval-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open and run `Loan.ipynb` in Jupyter Notebook or JupyterLab.

> **Note:** The notebook's data-loading cell uses a local file path (`D:\PYTHON\...`). Update it to point to `loan_approval_dataset.csv` in this repo, or place the CSV in that exact location, before running.

> **Note:** The final cells of the notebook use `input()` to collect applicant details interactively. These cells require running the notebook interactively (they won't execute via "Run All" in a non-interactive environment like GitHub's preview or automated pipelines).

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- joblib

## Future Improvements

- Try other models (Random Forest, XGBoost, SVM) and compare performance against Logistic Regression.
- Add cross-validation instead of a single train/test split for more reliable metrics.
- Perform hyperparameter tuning (e.g. `GridSearchCV`) to optimize model performance.
- Add feature importance / coefficient analysis to explain which factors most influence approval.
- Handle class imbalance explicitly (e.g. `class_weight='balanced'` or SMOTE) if the approved/rejected split is skewed.
- Replace the interactive `input()` prediction flow with a proper `predict.py` script or a simple web app (e.g. Streamlit/Flask) for easier use.
- Add input validation for the CIBIL score range, income, and asset values in the prediction flow.
- Log experiment results (metrics, parameters) for easier comparison across model versions.

## Author

**SarthakoZ**
GitHub: [github.com/SarthakoZ](https://github.com/SarthakoZ)
