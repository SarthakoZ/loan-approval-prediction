# Loan Approval Prediction

A machine learning project that predicts whether a loan application will be **Approved** or **Rejected** based on applicant financial and personal details, deployed as an interactive **Streamlit** app using a **Random Forest Classifier**.

**Live Demo:** [loan-approval-prediction-5qshpvdfgheb6bk2fcxzvf.streamlit.app](https://loan-approval-prediction-5qshpvdfgheb6bk2fcxzvf.streamlit.app/)

## Problem Statement

Financial institutions need a fast, data-driven way to assess loan applications. This project builds and compares several binary classification models to predict loan approval status using applicant details such as income, credit score (CIBIL), assets, and loan terms, then serves the best-performing model through a simple web interface.

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

Implemented in `Source.ipynb`:

1. **Data cleaning** — dropped duplicates, standardized column names and categorical values (lowercased, stripped whitespace), and treated invalid negative numeric entries as missing.
2. **Train/test split** — 80/20 stratified split on the target.
3. **Target encoding** — `loan_status` encoded with `LabelEncoder`.
4. **Preprocessing pipeline** — a `ColumnTransformer` combining:
   - Numeric features: `SimpleImputer` (mean) + `MinMaxScaler`
   - Categorical features: `SimpleImputer` (most frequent) + `OneHotEncoder`
5. **Model comparison** — six classifiers trained and evaluated via a shared `Pipeline` (preprocessor + model): Logistic Regression, KNN, Decision Tree, Random Forest, SVC, and XGBoost.
6. **Evaluation** — accuracy, precision, recall, F1-score, and confusion matrix for each model, plus 5-fold cross-validation and feature-importance analysis on the top model.
7. **Model selection** — Random Forest chosen as the final model based on accuracy and cross-validation performance.
8. **Persistence** — the fitted Random Forest pipeline (preprocessing + model) saved as `loan_approval_final_model.pkl` with `joblib`.
9. **Deployment** — `dev.py` loads the saved pipeline and serves predictions through a Streamlit form.

## Results

| Algorithm | Accuracy |
|---|---|
| **Random Forest** | **98.13%** |
| XGBoost | 98.01% |
| Decision Tree | 97.19% |
| SVC | 94.15% |
| Logistic Regression | 92.51% |
| KNN | 90.52% |

**Random Forest — Confusion Matrix**

|  | Predicted Rejected | Predicted Approved |
|---|---|---|
| **Actual Rejected** | 527 | 4 |
| **Actual Approved** | 12 | 311 |

Random Forest also achieved a mean 5-fold cross-validation accuracy of **98.06%**, and feature-importance analysis showed `cibil_score` as by far the most influential feature, followed by `loan_term` and `loan_amount`.

## How to Run

### Train / explore the model
1. Clone the repository:
   ```bash
   git clone https://github.com/SarthakoZ/loan-approval-prediction.git
   cd loan-approval-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   > `xgboost` is used in `Source.ipynb` but isn't listed in `requirements.txt` — install it separately with `pip install xgboost` if you want to run that section.
3. Open and run `Source.ipynb` in Jupyter Notebook or JupyterLab.

   > **Note:** The notebook's data-loading cell uses a local file path (`D:\Loan_approval\loan_approval_dataset.csv`). Update it to point to `loan_approval_dataset.csv` in this repo before running.

### Run the prediction app
Try it live at [loan-approval-prediction-5qshpvdfgheb6bk2fcxzvf.streamlit.app](https://loan-approval-prediction-5qshpvdfgheb6bk2fcxzvf.streamlit.app/), or run it locally:
1. With dependencies installed and `loan_approval_final_model.pkl` present in the project root, launch the Streamlit app:
   ```bash
   streamlit run dev.py
   ```
2. Enter applicant details in the form and click **Predict Loan Status** to get an Approved/Rejected result, along with a submitted-details summary and a model comparison table.

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- XGBoost (notebook only)
- joblib
- Streamlit

## Future Improvements

- Add `xgboost` to `requirements.txt` since the notebook depends on it.
- Replace the hardcoded local file path in the notebook with a relative path so it runs out of the box.
- Perform hyperparameter tuning (e.g. `GridSearchCV`) on the Random Forest model to push performance further.
- Handle class imbalance explicitly (e.g. `class_weight='balanced'` or SMOTE) if the approved/rejected split is skewed.
- Add input validation in the Streamlit app for CIBIL score range, income, and asset values.
- Log experiment results (metrics, parameters) across model versions for easier comparison.
- Add a `predict.py` / batch-scoring script for non-interactive use alongside the Streamlit app.

## License

This project is licensed under the [MIT License](LICENSE).

## Author

**SarthakoZ**
GitHub: [github.com/SarthakoZ](https://github.com/SarthakoZ)
