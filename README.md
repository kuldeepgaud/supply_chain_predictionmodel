# Supply Chain Prediction Model

A machine learning regression project that predicts **warehouse-level product demand/weight (`product_wg_ton`)** using supply-chain and warehouse-related features.

The project follows a structured ML workflow covering data ingestion, preprocessing, model training, evaluation, cross-validation, experiment tracking, and model artifact generation.

## Project Overview

The objective is to build a predictive model that can estimate product demand in tons based on historical supply-chain information.

### Key Steps

* Data ingestion
* Data cleaning and duplicate removal
* Feature and target separation
* Train-test splitting
* Missing-value handling
* Outlier treatment using Winsorization
* Numerical feature scaling
* Categorical feature encoding
* Random Forest regression
* 5-fold cross-validation
* Model evaluation using R² and MAE
* Feature importance analysis
* MLflow experiment tracking
* Model serialization

## Tech Stack

* **Python 3.11+**
* **Pandas & NumPy** — Data manipulation
* **Scikit-learn** — Preprocessing and machine learning
* **SciPy** — Outlier treatment
* **Random Forest Regressor** — Prediction model
* **MLflow** — Experiment tracking
* **Matplotlib / Seaborn / Plotly** — Visualization
* **UV** — Dependency and environment management

## Project Structure

```text
supply_chain_predictionmodel/
│
├── artifacts/
│   └── model.pkl
│
├── data/
│   ├── raw/
│   └── preprocessed/
│
├── mlruns/
│   └── MLflow experiment data
│
├── research/
│   └── Exploratory analysis / notebooks
│
├── src/
│   └── supply_chain_predictionmodel/
│       ├── data_ingestion.py
│       ├── data_preprocessing.py
│       ├── model_build.py
│       └── __init__.py
│
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

## Machine Learning Pipeline

```text
Raw Data
   ↓
Data Ingestion
   ↓
Data Cleaning
   ↓
Feature / Target Separation
   ↓
Train-Test Split
   ↓
Preprocessing
   ├── Missing Value Imputation
   ├── Winsorization
   ├── Min-Max Scaling
   └── One-Hot Encoding
   ↓
Random Forest Regressor
   ↓
5-Fold Cross Validation
   ↓
Model Evaluation
   ├── R² Score
   └── Mean Absolute Error
   ↓
MLflow Tracking
   ↓
Saved Model
```

## Model

The project uses a **Random Forest Regressor** with:

* 300 estimators
* Maximum depth of None
* Random state of 42

The model is evaluated using:

* **R² Score** — Measures the proportion of variance explained by the model.
* **Mean Absolute Error (MAE)** — Measures the average absolute prediction error.

Five-fold cross-validation is also performed to assess model consistency.

## Preprocessing

The preprocessing pipeline includes:

* Removal of duplicate records
* Removal of non-predictive identifier/year columns
* Median imputation for numerical features
* Most-frequent imputation for categorical features
* Winsorization for numerical outliers
* Min-Max scaling
* One-hot encoding for categorical variables

The target variable is `product_wg_ton`.

## Installation

Clone the repository:

```bash
git clone https://github.com/kuldeepgaud/supply_chain_predictionmodel.git
cd supply_chain_predictionmodel
```

Install dependencies using UV:

```bash
uv sync
```

Alternatively, install the project environment according to your preferred Python environment manager.

## Run the Project

After configuring the dataset path, run:

```bash
python main.py
```

The pipeline will:

1. Load the dataset
2. Preprocess the data
3. Train the Random Forest model
4. Perform cross-validation
5. Generate evaluation metrics
6. Track the experiment using MLflow
7. Save the trained model to:

```text
artifacts/model.pkl
```

The repository currently uses MLflow for experiment tracking and stores run information under `mlruns/`.

## Output

The model generates:

* Cross-validation R² scores
* Mean cross-validation R²
* Training R²
* Testing R²
* MAE
* Feature importance
* Trained model artifact

## Future Improvements

* Replace the hard-coded dataset path with a configurable path
* Save the preprocessing pipeline together with the model
* Perform hyperparameter optimization
* Add a Streamlit prediction interface
* Add automated model validation and CI/CD
* Add model explainability using SHAP

## License

This project is licensed under the **Apache License 2.0**.

## Author

**Kuldeep Gaud**

GitHub: [@kuldeepgaud](https://github.com/kuldeepgaud)
