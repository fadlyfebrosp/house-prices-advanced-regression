# End-to-End House Prices Prediction Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 1. Business Objective & Project Overview
This repository contains a production-ready, end-to-end machine learning pipeline designed to predict residential real estate prices in Ames, Iowa[cite: 1]. The primary objective is to deliver highly accurate, robust predictions using advanced regression techniques, strictly aligned with the standard Kaggle evaluation metric: **Root Mean Squared Error (RMSE) of log-transformed sale prices**[cite: 1]. 

The architecture emphasizes pipeline integrity, modularity, and deployment-readiness, bridging the gap between exploratory data science and software engineering best practices.

## 2. Key Features
*   **Zero-Leakage Preprocessing Pipeline:** Constructed using Scikit-Learn's `ColumnTransformer` and `Pipeline` to ensure strict separation of fit and transform steps across 79 high-dimensional features.
*   **Target & Feature Transformation:** Automated application of `np.log1p` to correct skewness in both the target variable (SalePrice) and highly skewed continuous features.
*   **Context-Aware Imputation:** Intelligent handling of missing values, including neighborhood-grouped median imputation for `LotFrontage`.
*   **Robust Outlier Handling:** Programmatic, safe removal of influential extreme outliers (e.g., specific `GrLivArea` anomalies) prior to model fitting to stabilize decision boundaries.
*   **Domain-Driven Feature Engineering:** Creation of high-signal composite features including `TotalSF` (Total Square Footage), `TotalBath` (Total Bathrooms), `HouseAge`, and standardized mapping of ordinal quality indicators.
*   **Advanced Ensemble Blending:** Integration of regularized linear models and state-of-the-art tree-based algorithms (Ridge, Lasso, XGBoost, LightGBM, CatBoost) optimized via rigorous 5-Fold Cross-Validation.
*   **Production Serialization:** Automated export of the final optimized pipeline via `joblib` for immediate downstream inference.

## 3. Tech Stack & Tooling
*   **Core Language:** Python 3.9+
*   **Data Manipulation:** Pandas, NumPy
*   **Machine Learning Framework:** Scikit-Learn
*   **Gradient Boosting Libraries:** XGBoost, LightGBM, CatBoost
*   **Pipeline Serialization:** Joblib
*   **Data Visualization:** Matplotlib, Seaborn

## 4. Project Architecture
```text
├── artifacts/
│   ├── production_pipeline.pkl   # Serialized Joblib model pipeline
│   └── scaler.pkl                # Serialized feature scalers
├── data/
│   ├── train.csv                 # Raw training data
│   ├── test.csv                  # Raw testing data
│   └── data_description.txt      # Data dictionary
├── notebooks/
│   └── house-prices-prediction.ipynb # Exploratory Data Analysis & Prototyping
├── src/
│   ├── __init__.py
│   ├── config.py                 # Global configurations & hyperparameters
│   ├── data_ingestion.py         # Data loading and initial cleaning (outliers)
│   ├── feature_engineering.py    # Custom transformers (TotalSF, TotalBath, HouseAge)
│   ├── preprocessing.py          # ColumnTransformer pipelines and imputation
│   ├── model.py                  # Ensemble Blending, Ridge, Lasso, XGBoost, LightGBM, CatBoost
│   └── inference.py              # Prediction script for new data
├── submission/
│   └── submission.csv            # Final Kaggle-formatted predictions
├── requirements.txt
└── README.md
```
## 5. Installation & Prerequisites
Step 1: Clone the repository
```text
git clone [https://github.com/your-organization/house-prices-prediction.git](https://github.com/your-organization/house-prices-prediction.git)
cd house-prices-prediction
```
Step 2: Create and activate a virtual environment
```text
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```
Step 3: Install dependencies
```text
pip install --upgrade pip
pip install -r requirements.txt
```
## 6. Usage & Workflow
Executing the Training Pipeline
To train the model from scratch, execute the main orchestration script. This will load the data, apply the ColumnTransformer preprocessing, perform 5-Fold CV, build the blended ensemble, and serialize the output.
```text
python -m src.train
```
Running the Notebook
For EDA and experimental modeling, launch Jupyter:
```text
jupyter notebook notebooks/house-prices-prediction.ipynb
```
Production Inference
To load the serialized joblib pipeline and generate predictions on new data, use the inference module:
```text
import joblib
import pandas as pd

# Load the production pipeline
pipeline = joblib.load('artifacts/production_pipeline.pkl')

# Load new data
new_data = pd.read_csv('data/test.csv')

# Generate predictions (the pipeline inherently handles inverse log1p if configured)
predictions = pipeline.predict(new_data)
print(predictions)
```
## 7. Contribution Guidelines & License
Contributing
We adhere to standard open-source workflows.

Fork the repository.

Create a feature branch (git checkout -b feature/model-optimization).

Commit your changes (git commit -m 'Add hyperparameter tuning for LightGBM').

Push to the branch (git push origin feature/model-optimization).

Open a Pull Request.

Please ensure all new code passes existing unit tests and complies with the black formatting standard.

## 8. License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
