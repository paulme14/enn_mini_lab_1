"""
Test file for Task 1
-----------------------------------------------------------
This file validates:
  1. Check cleaning result (structure + missing values)
  2. Baseline model performance on 1D input (livingSpace -> totalRent)
  3. Contribution of categorical feature(s)

Run manually via:
    python -m pytest -s tests/test_1_dataprocessing.py
(-v verbose delivers more details.)
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import clean_data
from src.baseline_model import train_baseline_model, evaluate_model

# ---------------------------------------------------------------------
# 1.1 Test: Check cleaning result (structure + missing values)
# ---------------------------------------------------------------------
def test_cleaned_data_structure():
    """Ensure cleaned dataset has >330 entries, no NaN, all numeric, >10 features."""
    df = pd.read_csv("data/train.csv")
    df_clean = clean_data(df)

    # Must have sufficient rows
    assert len(df_clean) > 330, f"Expected >300 rows, got {len(df_clean)}"

    # Must not have missing values
    assert not df_clean.isnull().any().any(), "Cleaned DataFrame still contains NaN values"

    # Must have at least 10 features and all numeric
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    assert len(numeric_cols) > 10, f"Expected >10 numeric features, got {len(numeric_cols)}"
    assert len(numeric_cols) == df_clean.shape[1], "Some columns are still non-numeric"

    # This test will already pass with the dummy implementation.
    # We will in addition check for the categorical features.

    print(f"✅ Data cleaning successful: {len(df_clean)} rows, {len(numeric_cols)} numeric features.")

# ---------------------------------------------------------------------
# 1.2 Test: Baseline model performance on 1D input (livingSpace -> totalRent)
# ---------------------------------------------------------------------
def test_baseline_model_performance():
    """Train baseline model (custom) on livingSpace and totalRent."""
    df_train = pd.read_csv("data/train.csv")
    df_val = pd.read_csv("data/validation.csv")

    # Clean both datasets
    df_train_clean = clean_data(df_train)
    df_val_clean = clean_data(df_val)

    # Convert to numpy arrays
    X_train = df_train_clean[["livingSpace"]].to_numpy()
    y_train = df_train_clean["totalRent"].to_numpy()
    X_val = df_val_clean[["livingSpace"]].to_numpy()
    y_val = df_val_clean["totalRent"].to_numpy()

    # Train and evaluate
    model = train_baseline_model(X_train, y_train)
    eval_result = evaluate_model(model, X_val, y_val)

    # Expected performance range
    assert 0.75 <= eval_result["r2"], f"Unexpected R²: {eval_result['r2']:.3f}"
    assert eval_result["rmse"] <= 250, f"Unexpected RMSE: {eval_result['rmse']:.1f}"

    print(f"✅ Baseline model performance within expected range — R²: {eval_result['r2']:.3f}, RMSE: {eval_result['rmse']:.1f}€")

# ---------------------------------------------------------------------
# 1.3 Test: Contribution of categorical feature(s)
# ---------------------------------------------------------------------
def test_categorical_feature_contribution():
    """Check that at least one encoded categorical feature contributes (R² > 0.1)."""
    df = pd.read_csv("data/train.csv")
    df_clean = clean_data(df)

    # Identify encoded categorical columns
    cat_cols = [c for c in df_clean.columns if c.endswith("_num")]
    assert len(cat_cols) > 0, "No categorical _num features found after cleaning."

    y = df_clean["totalRent"].to_numpy()

    # Track best feature and score
    best_feature, best_r2 = None, -1

    for cat_feature in cat_cols:
        X = df_clean[[cat_feature]].to_numpy()

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        if r2 > best_r2:
            best_feature, best_r2 = cat_feature, r2

    assert best_r2 > 0.05, (
        f"No categorical feature informative enough (best: {best_feature}, R²={best_r2:.2f})"
    )

    print(f"✅ Best categorical feature '{best_feature}' contributes to prediction (R²={best_r2:.2f})")
