"""
Test file for Task 4
-----------------------------------------------------------
This file validates:
  1. Check cleaning result (structure + missing values) for Bielefeld
  2. Simple transfer of Baseline model to different city 
  3. Overfitting on small data set (for Bielefeld)
  4. Implementation of Gradient Descent
  5. Further training on small data for better transfer to novel city

Run manually via:
    python -m pytest -s tests/test_4_spatialtransfer.py
(-v verbose delivers more details.)
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import clean_data
from src.gradient_descent import GradientDescentLinearModel
from src.spatial_transfer import run_cross_domain_evaluation_spatial

# ---------------------------------------------------------------------
# Helper: load and clean data
# ---------------------------------------------------------------------
def load_all_clean_data(selected_n=5):
    df_MS_train = pd.read_csv("data/train.csv")
    df_MS_val   = pd.read_csv("data/validation.csv")
    df_BI_train = pd.read_csv("data/train_bielefeld.csv")
    df_BI_val   = pd.read_csv("data/validation_bielefeld.csv")
    df_MS_train_clean = clean_data(df_MS_train)
    df_MS_val_clean   = clean_data(df_MS_val)
    df_BI_train_clean = clean_data(df_BI_train)
    df_BI_val_clean   = clean_data(df_BI_val)
    def to_xy(df_clean):
        #X = df_clean[feats].to_numpy()
        X = df_clean.drop(columns=["totalRent"]).to_numpy()
        y = df_clean["totalRent"].to_numpy()
        return X, y
    X_MS_train, y_MS_train = to_xy(df_MS_train_clean)
    X_MS_val,   y_MS_val   = to_xy(df_MS_val_clean)
    X_BI_train, y_BI_train = to_xy(df_BI_train_clean)
    X_BI_val,   y_BI_val   = to_xy(df_BI_val_clean)
    return {
        "X_MS_train": X_MS_train, "y_MS_train": y_MS_train,
        "X_MS_val":   X_MS_val,   "y_MS_val":   y_MS_val,
        "X_BI_train": X_BI_train, "y_BI_train": y_BI_train,
        "X_BI_val":   X_BI_val,   "y_BI_val":   y_BI_val
    }

# ---------------------------------------------------------------------
# 4.1 Test: Cleaning Bielefeld dataset
# ---------------------------------------------------------------------
def test_bielefeld_cleaned_data():
    """Ensure that applying clean_data to Bielefeld data retains >25 valid samples."""
    df = pd.read_csv("data/train_bielefeld.csv")
    df_clean = clean_data(df)

    # Must have sufficient rows
    assert len(df_clean) > 25, f"Expected >25 rows, got only {len(df_clean)}"

    # Must not contain NaN
    assert not df_clean.isnull().any().any(), "Cleaned Bielefeld data still contains NaN values"

    # Must contain only numeric columns (same as train/val Münster)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    assert len(numeric_cols) == df_clean.shape[1], "Some Bielefeld columns are not numeric after cleaning"

    print(f"✅ Bielefeld cleaning successful: {len(df_clean)} rows, {len(numeric_cols)} numeric features.")

# ---------------------------------------------------------------------
# 4.2-4.3 Test: Spatial transfer and Overfitting with little BI-Data
# ---------------------------------------------------------------------
def test_cross_domain_generalization():
    data = load_all_clean_data()
    X_MS_train = data["X_MS_train"]
    y_MS_train = data["y_MS_train"]
    X_MS_val   = data["X_MS_val"]
    y_MS_val   = data["y_MS_val"]
    X_BI_train = data["X_BI_train"]
    y_BI_train = data["y_BI_train"]
    X_BI_val   = data["X_BI_val"]
    y_BI_val   = data["y_BI_val"]

    # Run evaluations on transfer of Muenster model 
    # as well as training a model on the small data set
    results = run_cross_domain_evaluation_spatial(
        X_MS_train,y_MS_train,X_MS_val,y_MS_val,
        X_BI_train,y_BI_train,X_BI_val,y_BI_val
    )

    # sanity checks
    for key, r in results.items():
        assert r["r2_train"] > 0, f"{key}: r2_train <= 0"
        assert r["r2_val"]   > 0, f"{key}: r2_val <= 0"
        assert r["rmse_train"] > 0, f"{key}: rmse_train <= 0"
        assert r["rmse_val"]   > 0, f"{key}: rmse_val <= 0"

    # task 4.2: Muenster model should degrade on BI (we just check this with a fixed margin of 0.1)
    assert results["MS_train_MS_val"]["r2_val"] > results["MS_train_BI_val"]["r2_val"] + 0.1, \
        "Expected: MS→MS > MS→BI"

    print("✅ Cross-domain generalization tests passed.")

    # task 4.3: Overfitting on small BI data (we just check this with a fixed margin of 0.1)
    assert results["BI_train_BI_val"]["r2_train"] + 0.1 > results["BI_train_BI_val"]["r2_val"] + 0.1, \
        "Expected: BItrain→BItrain > BItrain→BIval"

    print("✅ Bielefeld overfitting tests passed.")

# ---------------------------------------------------------------------
# 4.4-4.5 Test: Gradient Descent
# ---------------------------------------------------------------------
def test_gradient_descent_on_muenster():
    data = load_all_clean_data()
    X_MS_train = data["X_MS_train"]
    y_MS_train = data["y_MS_train"]
    X_MS_val   = data["X_MS_val"]
    y_MS_val   = data["y_MS_val"]
    X_BI_train = data["X_BI_train"]
    y_BI_train = data["y_BI_train"]
    X_BI_val   = data["X_BI_val"]
    y_BI_val   = data["y_BI_val"]

    # run evaluations
    results_neq = run_cross_domain_evaluation_spatial(
        X_MS_train,y_MS_train,X_MS_val,y_MS_val,
        X_BI_train,y_BI_train,X_BI_val,y_BI_val
    )

    model = GradientDescentLinearModel(learning_rate=0.005, epochs=1000)
    model.fit(X_MS_train, y_MS_train, X_MS_val, y_MS_val)

    # Evaluate the model
    result_gd_ms = model.evaluate(X_MS_train, y_MS_train, X_MS_val, y_MS_val)

    # Check that gradient descent comes close enough to the performance of the normal equation
    assert results_neq["MS_train_MS_val"]["r2_val"] - 0.01 < result_gd_ms["r2_val"]
    print("✅ Gradient Descent approaches similar level (compared to normal equation) R2: ", result_gd_ms["r2_val"], " (neq R2:", results_neq["MS_train_MS_val"]["r2_val"], ")")
    #result_gd_bi = model.evaluate(X_BI_train, y_BI_train, X_BI_val, y_BI_val)

    model = GradientDescentLinearModel(learning_rate=0.005, epochs=1000)
    res_transfer = model.transfer_training(
        X_MS_train, y_MS_train,
        X_MS_val,   y_MS_val,
        X_BI_train, y_BI_train,
        X_BI_val,   y_BI_val,
        fine_tune_steps=1000
    )
    # Check that the fine-tuning / further training leads to better results for
    # the Bielefeld validation data compared to a model trained only on the
    # Bielefeld training data
    assert results_neq["BI_train_BI_val"]["r2_val"] < res_transfer["after_ft_ft"]["r2_val"]
    # and compared to the model trained on Muenster data
    assert results_neq["MS_train_BI_val"]["r2_val"] < res_transfer["after_ft_ft"]["r2_val"]
    # Check for r2 on validation set greater 0.65
    assert res_transfer["after_ft_ft"]["r2_val"] > 0.65

    print("✅ Bielefeld transfer - from ", results_neq["BI_train_BI_val"]["r2_val"], " improved to (after transfer) ", res_transfer["after_ft_ft"]["r2_val"])
