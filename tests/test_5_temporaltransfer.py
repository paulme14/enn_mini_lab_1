"""
Test file for Task 5
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

from src.temporal_transfer import clean_data_2025, cross_validate_2025_model, get_common_features
from src.preprocessing import clean_data
from src.gradient_descent import GradientDescentLinearModel
from src.visualization import plot_learning_curve

# ---------------------------------------------------------------------
# Helper: load and clean data
# ---------------------------------------------------------------------
def load_temporal_clean_data():
    df_2018_train = pd.read_csv("data/train.csv")
    df_2018_val   = pd.read_csv("data/validation.csv")
    df_2025_train = pd.read_csv("data/train_2025.csv")
    df_2025_val   = pd.read_csv("data/validation_2025.csv")

    df_2018_train_clean = clean_data(df_2018_train)
    df_2018_val_clean   = clean_data(df_2018_val)
    df_2025_train_clean = clean_data_2025(df_2025_train)
    df_2025_val_clean   = clean_data_2025(df_2025_val)

    # shared feature set
    common_feats = get_common_features()

    def to_xy(df):
        X = df[common_feats].to_numpy()
        y = df["totalRent"].to_numpy()
        return X, y

    X_2018_train, y_2018_train = to_xy(df_2018_train_clean)
    X_2018_val,   y_2018_val   = to_xy(df_2018_val_clean)
    X_2025_train, y_2025_train = to_xy(df_2025_train_clean)
    X_2025_val,   y_2025_val   = to_xy(df_2025_val_clean)

    return {
        "X_2018_train": X_2018_train, "y_2018_train": y_2018_train,
        "X_2018_val":   X_2018_val,   "y_2018_val":   y_2018_val,
        "X_2025_train": X_2025_train, "y_2025_train": y_2025_train,
        "X_2025_val":   X_2025_val,   "y_2025_val":   y_2025_val
    }

# ---------------------------------------------------------------------
# 5.1 Test: Overfitting when using limited 2025 data
# ---------------------------------------------------------------------
def test_2025_overfitting():
    df25_train = pd.read_csv("data/train_2025.csv")
    df25_val   = pd.read_csv("data/validation_2025.csv")

    df25_train_clean = clean_data_2025(df25_train)
    df25_val_clean   = clean_data_2025(df25_val)

    def to_xy(df):
        X = df.drop(columns=["totalRent"]).to_numpy()
        y = df["totalRent"].to_numpy()
        return X, y
    
    X25_train, y25_train = to_xy(df25_train_clean)
    X25_val,   y25_val   = to_xy(df25_val_clean)
    # ---------- Cross-validation on VERY SMALL dataset ----------
    cv = cross_validate_2025_model(X25_train, y25_train, k=5)

    print("CV train R²:", cv["mean_train_r2"])
    print("CV val   R²:", cv["mean_val_r2"])

    # ---------- Expect explicit overfitting ----------
    assert cv["mean_train_r2"] > 0.90, "Expected heavy overfitting (train R² > 0.9)"
    assert cv["mean_val_r2"] < 0.80, "Expected poor generalization (val R² < 0.8)"

    # ---------- Validate final model on true validation set ----------
    m = LinearRegression()
    m.fit(X25_train, y25_train)
    y25_final = m.predict(X25_val)
    r2_final = r2_score(y25_val, y25_final)
    print("Final test R²:", r2_final)

    # Should also be bad
    assert r2_final < 0.8, "Expected poor R² on held-out 2025 validation data."

    # ---------- Same on common features between both data sets ------
    common_feats = ["balcony", "floor", "livingSpace", "noParkSpaces", "noRooms", "yearConstructed"]
    #common_feats = ["balcony", "cellar", "floor", "garden", "lastRefurbish", "livingSpace", "noParkSpaces", "noRooms", "yearConstructed"]

    X_train = df25_train_clean[common_feats].to_numpy()
    y_train = df25_train_clean["totalRent"].to_numpy()

    X_val = df25_val_clean[common_feats].to_numpy()
    y_val = df25_val_clean["totalRent"].to_numpy()

    cv = cross_validate_2025_model(X_train, y_train, k=5)

    print("CV train common features R²:", cv["mean_train_r2"])
    print("CV val   common features R²:", cv["mean_val_r2"])

    mCF = LinearRegression()
    mCF.fit(X_train, y_train)
    y_final = mCF.predict(X_val)
    r2_final_CF = r2_score(y_val, y_final)
    print("Final test R²:", r2_final_CF)

    print("✅ 2025 overfitting test passed.")

# ---------------------------------------------------------------------
# 5.2 Test: Gradient Descent in Online Learning
# ---------------------------------------------------------------------
def test_gradient_descent_temporal_transfer():
    data = load_temporal_clean_data()

    X18_train = data["X_2018_train"]
    y18_train = data["y_2018_train"]
    X18_val   = data["X_2018_val"]
    y18_val   = data["y_2018_val"]

    X25_train = data["X_2025_train"]
    y25_train = data["y_2025_train"]
    X25_val   = data["X_2025_val"]
    y25_val   = data["y_2025_val"]

    # Comparison: Trained on 2018 (common features) data
    # tested on validation 2018 and 2025 data
    m18 = LinearRegression()
    m18.fit(X18_train, y18_train)

    y18_pred = m18.predict(X18_val)
    r2_18_on_18 = r2_score(y18_val, y18_pred)
    y25_pred = m18.predict(X25_val)
    r2_18_on_25 = r2_score(y25_val, y25_pred)
    print("R2 (2018 val): ", r2_18_on_18, " - transfer (2025 val): ", r2_18_on_25)
    assert r2_18_on_25 < 0.3, \
        f"Expected R² < 0.3 in transfer, got {r2_18_on_25:.3f}"
    # ---------------------------------------------------------
    # 1) Normal GD training on 2018 (baseline)
    # ---------------------------------------------------------
    model = GradientDescentLinearModel(learning_rate=0.005, epochs=1000)
    model.fit(X18_train, y18_train, X18_val, y18_val)

    baseline_2018 = model.evaluate(X18_train, y18_train, X25_val, y25_val)
    print("GD baseline (2018→2025) R²:", baseline_2018["r2_val"])

    # ---------------------------------------------------------
    # 2) Transfer training EXACTLY like in spatial transfer
    # ---------------------------------------------------------
    model = GradientDescentLinearModel(learning_rate=0.005, epochs=1000)

    res_transfer = model.transfer_training(
        X18_train, y18_train,
        X18_val,   y18_val,
        X25_train, y25_train,
        X25_val,   y25_val,
        fine_tune_steps=10000      # exakt wie in Spatial Transfer
    )

    print("R2 (2025 val) before fine-tuning: ", res_transfer["before_pre_ft"]["r2_val"] , " - after fine-tuning: ", res_transfer["after_ft_ft"]["r2_val"])
    res_after = res_transfer["after_ft_ft"]["r2_val"]
    # ---------------------------------------------------------
    # 3) Expected performance
    # ---------------------------------------------------------
    assert 0.7 <= res_after <= 0.85, \
        f"Expected R² ~0.75–0.80 after fine-tuning, got {res_after:.3f}"

    print("✅ Temporal GD transfer OK — After transfer R²:", res_after)

# ---------------------------------------------------------------------
# 5.3 Test: Learning Curve for Training and further training
# ---------------------------------------------------------------------
def test_learning_curve_output():
    """Test: learning curve PDF is created."""
    data = load_temporal_clean_data()

    X18_train = data["X_2018_train"]
    y18_train = data["y_2018_train"]
    X18_val   = data["X_2018_val"]
    y18_val   = data["y_2018_val"]
    X25_train = data["X_2025_train"]
    y25_train = data["y_2025_train"]
    X25_val   = data["X_2025_val"]
    y25_val   = data["y_2025_val"]

    model = GradientDescentLinearModel(learning_rate=0.005, epochs=1000)

    # run temporal transfer
    model.transfer_training(
        X18_train, y18_train,
        X18_val,   y18_val,
        X25_train, y25_train,
        X25_val,   y25_val,
        fine_tune_steps=2000
    )

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Baseline model for comparison
    model_bl = GradientDescentLinearModel(learning_rate=0.005, epochs=2000)
    model_bl.fit(X25_train, y25_train, X25_val, y25_val)
    baseline_2025 = model_bl.evaluate(X25_train, y25_train, X25_val, y25_val)

    # plot curves
    curve_pdf = plot_learning_curve(model, output_dir=output_dir, file_name="Task_5_3_learning curve", ref_rmse=baseline_2025["rmse_val"])

    # test existence
    assert os.path.exists(curve_pdf), "Learning curve PDF was not created."

    print("✅  Learning curve successfully created:", curve_pdf)
