"""
Test file for Task 2 — Feature Analysis and Model Extension
-----------------------------------------------------------
This file validates:
  1. Single feature ranking correctness and structure
  2. Stepwise feature selection progression
  3. Five-feature model performance
  4. Visualization of model complexity vs. performance

Run manually via:
    python -m pytest -s tests/test_2_feature_analysis.py
"""

import pandas as pd
import os, sys, math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_analysis import analyze_single_features, stepwise_selection
from src.preprocessing import clean_data
from src.baseline_model import evaluate_feature_set
from src.visualization import plot_feature_performance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


# ---------------------------------------------------------------------
# 2.1 Test – Single feature ranking (test is just checking the structure)
# ---------------------------------------------------------------------
def test_single_feature_ranking():
    """Check that analyze_single_features returns a valid descending R² ranking."""
    df_train = pd.read_csv("data/train.csv")
    df_clean = clean_data(df_train) 

    ranking = analyze_single_features(df_clean)

    # --- Structural checks ---
    assert isinstance(ranking, list), "analyze_single_features() must return a list"
    assert all(isinstance(item, dict) and "feature" in item and "r2" in item for item in ranking), \
        "Each item must be a dict with keys 'feature' and 'r2'"
    assert all(isinstance(item["r2"], (int, float)) and math.isfinite(float(item["r2"])) for item in ranking), \
        "R² values must be numeric"

    # --- Logical checks ---
    assert all(0 <= item["r2"] <= 1 for item in ranking), \
        "R² values must be within [0, 1]"
    assert len(ranking) >= 10, "Expected at least 10 features in ranking"
    assert all(ranking[i]["r2"] >= ranking[i+1]["r2"] for i in range(len(ranking)-1)), \
        "R² values are not strictly descending"

    # --- Check categorical feature presence ---
    df_cols = df_clean.columns
    cat_features = [
        item["feature"]
        for item in ranking
        if item["feature"].endswith("_num") and item["feature"] in df_cols
    ]
    assert len(cat_features) >= 2, "At least two categorical (_num) features expected"

    print(f"✅ Single feature ranking valid ({len(ranking)} features, top: {ranking[0]})")


# ---------------------------------------------------------------------
# 2.2 Test – Stepwise selection structure (test is just checking the structure)
# ---------------------------------------------------------------------
def test_stepwise_selection_structure():
    """Check that stepwise_selection returns increasing R² values."""
    df_train = clean_data(pd.read_csv("data/train.csv"))
    df_val = clean_data(pd.read_csv("data/validation.csv"))

    result = stepwise_selection(df_train, df_val)

    assert isinstance(result, list), "Result must be a list"
    assert all(isinstance(r, dict) and isinstance(r["features"], list) for r in result), \
        "Each entry must be a dict: n_features, feature_list, r2_value"
    assert all(0 <= r2["r2"] <= 1 for r2 in result), "R² values must be within [0,1]"
    assert all(result[i]["r2"] <= result[i + 1]["r2"] for i in range(len(result) - 1)), \
        "R² values must increase monotonically"

    print(f"✅ Stepwise selection returns {len(result)} feature combinations.")


# ---------------------------------------------------------------------
# 2.3 Test – Model with five features meets performance threshold
# ---------------------------------------------------------------------
def test_stepwise_model_performance():
    """Check that the five-feature model achieves sufficient performance."""
    df_train = clean_data(pd.read_csv("data/train.csv"))
    df_val = clean_data(pd.read_csv("data/validation.csv"))
    # This test checks the model performance for your best five found features
    # Important: After submission, this will be tested against a real (unknown) test set

    # Run stepwise selection (Task 2.2)
    result = stepwise_selection(df_train, df_val)

    # Find the 5-feature combination
    assert result[4] is not None, "No feature set with 5 features found"
    selected_features = result[4]["features"]

    # Prepare data
    X_train = df_train[selected_features]
    y_train = df_train["totalRent"]
    X_val = df_val[selected_features]
    y_val = df_val["totalRent"]

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    # Expected thresholds
    assert r2 > 0.8, f"Expected R² > 0.8, got {r2:.3f}"
    assert rmse < 230, f"Expected RMSE < 230 €, got {rmse:.1f} €"

    print(f"✅ 5-feature model performance on validation set: R²={r2:.3f}, RMSE={rmse:.1f}")

# ---------------------------------------------------------------------
# 2.4 Test – Visualization of feature complexity performance
# ---------------------------------------------------------------------
def test_visualization_feature_performance():
    """
    Create and verify the feature-complexity performance plot
    using real stepwise selection results from Task 2.
    """
    import os

    df_train = clean_data(pd.read_csv("data/train.csv"))
    df_val = clean_data(pd.read_csv("data/validation.csv"))

    # Use the actual stepwise selection results from Task 2
    stepwise_results = stepwise_selection(df_train, df_val)

    # Create visualization
    output_path = plot_feature_performance(
        stepwise_results,
        output_dir="results",
        file_name="Task_2_feature_performance"
    )

    # Check file existence
    assert os.path.exists(output_path), f"Expected plot not found: {output_path}"

    print(f"✅ Task 2 visualization successfully saved → {output_path}")
