"""
Test file for Task 3 — Polynomial Analysis
-----------------------------------------------------------
This file validates:
  1. Single feature ranking correctness and structure
  2. Stepwise feature selection progression
  3. Five-feature model performance
  4. Visualization of model complexity vs. performance

Run manually via:
    python -m pytest -s tests/test_3_polynomial_analysis.py
"""

import pandas as pd
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error

from src.preprocessing import clean_data
from src.polynomial_analysis import evaluate_polynomial_models, analyze_polynomial_performance, get_best_polynomial_model, build_polynomial_design_matrix
from src.visualization import plot_polynomial_results, plot_heatmap_performance

# ---------------------------------------------------------------------
# Helper: load and clean data
# ---------------------------------------------------------------------
def load_clean_data():
    """Load training and validation data, clean and return arrays."""
    df_train = pd.read_csv("data/train.csv")
    df_val = pd.read_csv("data/validation.csv")

    df_train_clean = clean_data(df_train)
    df_val_clean = clean_data(df_val)

    # Candidate numeric features
    # TODO for students: You have to select the optimal features.
    candidate_features = [
        "livingSpace",
        "yearConstructed",
        "noParkSpaces",
        "picturecount",
        "noRooms",
        "totalRent",
    ]

    # Filter to only those that still exist
    selected_features = [f for f in candidate_features if f in df_train_clean.columns]
    #print(f"Using features: {selected_features}")

    # Prepare numpy arrays
    X_train = df_train_clean[selected_features].to_numpy()
    y_train = df_train_clean["totalRent"].to_numpy()
    X_val = df_val_clean[selected_features].to_numpy()
    y_val = df_val_clean["totalRent"].to_numpy()

    #print(f"Loaded and cleaned data: {len(X_train)} train, {len(X_val)} val samples")
    return X_train, y_train, X_val, y_val

# ---------------------------------------------------------------------
# 3.1 Test — Structure and plausibility of polynomial model results
# ---------------------------------------------------------------------
def test_polynomial_results_structure():
    """Check that polynomial model summary results are valid."""
    X_train, y_train, X_val, y_val = load_clean_data()

    # Raw polynomial evaluations
    poly_results = evaluate_polynomial_models(X_train, y_train, X_val, y_val)
    assert isinstance(poly_results, list)
    assert len(poly_results) > 0

    # Summarize (optimal + overfit degrees)
    summary = analyze_polynomial_performance(poly_results)

    # --- Structure checks ---
    assert isinstance(summary, list), "Summary must be a list"
    assert len(summary) >= 2, "Expected at least 2 feature combinations"

    required_keys = {
        "features",
        "optimal_degree", "optimal_r2_val", "optimal_rmse_val",
        "overfit_degree", "overfit_r2_val", "overfit_rmse_val"
    }

    for entry in summary:
        assert isinstance(entry, dict), "Each summary entry must be a dict"
        assert required_keys.issubset(entry.keys()), "Missing keys in summary entry"
        assert isinstance(entry["features"], list)
        assert len(entry["features"]) >= 1

    # --- Logical checks ---
    for entry in summary[0:2]:
        # order of degree check
        assert entry["optimal_degree"] < entry["overfit_degree"], \
            "Optimal degree must be smaller than overfit degree"

        # RMSE comparison
        assert entry["optimal_rmse_val"] < entry["overfit_rmse_val"], \
            "Optimal RMSE must be lower than overfit RMSE"

        # R² comparison
        assert entry["optimal_r2_val"] >= entry["overfit_r2_val"], \
            "Optimal R² should be >= overfit R² for validation set"

    print(f"✅ Polynomial summary returned for {len(summary)} feature sets — structure, degrees, and error metrics valid.")


# ---------------------------------------------------------------------
# 3.2 Test — Visualization output creation (heatmap)
# ---------------------------------------------------------------------

def test_visualization_output_heatmap():
    """Ensure both polynomial curve and heatmap plots are generated and saved as PDFs."""
    X_train, y_train, X_val, y_val = load_clean_data()
    results = evaluate_polynomial_models(X_train, y_train, X_val, y_val)

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Validation heatmap (uses flat list of dicts directly)
    heatmap_pdf = plot_heatmap_performance(
        results,  # just pass the list of dicts
        output_dir=output_dir,
        metric="r2_val",
        file_name="Task_3_2_polynomial_heatmap"
    )

    # Check both files exist
    assert os.path.exists(heatmap_pdf), f"Heatmap plot not found: {heatmap_pdf}"

    print(f"✅ Visualization file (Heatmap) created successfully:\n - {heatmap_pdf}")

# ---------------------------------------------------------------------
# 3.3 Test — Visualization output creation: five 2D (one each feature combination)
# ---------------------------------------------------------------------
def test_visualization_output_2D():
    """Ensure both polynomial curve and heatmap plots are generated and saved as PDFs."""
    X_train, y_train, X_val, y_val = load_clean_data()
    results = evaluate_polynomial_models(X_train, y_train, X_val, y_val)

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # Polynomial performance plot
    poly_pdf = plot_polynomial_results(results, output_dir=output_dir, file_name="Task_3_3_degree")

    # Check all 3 files exist
    files = [f for f in os.listdir(output_dir) if f.startswith("Task_3_3")]
    assert len(files) == 3, f"Expected 3 files starting with 'Task_3_3', found {len(files)}"

    print(f"✅ Visualization files created successfully:\n - {poly_pdf}")


# ---------------------------------------------------------------------
# 3.4 Test — Best polynomial model quality
# ---------------------------------------------------------------------
def test_best_polynomial_model_quality():
    """Ensure that best polynomial model reaches expected performance range."""
    X_train, y_train, X_val, y_val = load_clean_data()

    results = evaluate_polynomial_models(X_train, y_train, X_val, y_val)
    best_model = get_best_polynomial_model(results)

    df_train, y_train, df_val, y_val = load_clean_data()   # adapt if needed

    X_train_poly = build_polynomial_design_matrix(X_train, len(best_model["features"]), best_model["degree"])
    X_val_poly = build_polynomial_design_matrix(X_val, len(best_model["features"]), best_model["degree"])

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred_train = model.predict(X_train_poly)
    y_pred_val   = model.predict(X_val_poly)

    r2_train = r2_score(y_train, y_pred_train)
    r2_val = r2_score(y_val, y_pred_val)
    rmse_train = root_mean_squared_error(y_train, y_pred_train)
    rmse_val = root_mean_squared_error(y_val, y_pred_val)

    assert r2_train > 0.83, f"Training R² too low: {r2_train:.3f}"
    assert 0.8 <= r2_val <= 0.90, f"Validation R² not in expected range: {r2_val:.3f}"

    print(f"✅ Best model performance OK — R² train: {r2_train:.3f}, val: {r2_val:.3f}")