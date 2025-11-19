"""
Implements the feature ranking and stepwise selection procedures
for Task 2 of the ENN Minilab project.

This module provides functions for:
  - Compute single-feature performance (analyze_single_features)
  - Implement stepwise feature selection (stepwise_selection)
  
Current version: Dummy implementation returning example data
so that tests can be applied (but will fail).
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Single-feature analysis (Task 2.1)
# ---------------------------------------------------------------------
def analyze_single_features(df: pd.DataFrame, target_col: str = "totalRent"):
    """
    Analyze each feature independently to assess its correlation
    (R² performance) with the target variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset including numeric and encoded categorical columns.
    target_col : str, default="totalRent"
        Name of the target variable.
    
    Returns
    -------
    list of dict
        Example format:
        [
            {"feature": "livingSpace", "r2": 0.75},
            {"feature": "numberOfRooms", "r2": 0.68},
            ...
        ]
        Sorted in descending order by R².
    """
    # --- Dummy data for testing ---
    features = [
        "livingSpace", "numberOfRooms", "yearConstructed",
        "baseRent", "floor", "heatingType_num",
        "condition_num", "balcony", "cellar", "lift"
    ]
    np.random.seed(42)
    r2_values = np.linspace(0.9, 0.1, len(features))  # strictly descending

    return [{"feature": f, "r2": float(r)} for f, r in zip(features, r2_values)]


# ---------------------------------------------------------------------
# Stepwise feature selection (Task 2.2)
# ---------------------------------------------------------------------
<<<<<<< HEAD
def stepwise_selection(df: pd.DataFrame, features: list, target_col: str = "totalRent"):
=======
def stepwise_selection(df_train: pd.DataFrame, df_val: pd.DataFrame):
>>>>>>> upstream/main
    """
    Simulate a stepwise feature selection process that gradually adds features
    and evaluates the model performance.

    Parameters
    ----------
<<<<<<< HEAD
    df : pd.DataFrame
        Cleaned dataset.
    features : list
        List of features in the order they are added.
    target_col : str
        Target variable name.
=======
    df_train : pd.DataFrame
        Cleaned training dataset.
    df_val : pd.DataFrame
        Cleaned validation dataset.
>>>>>>> upstream/main

    Returns
    -------
    list of dict
        Example format:
        [
            {"n_features": 1, "features": ["livingSpace"], "r2": 0.70, "rmse": 250.0},
            {"n_features": 2, "features": ["livingSpace", "numberOfRooms"], "r2": 0.78, "rmse": 210.0},
            ...
        ]
    """
    # --- Dummy results for test verification ---
    dummy_results = [
        {"n_features": 1, "features": ["livingSpace"], "r2": 0.70, "rmse": 260.0},
        {"n_features": 2, "features": ["livingSpace", "noRooms"], "r2": 0.78, "rmse": 210.0},
        {"n_features": 3, "features": ["livingSpace", "noRooms", "floor"], "r2": 0.82, "rmse": 190.0},
        {"n_features": 4, "features": ["livingSpace", "noRooms", "floor", "picturecount"], "r2": 0.85, "rmse": 180.0},
        {"n_features": 5, "features": ["livingSpace", "noRooms", "floor", "picturecount", "noParkSpaces"], "r2": 0.87, "rmse": 175.0},
    ]
    return dummy_results
