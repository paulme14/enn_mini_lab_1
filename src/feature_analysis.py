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
    # Implementation
    features = df.columns.to_list()
    features.remove(target_col)
    output = []
    for feature in features:
        X = df[feature]
        Y = df.loc[X.index, target_col]
        r_value = X.corr(Y)
        output.append({'feature': feature, "r2": abs(r_value)})

    # order descending and return
    return sorted(output, key=lambda x: x['r2'], reverse=True)
    

# ---------------------------------------------------------------------
# Stepwise feature selection (Task 2.2)
# ---------------------------------------------------------------------
def stepwise_selection(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """
    Simulate a stepwise feature selection process that gradually adds features
    and evaluates the model performance.

    Parameters
    ----------
    df_train : pd.DataFrame
        Cleaned training dataset.
    df_val : pd.DataFrame
        Cleaned validation dataset.

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
