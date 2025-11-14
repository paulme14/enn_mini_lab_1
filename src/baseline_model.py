"""
Baseline linear regression model for the regression exercises.

This version provides 
    - A class interface (BaselineLinearModel) for extension by students
    - the functions train_baseline_model() and evaluate_model()
        used in automated tests and notebooks.

The current implementation internally uses scikit-learn LinearRegression.
Task: You have to implement the model yourself.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error


# ---------------------------------------------------------------------
# Linear Regression Model - Class definition
# ---------------------------------------------------------------------
class BaselineLinearModel:
    """A minimal linear regression model (with a scikit-learn like interface)."""

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit model using scikit-learn LinearRegression."""
        self.model.fit(X, y)

        # TODO:
        # Replace this later with your own implementation using
        # the Normal Equation.
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values."""
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        """Compute R² and RMSE metrics."""
        y_pred = self.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = root_mean_squared_error(y, y_pred)
        return {"r2": float(r2), "rmse": float(rmse)}


# ---------------------------------------------------------------------
# Task 1 — Basic linear regression functions
# Functions to run the models as used in the tests of exercise 1.
# ---------------------------------------------------------------------
def train_baseline_model(X_train, y_train):
    """
    Train a simple linear regression model using scikit-learn.
    Students will later replace this with their own implementation
    using the normal equation.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val):
    """
    Evaluate a trained model on validation data.
    Returns R² and RMSE in a dict.
    """
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    rmse = root_mean_squared_error(y_val, y_pred)
    return {"r2": r2, "rmse": rmse}

# ---------------------------------------------------------------------
# Task 2 — Evaluate arbitrary feature combinations
# ---------------------------------------------------------------------
def evaluate_feature_set(df_train, df_val, feature_list, target="totalRent"):
    """
    Evaluate a linear regression model trained on a given feature list.
    Used by the stepwise feature selection procedure in Task 2.
    """

    X_train = df_train_sub[feature_list].to_numpy()
    y_train = df_train_sub[target].to_numpy()
    X_val = df_val_sub[feature_list].to_numpy()
    y_val = df_val_sub[target].to_numpy()

    model = train_baseline_model(X_train, y_train)
    y_pred = model.predict(X_val)

    r2 = r2_score(y_val, y_pred)
    rmse = root_mean_squared_error(y_val, y_pred)

    return {"r2": float(r2), "rmse": float(rmse)}
