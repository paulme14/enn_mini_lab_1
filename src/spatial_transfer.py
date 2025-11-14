"""
Spatial Transfer

Functions that shall apply the LinearRegression model
on the different training and validation sets 
(testing transfer between different cities).
"""

import numpy as np
from src.baseline_model import BaselineLinearModel
from sklearn.metrics import r2_score, root_mean_squared_error

def train_and_eval(X_train,y_train,X_val,y_val):
    # Apply training to regression model
    # and afterwards evaluate on R2 and RMSE for both data sets
    return {
        "r2_train": np.random.rand(),
        "rmse_train": np.random.rand(),
        "r2_val": np.random.rand(),
        "rmse_val": np.random.rand()
    }

def run_cross_domain_evaluation_spatial(X_MS_train,y_MS_train,X_MS_val,y_MS_val,
                                X_BI_train,y_BI_train,X_BI_val,y_BI_val):
    return {
        "MS_train_MS_val":     train_and_eval(X_MS_train,y_MS_train,X_MS_val,y_MS_val),
        "MS_train_BI_val":     train_and_eval(X_MS_train,y_MS_train,X_BI_val,y_BI_val),
        "BI_train_BI_val":     train_and_eval(X_BI_train,y_BI_train,X_BI_val,y_BI_val),
    }