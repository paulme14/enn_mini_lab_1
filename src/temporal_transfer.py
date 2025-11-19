import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from src.gradient_descent import GradientDescentLinearModel
import pandas as pd

def get_common_features():
    """
    Return a fixed list of features that both data sets have in common
    and for which you cleaned the data
    """
    return [
        "livingSpace",
        "yearConstructed",
        "noParkSpaces",
        "picturecount",
        "noRooms",
    ]

def clean_data_2025(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean 2025 dataset using the same structure as clean_data(), but adapted
    to the available 2025 columns.
    """
    df_clean = df.copy()

    # Filter invalid rows
    if "livingSpace" in df_clean.columns:
        df_clean = df_clean[df_clean["livingSpace"] > 10]

    # Fill numeric columns (median)
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Fill non-numeric columns (mode)
    for col in df_clean.select_dtypes(exclude=[np.number]).columns:
        try:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        except Exception:
            df_clean[col] = df_clean[col].fillna("unknown")

    # Encode categorical columns
    df_clean = encode_categorical(df_clean)

    # Keep only numeric columns
    df_clean = df_clean.select_dtypes(include=[np.number])

    # Final safety check
    df_clean = df_clean.dropna(axis=0, how="any")

    return df_clean

def cross_validate_2025_model(X, y, k=5):
    """
    Dummy version of the cross-validation function.
    Students should implement the full logic based on the provided structure.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    k : int
        Number of folds for K-fold cross-validation.

    Returns
    -------
    dict
        Dictionary containing mean training/validation R² scores and lists
        of all individual fold scores.
    """

    # ------------------------------------------------------------
    # 1. Create a KFold splitter (with shuffle + fixed random_state)
    #    e.g. use sklearn.model_selection.KFold

    # Lists to store training and validation scores for each fold.
    train_scores = []
    val_scores = []

    # 2. Loop over the folds and for each split:
    #      - extract train and validation indices
    #      - create train/validation splits of X and y

    # 3. Create and fit a LinearRegression model on the training data
    #    use your BaseModel, evaluate this

    # Return a dictionary with:
    #      - mean_train_r2   (float)
    #      - mean_val_r2     (float)
    #      - all_train_scores (list of float)
    #      - all_val_scores   (list of float)
    # ------------------------------------------------------------
    return {
        "mean_train_r2": 0.95,     # replace with np.mean(train_scores)
        "mean_val_r2": 0.5,       # replace with np.mean(val_scores)
        "all_train_scores": train_scores,
        "all_val_scores": val_scores,
    }

