"""
Preprocessing utilities for the regression exercises.
You have to implement the different functions (according to the exercises).

This module provides functions for:
- Cleaning raw housing data (handling missing values, removing outliers)
- Encoding categorical features into numeric representations
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Encode categorical features (required to fully implement for Task 1.3)
# ---------------------------------------------------------------------
def col_encoder(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, dict]:
    df[col], mapping = pd.factorize(df[col])
    df = df.rename(columns={col: f'{col}_num'})
    return df, dict(enumerate(mapping))


# ---------------------------------------------------------------------
# Clean full dataset (required to fully implement for Task 1.1)
# ---------------------------------------------------------------------
def clean_data(df: pd.DataFrame, get_cat_feature_mapping: bool = False) -> pd.DataFrame:
    """
    Perform complete cleaning pipeline:
    - Remove invalid rows (e.g., livingSpace <= 10)
    - Fill missing values (median for numeric, mode for categorical)
    - Encode categorical features
    - Keep only numeric columns
    """

    df_clean = df.copy()

    # Fill missing values
    # Fill totalRent with baseRent + serviceCharge + heatingCosts
    df_clean['totalRent'] = df_clean['totalRent'].fillna(df_clean['baseRent'] + df_clean['serviceCharge'].fillna(0) + df_clean['heatingCosts'].fillna(0))

    # Fill missing categorical values with 'missing'
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    df_clean[categorical_cols] = df_clean[categorical_cols].fillna('missing')

    # Fill missing numerical values with median
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numerical_cols] = df_clean[numerical_cols].fillna(df_clean[numerical_cols].median())

    # Filter invalid rows

    # Living space
    if "livingSpace" in df_clean.columns:
        df_clean = df_clean[(df_clean["livingSpace"] > 10) & (df_clean["livingSpace"] < 500)]

    # noParkSpaces
    if "noParkSpaces" in df_clean.columns:
        df_clean = df_clean[df_clean["noParkSpaces"] < 10]

    # geo_plz
    if "geo_plz" in df_clean.columns:
        df_clean = df_clean[(df_clean["geo_plz"] < 48400)]

    # floor
    if "floor" in df_clean.columns:
        df_clean = df_clean[df_clean["floor"] < 10]

    # numberOfFloors
    if "numberOfFloors" in df_clean.columns:
        df_clean = df_clean[(df_clean["numberOfFloors"] > 0 ) & (df_clean["numberOfFloors"] < 10)]

    # lastRefurbish
    if "lastRefurbish" in df_clean.columns:
        df_clean = df_clean[df_clean["lastRefurbish"] >= 2000]

    # americanArea
    if "americanArea" in df_clean.columns:
        df_clean = df_clean[df_clean["americanArea"] < 20000]
    

    # Encode categorical columns
    df_cols_nono_num = df.select_dtypes(exclude='number').columns.to_list()
    dict_of_mappings = {}
    for col in df_cols_nono_num:
        df_clean, mapping = col_encoder(df_clean, col)
        dict_of_mappings[col] = mapping


    # Keep only numeric columns
    df_clean = df_clean.select_dtypes(include=[np.number])

    # Final safety check
    df_clean = df_clean.dropna(axis=0, how="any")

    return df_clean

# ---------------------------------------------------------------------
# Inspect missing values
#
# Function showcasing further panda DataFrame handling.
# ---------------------------------------------------------------------
def inspect_missing_values(df: pd.DataFrame):
    """
    Returns a summary of missing values per column.
    Useful for exploration and debugging.
    """
    missing = df.isnull().sum()
    total = len(df)
    percent = (missing / total * 100).round(2)
    summary = pd.DataFrame({"missing": missing, "percent": percent})
    return summary[summary["missing"] > 0].sort_values(by="percent", ascending=False)
