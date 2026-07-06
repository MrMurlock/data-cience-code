import os

import pandas as pd

from data_cience_code.config import config
from data_cience_code.data import load
from data_cience_code.models.classification.association_rules import (
    discretize_column,
    group_occupation,
    prepare_transactions,
)


def _discretize_sleep_duration(df: pd.DataFrame) -> pd.DataFrame:
    return discretize_column(
        df, "Sleep Duration",
        bins=[5.8, 6.4, 7.8, 8.6],
        labels=["Corto", "Medio", "Largo"],
        new_column="sleep_duration_cat",
    )


def _discretize_stress(df: pd.DataFrame) -> pd.DataFrame:
    return discretize_column(
        df, "Stress Level",
        bins=[2, 4, 6, 9],
        labels=["Bajo", "Medio", "Alto"],
        new_column="stress_cat",
    )


def _discretize_activity(df: pd.DataFrame) -> pd.DataFrame:
    return discretize_column(
        df, "Physical Activity Level",
        bins=[29, 45, 75, 91],
        labels=["Bajo", "Medio", "Alto"],
        new_column="activity_cat",
    )


def _discretize_heart_rate(df: pd.DataFrame) -> pd.DataFrame:
    return discretize_column(
        df, "Heart Rate",
        bins=[64, 67, 72, 87],
        labels=["Bajo", "Normal", "Alto"],
        new_column="heart_rate_cat",
    )


def _discretize_daily_steps(df: pd.DataFrame) -> pd.DataFrame:
    return discretize_column(
        df, "Daily Steps",
        bins=[2999, 5600, 8000, 10001],
        labels=["Sedentario", "Moderado", "Activo"],
        new_column="daily_steps_cat",
    )


def _discretize_bp_systolic(df: pd.DataFrame) -> pd.DataFrame:
    return discretize_column(
        df, "bp_systolic",
        bins=[0, 119, 129, 200],
        labels=["Normal", "Elevada", "Alta"],
        new_column="bp_systolic_cat",
    )


def _discretize_bp_diastolic(df: pd.DataFrame) -> pd.DataFrame:
    return discretize_column(
        df, "bp_diastolic",
        bins=[0, 79, 89, 200],
        labels=["Normal", "Elevada", "Alta"],
        new_column="bp_diastolic_cat",
    )


def run() -> pd.DataFrame:
    """Load cleaned.csv, apply all discretizations, and return one-hot encoded transactions."""
    print("=" * 60)
    print("  Association Rules - Data Preparation")
    print("=" * 60)

    df = load.load_csv("cleaned.csv", folder="processed")
    print(f"Loaded cleaned.csv: {df.shape}")

    # Discretize numeric columns
    df = _discretize_sleep_duration(df)
    df = _discretize_stress(df)
    df = _discretize_activity(df)
    df = _discretize_heart_rate(df)
    df = _discretize_daily_steps(df)
    df = _discretize_bp_systolic(df)
    df = _discretize_bp_diastolic(df)
    print("Discretized all numeric columns")

    # Group occupation
    df = group_occupation(df)
    print("Grouped occupations")

    # Ensure age_group and quality_group exist (should be in cleaned.csv)
    for col in ["age_group", "quality_group"]:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' in cleaned.csv")

    # Build list of columns to keep (categorical only)
    cat_columns = [
        "Gender",
        "age_group",
        "quality_group",
        "sleep_duration_cat",
        "stress_cat",
        "activity_cat",
        "heart_rate_cat",
        "daily_steps_cat",
        "bp_systolic_cat",
        "bp_diastolic_cat",
        "occupation_group",
        "BMI Category",
        "Sleep Disorder",
    ]

    # Drop all other columns
    cols_to_drop = [c for c in df.columns if c not in cat_columns]
    df_cat = df.drop(columns=cols_to_drop, errors="ignore")
    print(f"Categorical dataframe: {df_cat.shape}")

    # One-hot encode
    encoded = prepare_transactions(df_cat)
    print(f"Encoded transactions: {encoded.shape}")

    # Save prepared data
    out_dir = config.get_dataset_path("processed")
    prepared_path = os.path.join(out_dir, "prepared_association_rules.csv")
    df_cat.to_csv(prepared_path, index=False)
    print(f"Saved categorical data to: {prepared_path}")

    print("Preparation complete\n")
    return encoded
