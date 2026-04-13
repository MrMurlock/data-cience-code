import pandas as pd


def get_shape(df: pd.DataFrame) -> tuple[int, int]:
    """Return (rows, columns) of dataframe."""
    return df.shape


def get_dtypes(df: pd.DataFrame) -> pd.Series:
    """Return data types of each column."""
    return df.dtypes


def get_null_counts(df: pd.DataFrame) -> pd.Series:
    """Return count of null values per column."""
    return df.isnull().sum()


def get_null_percentages(df: pd.DataFrame) -> pd.Series:
    """Return percentage of null values per column."""
    return (df.isnull().sum() / len(df)) * 100


def get_duplicates(df: pd.DataFrame) -> int:
    """Return count of duplicate rows."""
    return df.duplicated().sum()


def validate(df: pd.DataFrame) -> dict:
    """Run all validations and return summary dict."""
    return {
        "shape": get_shape(df),
        "dtypes": get_dtypes(df),
        "null_counts": get_null_counts(df),
        "null_percentages": get_null_percentages(df),
        "duplicates": get_duplicates(df),
    }