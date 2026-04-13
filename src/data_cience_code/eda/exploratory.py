import pandas as pd

def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Return statistical summary of numeric columns."""
    return df.describe()


def describe_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Return value counts for categorical columns as a formatted DataFrame."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    result = pd.DataFrame()
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False)
        vc.name = col
        result = pd.concat([result, vc], axis=1)
    return result.fillna("-")


def get_distribution(df: pd.DataFrame, column: str) -> pd.Series:
    """Return value counts for a specific column."""
    return df[column].value_counts()


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return correlation matrix for numeric columns."""
    return df.select_dtypes(include=["number"]).corr()


def get_mean(df: pd.DataFrame, column: str) -> float:
    """Return mean of a numeric column."""
    return df[column].mean()


def get_median(df: pd.DataFrame, column: str) -> float:
    """Return median of a numeric column."""
    return df[column].median()


def get_std(df: pd.DataFrame, column: str) -> float:
    """Return standard deviation of a numeric column."""
    return df[column].std()