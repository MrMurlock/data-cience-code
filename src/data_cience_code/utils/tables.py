import pandas as pd


def df_to_markdown(df: pd.DataFrame, precision: int = 2) -> str:
    if df.empty:
        return ""
    df_fmt = df.copy()
    for c in df_fmt.select_dtypes(include=["float64", "float32"]).columns:
        df_fmt[c] = df_fmt[c].round(precision)
    return df_fmt.to_markdown(index=False)
