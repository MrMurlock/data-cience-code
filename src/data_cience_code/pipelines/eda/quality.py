import pandas as pd
import numpy as np


def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        rows.append(
            {
                "column": c,
                "dtype": str(df[c].dtype),
                "non_null_count": df[c].notna().sum(),
                "null_count": df[c].isna().sum(),
            }
        )
    ov = pd.DataFrame(rows)
    ov["null_pct"] = (ov["null_count"] / len(df) * 100).round(2)
    return ov


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    mv = df.isna().sum().reset_index()
    mv.columns = ["column", "null_count"]
    mv = mv[mv["null_count"] > 0].copy()
    mv["null_pct"] = (mv["null_count"] / len(df) * 100).round(2)
    return mv.sort_values("null_count", ascending=False)


def duplicate_count(df: pd.DataFrame) -> dict:
    return {"duplicate_count": int(df.duplicated().sum())}


def duplicate_stats(df: pd.DataFrame) -> dict:
    """Agregado de duplicados exactos: filas totales, duplicados y porcentaje."""
    rows = int(len(df))
    dup = int(df.duplicated().sum())
    return {
        "rows": rows,
        "duplicates": dup,
        "unique": rows - dup,
        "duplicate_pct": round(dup / rows * 100, 1) if rows > 0 else 0.0,
    }


def missing_raw_summary(df: pd.DataFrame) -> dict:
    """Faltantes del dataset crudo por columna ({columna: n}).

    Documenta lo que el preprocessing corrige posteriormente
    (ej. los 'None' de Sleep Disorder interpretados como faltantes).
    """
    counts = df.isna().sum()
    return {column: int(count) for column, count in counts.items() if count > 0}


def categorical_cardinality(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    rows = []
    for c in cat_cols:
        s = df[c].dropna()
        vc = s.value_counts()
        rows.append(
            {
                "column": c,
                "n_categories": int(vc.size),
                "top_category": vc.index[0] if len(vc) > 0 else None,
                "top_count": int(vc.iloc[0]) if len(vc) > 0 else 0,
                "top_pct": round((vc.iloc[0] / len(s) * 100) if len(s) > 0 else 0, 2),
            }
        )
    return pd.DataFrame(rows)


def numeric_ranges(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    rows = []
    for c in num_cols:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        out_count = int(((s < lower) | (s > upper)).sum())
        rows.append(
            {
                "column": c,
                "min": round(float(s.min()), 2),
                "q1": round(float(q1), 2),
                "median": round(float(s.median()), 2),
                "q3": round(float(q3), 2),
                "max": round(float(s.max()), 2),
                "mean": round(float(s.mean()), 2),
                "iqr": round(float(iqr), 2),
                "outlier_count_iqr15": out_count,
                "has_outliers": bool(out_count > 0),
            }
        )
    return pd.DataFrame(rows)
