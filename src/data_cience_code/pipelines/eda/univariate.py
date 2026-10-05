import numpy as np
import pandas as pd

from data_cience_code.utils.plots import bar_plot, box_plot, hist_plot

from .config_eda import artifacts, save_image


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    rows = []
    for c in num_cols:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        rows.append(
            {
                "column": c,
                "count": int(s.count()),
                "mean": round(float(s.mean()), 2),
                "std": round(float(s.std()), 2),
                "min": round(float(s.min()), 2),
                "q1": round(float(s.quantile(0.25)), 2),
                "median": round(float(s.median()), 2),
                "q3": round(float(s.quantile(0.75)), 2),
                "max": round(float(s.max()), 2),
                "skew": round(float(s.skew()), 3),
                "kurtosis": round(float(s.kurtosis()), 3),
            }
        )
    return pd.DataFrame(rows)


def numeric_skewness_metric(df: pd.DataFrame) -> dict:
    num_cols = df.select_dtypes(include=[np.number]).columns
    out = {}
    for c in num_cols:
        s = df[c].dropna()
        if len(s) > 0:
            out[c] = round(float(s.skew()), 3)
    return out


def numeric_kurtosis_metric(df: pd.DataFrame) -> dict:
    num_cols = df.select_dtypes(include=[np.number]).columns
    out = {}
    for c in num_cols:
        s = df[c].dropna()
        if len(s) > 0:
            out[c] = round(float(s.kurtosis()), 3)
    return out


def categorical_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    rows = []
    for c in cat_cols:
        s = df[c].dropna()
        vc = s.value_counts(dropna=False)
        total = len(s)
        for idx, cnt in vc.items():
            rows.append(
                {
                    "variable": c,
                    "category": str(idx),
                    "abs": int(cnt),
                    "rel_pct": round((cnt / total * 100) if total > 0 else 0, 2),
                }
            )
    return pd.DataFrame(rows)


# Gráficos univariados que se publican como artifacts, con sus nombres lógicos,
# referenciados luego por el reporte y la presentación.

HIST_PLOTS = {
    "Sleep Duration": "sleep_duration_hist",
    "Heart Rate": "heart_rate_distribution",
    "Physical Activity Level": "physical_activity_hist",
    "Stress Level": "stress_level_hist",
    "Quality of Sleep": "quality_of_sleep_hist",
}

BOX_PLOTS = {
    "Sleep Duration": "sleep_duration_boxplot",
    "bp_systolic": "bp_systolic_boxplot",
    "bp_diastolic": "bp_diastolic_boxplot",
}

BAR_PLOTS = {
    "Gender": "gender_distribution",
    "Occupation": "occupation_distribution",
    "Sleep Disorder": "sleep_disorder_distribution",
    "BMI Category": "bmi_category_distribution",
}


def save_univariate_artifacts(df: pd.DataFrame) -> None:
    """Resúmenes estadísticos y gráficos del análisis univariado (§3 del spec)."""
    artifacts.save_table("numeric_summary", numeric_summary(df))
    artifacts.save_metric("numeric_skewness", numeric_skewness_metric(df))
    artifacts.save_metric("numeric_kurtosis", numeric_kurtosis_metric(df))
    artifacts.save_table("categorical_frequencies", categorical_frequencies(df))

    for col, name in HIST_PLOTS.items():
        if col in df.columns:
            fig, _ = hist_plot(df, col)
            save_image(name, fig)

    for col, name in BOX_PLOTS.items():
        if col in df.columns:
            fig, _ = box_plot(df, col)
            save_image(name, fig)

    for col, name in BAR_PLOTS.items():
        if col in df.columns:
            fig, _ = bar_plot(df, col)
            save_image(name, fig)
