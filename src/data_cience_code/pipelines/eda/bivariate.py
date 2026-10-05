"""Análisis bivariado del pipeline EDA.

Implementa la sección 4 de docs/1.EDA.md sobre las relaciones seleccionadas
(exploración visual previa + relevancia de dominio), no sobre todas las
combinaciones posibles de variables.

* numérico × numérico: Pearson y Spearman, scatter plots con vallas 1.5*IQR.
* categórico × numérico: boxplots y comparación de grupos; con 2 grupos se usa
  Welch t-test o Mann-Whitney, con más de 2 ANOVA o Kruskal-Wallis, según la
  aproximación a la normalidad por grupo (Shapiro-Wilk). Se reportan tamaños
  de efecto (Cohen's d, r rank-biserial, eta², epsilon²).
* categórico × categórico: tablas de contingencia, chi-cuadrado de
  independencia y V de Cramér.

Todos los resultados se guardan como artifacts con nombres lógicos
reutilizables por el reporte y la presentación (docs/1.EDA.md §11).
"""
from math import sqrt

import numpy as np
import pandas as pd
from scipy import stats

from data_cience_code.utils.plots import (
    corr_heatmap,
    contingency_heatmap,
    group_box_plot,
    scatter_plot,
)

from .config_eda import artifacts, save_image

# Relaciones numéricas seleccionadas (§4.1). El tercer elemento es el nombre
# lógico de la scatter y de las métricas de correlación del par.
NUMERIC_PAIRS = [
    ("Sleep Duration", "Quality of Sleep", "sleep_duration_quality"),
    ("Sleep Duration", "Physical Activity Level", "sleep_duration_activity"),
    ("Sleep Duration", "Stress Level", "sleep_duration_stress"),
    ("Sleep Duration", "Heart Rate", "sleep_duration_heart_rate"),
    ("bp_systolic", "bp_diastolic", "bp_scatter"),
]

# Comparaciones categórica × numérica seleccionadas (§4.2).
GROUP_VARIATES = {
    "Sleep Disorder": ("Sleep Duration", "Quality of Sleep", "Stress Level", "Heart Rate"),
    "BMI Category": ("Sleep Duration", "Quality of Sleep", "Heart Rate"),
    "Gender": ("Sleep Duration", "Quality of Sleep", "Stress Level", "Heart Rate"),
    "Occupation": ("Sleep Duration", "Quality of Sleep", "Heart Rate"),
}

# Boxplots publicados como artifacts: relaciones destacadas por el reporte.
BOXPLOT_IMAGES = {
    ("Sleep Disorder", "Sleep Duration"): "disorder_sleep_duration_boxplot",
    ("Sleep Disorder", "Quality of Sleep"): "disorder_quality_of_sleep_boxplot",
    ("Sleep Disorder", "Stress Level"): "disorder_stress_level_boxplot",
    ("Sleep Disorder", "Heart Rate"): "disorder_heart_rate_boxplot",
    ("BMI Category", "Sleep Duration"): "bmi_sleep_duration_boxplot",
    ("BMI Category", "Heart Rate"): "bmi_heart_rate_boxplot",
    ("Gender", "Quality of Sleep"): "gender_quality_of_sleep_boxplot",
    ("Occupation", "Sleep Duration"): "occupation_sleep_duration_boxplot",
}

# Pares categóricos seleccionados (§4.3): contingencia + chi² + V de Cramér.
CATEGORICAL_PAIRS = [
    ("Occupation", "Sleep Disorder", "occupation_sleep_disorder"),
    ("BMI Category", "Sleep Disorder", "bmi_sleep_disorder"),
    ("Gender", "Sleep Disorder", "gender_sleep_disorder"),
]


def numeric_relations(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson y Spearman para los pares numéricos seleccionados.

    Por cada par guarda métricas individuales `{slug}_pearson` y
    `{slug}_spearman` con el coeficiente y su p-value.
    """
    rows = []
    for x, y, slug in NUMERIC_PAIRS:
        data = df[[x, y]].dropna()
        if data[x].nunique() < 2 or data[y].nunique() < 2:
            continue
        pearson = stats.pearsonr(data[x], data[y])
        spearman = stats.spearmanr(data[x], data[y])
        artifacts.save_metric(
            f"{slug}_pearson",
            {"pearson_r": round(float(pearson[0]), 3), "p_value": float(pearson[1])},
        )
        artifacts.save_metric(
            f"{slug}_spearman",
            {"spearman_rho": round(float(spearman[0]), 3), "p_value": float(spearman[1])},
        )
        rows.append(
            {
                "x": x,
                "y": y,
                "n": int(len(data)),
                "pearson_r": round(float(pearson[0]), 3),
                "pearson_p": float(pearson[1]),
                "spearman_rho": round(float(spearman[0]), 3),
                "spearman_p": float(spearman[1]),
            }
        )
    return pd.DataFrame(rows)


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Matriz de correlación (pearson o spearman) entre todas las variables numéricas."""
    corr = df.select_dtypes(include=[np.number]).corr(method=method).round(3)
    return corr.reset_index().rename(columns={"index": "column"})


def _cohens_d(group_a: pd.Series, group_b: pd.Series) -> float:
    """Diferencia estandarizada de medias (signo negativo: grupo a < grupo b)."""
    na, nb = len(group_a), len(group_b)
    sa, sb = float(group_a.std(ddof=1)), float(group_b.std(ddof=1))
    pooled = sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((group_a.mean() - group_b.mean()) / pooled)


def _rank_biserial(group_a: pd.Series, group_b: pd.Series, u_a: float) -> float:
    """Correlación rank-biserial de Mann-Whitney: 2U/(n1·n2) − 1.

    Positivo cuando el primer grupo tiende a valores mayores.
    """
    total = len(group_a) * len(group_b)
    if total == 0:
        return 0.0
    return float(2.0 * u_a / total - 1.0)


def _eta_squared(groups: list[pd.Series]) -> float:
    """Requiere más de dos grupos, se usa como tamaño de efecto para ANOVA."""
    all_values = pd.concat(groups)
    grand_mean = all_values.mean()
    ss_between = sum(len(g) * float(g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = float(((all_values - grand_mean) ** 2).sum())
    return ss_between / ss_total if ss_total > 0 else 0.0


def _epsilon_squared(h_statistic: float, n: int) -> float:
    """Tamaño de efecto epsilon² para Kruskal-Wallis: H(n+1)/(n(n−1))."""
    if n <= 1:
        return 0.0
    return float(h_statistic * (n + 1) / (n * (n - 1)))


def _normality_ok(groups: list[pd.Series]) -> bool:
    """Shapiro-Wilk por grupo (p > 0.05).

    Con menos de 3 observaciones por grupo el supuesto no puede evaluarse:
    se responde 'False' y se prefiere un test no paramétrico.
    """
    if min(len(g) for g in groups) < 3:
        return False
    for g in groups:
        if float(stats.shapiro(g).pvalue) <= 0.05:
            return False
    return True


def _normality_label(groups: list[pd.Series]) -> str:
    if min(len(g) for g in groups) < 3:
        return "grupo pequeño (n < 3)"
    return "no normal (Shapiro p <= 0.05)"


def group_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    """n, media, sd y mediana por grupo para cada comparación seleccionada."""
    rows = []
    for cat, numeric_vars in GROUP_VARIATES.items():
        if cat not in df.columns:
            continue
        for num in numeric_vars:
            if num not in df.columns:
                continue
            grouped = df[[cat, num]].dropna().groupby(cat)[num]
            for group, s in grouped:
                rows.append(
                    {
                        "categorical_var": cat,
                        "numeric_var": num,
                        "group": str(group),
                        "n": int(len(s)),
                        "mean": round(float(s.mean()), 2),
                        "std": round(float(s.std()), 2),
                        "median": round(float(s.median()), 2),
                    }
                )
    return pd.DataFrame(rows)


def group_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Compara un numérico entre grupos con el test según supuesto y nº de grupos.

    Dos grupos: Welch t-test (normalidad aceptada) o Mann-Whitney.
    Más de dos: ANOVA (normalidad aceptada) o Kruskal-Wallis.
    Guarda boxplots de las relaciones destacadas en BOXPLOT_IMAGES.
    """
    rows = []
    for cat, numeric_vars in GROUP_VARIATES.items():
        if cat not in df.columns:
            continue
        for num in numeric_vars:
            if num not in df.columns:
                continue
            data = df[[cat, num]].dropna()
            group_names = sorted(data[cat].unique())
            if len(group_names) < 2:
                continue
            groups = [data.loc[data[cat] == name, num] for name in group_names]

            if len(groups) == 2:
                if _normality_ok(groups):
                    res = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                    outcome = {
                        "test": "Welch t-test",
                        "statistic": round(float(res.statistic), 3),
                        "p_value": float(res.pvalue),
                        "effect_name": "cohens_d",
                        "effect_size": round(abs(_cohens_d(groups[0], groups[1])), 3),
                        "normality_check": "normal (Shapiro p > 0.05)",
                    }
                else:
                    res = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
                    outcome = {
                        "test": "Mann-Whitney U",
                        "statistic": round(float(res.statistic), 3),
                        "p_value": float(res.pvalue),
                        "effect_name": "rank_biserial",
                        "effect_size": round(
                            abs(_rank_biserial(groups[0], groups[1], float(res.statistic))), 3
                        ),
                        "normality_check": _normality_label(groups),
                    }
            else:
                if _normality_ok(groups):
                    res = stats.f_oneway(*groups)
                    outcome = {
                        "test": "ANOVA",
                        "statistic": round(float(res.statistic), 3),
                        "p_value": float(res.pvalue),
                        "effect_name": "eta_squared",
                        "effect_size": round(_eta_squared(groups), 3),
                        "normality_check": "normal (Shapiro p > 0.05)",
                    }
                else:
                    res = stats.kruskal(*groups)
                    outcome = {
                        "test": "Kruskal-Wallis",
                        "statistic": round(float(res.statistic), 3),
                        "p_value": float(res.pvalue),
                        "effect_name": "epsilon_squared",
                        "effect_size": round(_epsilon_squared(float(res.statistic), int(len(data))), 3),
                        "normality_check": _normality_label(groups),
                    }

            rows.append(
                {
                    "categorical_var": cat,
                    "numeric_var": num,
                    "n_groups": len(groups),
                    "min_group_n": int(min(len(g) for g in groups)),
                    **outcome,
                }
            )

            key = (cat, num)
            if key in BOXPLOT_IMAGES:
                fig, _ = group_box_plot(df, cat, num, title=f"{num} by {cat}")
                save_image(BOXPLOT_IMAGES[key], fig)

    return pd.DataFrame(rows)


def categorical_associations(df: pd.DataFrame) -> pd.DataFrame:
    """Chi-cuadrado + V de Cramér por par categórico seleccionado (§4.3).

    También guarda la tabla de contingencia, la métrica `cramers_v_{slug}`
    y un heatmap de proporciones por fila. `min_expected` y `pct_cells_lt5`
    permiten evaluar el supuesto de conteos esperados suficientes.
    """
    rows = []
    for var_x, var_y, slug in CATEGORICAL_PAIRS:
        ct = pd.crosstab(df[var_x], df[var_y])
        res = stats.chi2_contingency(ct)
        n = int(ct.to_numpy().sum())
        k = min(ct.shape) - 1
        cramers_v = float(sqrt(res.statistic / (n * k))) if n > 0 and k > 0 else 0.0
        expected = np.asarray(res.expected_freq)

        artifacts.save_table(f"contingency_{slug}", ct.reset_index())
        artifacts.save_metric(f"cramers_v_{slug}", round(cramers_v, 3))

        fig, _ = contingency_heatmap(
            df, var_x, var_y, title=f"{var_x} x {var_y} (% dentro de {var_x})"
        )
        save_image(f"{slug}_prop", fig)

        rows.append(
            {
                "var_x": var_x,
                "var_y": var_y,
                "n": n,
                "chi2": round(float(res.statistic), 2),
                "dof": int(res.dof),
                "p_value": float(res.pvalue),
                "cramers_v": round(cramers_v, 3),
                "min_expected": round(float(expected.min()), 2),
                "pct_cells_lt5": round(float((expected < 5).mean() * 100), 1),
            }
        )
    return pd.DataFrame(rows)


def save_bivariate_artifacts(df: pd.DataFrame) -> None:
    """Genera y guarda todos los artifacts del análisis bivariado."""
    # Numérico × numérico
    artifacts.save_table("bivariate_numeric", numeric_relations(df))
    artifacts.save_table("correlation_matrix", correlation_matrix(df))
    artifacts.save_table("correlation_matrix_spearman", correlation_matrix(df, method="spearman"))
    fig, _ = corr_heatmap(df)
    save_image("corr_heatmap", fig)
    # Spearman es más robusta ante outliers: comparar ambas matrices permite
    # detectar relaciones distorsionadas por valores extremos (§5.1 del reporte).
    fig, _ = corr_heatmap(df, method="spearman", title="Matriz de correlación (Spearman)")
    save_image("corr_heatmap_spearman", fig)
    for x, y, slug in NUMERIC_PAIRS:
        if x in df.columns and y in df.columns:
            fig, _ = scatter_plot(df, x, y, title=f"{y} vs {x}")
            save_image(slug, fig)

    # Categórico × numérico
    artifacts.save_table("group_descriptive", group_descriptives(df))
    artifacts.save_table("group_tests", group_comparisons(df))

    # Categórico × categórico
    artifacts.save_table("categorical_associations", categorical_associations(df))
