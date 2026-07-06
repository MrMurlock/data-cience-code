import time
from typing import Any

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth


def discretize_column(
    df: pd.DataFrame,
    column: str,
    bins: list[float],
    labels: list[str],
    new_column: str | None = None,
) -> pd.DataFrame:
    df = df.copy()
    new_col = new_column or f"{column}_cat"
    df[new_col] = pd.cut(df[column], bins=bins, labels=labels, right=False)
    return df


def group_occupation(
    df: pd.DataFrame,
    column: str = "Occupation",
    new_column: str = "occupation_group",
) -> pd.DataFrame:
    mapping = {
        "Doctor": "Healthcare",
        "Nurse": "Healthcare",
        "Engineer": "Technical",
        "Software Engineer": "Technical",
        "Scientist": "Technical",
        "Teacher": "Education",
        "Accountant": "Business",
        "Manager": "Business",
        "Salesperson": "Sales",
        "Sales Representative": "Sales",
        "Lawyer": "Legal",
    }
    df = df.copy()
    df[new_column] = df[column].map(mapping)
    return df


def prepare_transactions(
    df: pd.DataFrame,
    cols_to_drop: list[str] | None = None,
    prefix_sep: str = "_",
) -> pd.DataFrame:
    df = df.copy()
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop, errors="ignore")
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    encoded = pd.get_dummies(df[cat_cols], prefix_sep=prefix_sep, dtype=int)
    return encoded.astype(bool)


def run_apriori(
    encoded_df: pd.DataFrame,
    min_support: float = 0.05,
    metric: str = "confidence",
    min_threshold: float = 0.5,
    max_len: int | None = None,
) -> tuple[pd.DataFrame, float]:
    start = time.perf_counter()
    frequent_itemsets = apriori(
        encoded_df, min_support=min_support, use_colnames=True, max_len=max_len
    )
    rules = association_rules(
        frequent_itemsets, metric=metric, min_threshold=min_threshold
    )
    elapsed = time.perf_counter() - start
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    return rules, elapsed


def run_fp_growth(
    encoded_df: pd.DataFrame,
    min_support: float = 0.05,
    metric: str = "confidence",
    min_threshold: float = 0.5,
    max_len: int | None = None,
) -> tuple[pd.DataFrame, float]:
    start = time.perf_counter()
    frequent_itemsets = fpgrowth(
        encoded_df, min_support=min_support, use_colnames=True, max_len=max_len
    )
    rules = association_rules(
        frequent_itemsets, metric=metric, min_threshold=min_threshold
    )
    elapsed = time.perf_counter() - start
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    return rules, elapsed


def compare_rules(
    rules_a: pd.DataFrame, rules_b: pd.DataFrame
) -> dict[str, Any]:
    total_a = len(rules_a)
    total_b = len(rules_b)

    if total_a > 0 and total_b > 0:
        set_a = set(
            (tuple(sorted(a)), tuple(sorted(c)))
            for a, c in zip(rules_a["antecedents"], rules_a["consequents"])
        )
        set_b = set(
            (tuple(sorted(a)), tuple(sorted(c)))
            for a, c in zip(rules_b["antecedents"], rules_b["consequents"])
        )
        unique_a = len(set_a)
        unique_b = len(set_b)
        overlap = len(set_a & set_b)
    else:
        unique_a = total_a
        unique_b = total_b
        overlap = 0

    return {
        "total_a": total_a,
        "total_b": total_b,
        "unique_a": unique_a,
        "unique_b": unique_b,
        "overlap": overlap,
        "overlap_pct_a": round(overlap / unique_a * 100, 2) if unique_a > 0 else 0.0,
        "overlap_pct_b": round(overlap / unique_b * 100, 2) if unique_b > 0 else 0.0,
    }


def filter_rules_by_rhs(
    rules: pd.DataFrame, pattern: str
) -> pd.DataFrame:
    mask = rules["consequents"].apply(
        lambda x: any(pattern in str(item) for item in x)
    )
    return rules[mask].copy().reset_index(drop=True)


def filter_rules_by_antecedent(
    rules: pd.DataFrame, pattern: str
) -> pd.DataFrame:
    mask = rules["antecedents"].apply(
        lambda x: any(pattern in str(item) for item in x)
    )
    return rules[mask].copy().reset_index(drop=True)


def rules_to_dataframe(rules: pd.DataFrame) -> pd.DataFrame:
    df = rules.copy()
    df["antecedents"] = df["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    df["consequents"] = df["consequents"].apply(lambda x: ", ".join(sorted(x)))
    cols = [
        "antecedents", "consequents",
        "support", "confidence", "lift",
        "leverage", "conviction", "zhangs_metric",
        "antecedent support", "consequent support",
    ]
    return df[[c for c in cols if c in df.columns]]
