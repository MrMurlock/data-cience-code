import os
from typing import Any

import pandas as pd

from data_cience_code.config import config
from data_cience_code.models.classification.association_rules import (
    compare_rules,
    filter_rules_by_rhs,
    run_apriori,
    run_fp_growth,
    rules_to_dataframe,
)


def _get_output_dir() -> str:
    out_dir = config.get_output_path("reports", "association_rules")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _log_separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run(encoded_df: pd.DataFrame) -> dict[str, Any]:
    """Run Apriori and FP-Growth at multiple min_support levels, compare results."""
    _log_separator("Association Rules Mining")

    encoded_df = encoded_df.astype(bool)
    print(f"Transaction shape: {encoded_df.shape}")
    print(f"Unique items: {encoded_df.shape[1]}")

    min_supports = [0.1, 0.15, 0.2, 0.25]
    max_len = 3
    metric = "confidence"
    min_threshold = 0.5

    all_rules_apriori = []
    all_rules_fp = []
    apriori_counts: dict[float, int] = {}
    fp_counts: dict[float, int] = {}
    total_time_apriori = 0.0
    total_time_fp = 0.0

    for ms in min_supports:
        print(f"\n--- min_support = {ms} ---")

        rules_a, time_a = run_apriori(
            encoded_df, min_support=ms, metric=metric, min_threshold=min_threshold,
            max_len=max_len,
        )
        rules_f, time_f = run_fp_growth(
            encoded_df, min_support=ms, metric=metric, min_threshold=min_threshold,
            max_len=max_len,
        )

        print(f"  Apriori:    {len(rules_a):>4} rules in {time_a:.4f}s")
        print(f"  FP-Growth: {len(rules_f):>4} rules in {time_f:.4f}s")

        apriori_counts[ms] = len(rules_a)
        fp_counts[ms] = len(rules_f)
        total_time_apriori += time_a
        total_time_fp += time_f

        if not rules_a.empty:
            rules_a["min_support"] = ms
            all_rules_apriori.append(rules_a)
        if not rules_f.empty:
            rules_f["min_support"] = ms
            all_rules_fp.append(rules_f)

    # Combine all rules
    rules_apriori = pd.concat(all_rules_apriori, ignore_index=True) if all_rules_apriori else pd.DataFrame()
    rules_fp = pd.concat(all_rules_fp, ignore_index=True) if all_rules_fp else pd.DataFrame()

    # Comparison across all supports
    _log_separator("Comparison Summary")
    comparison = compare_rules(rules_apriori, rules_fp)
    print(f"  Apriori   -> {comparison['total_a']} total, {comparison['unique_a']} unique rules")
    print(f"  FP-Growth -> {comparison['total_b']} total, {comparison['unique_b']} unique rules")
    print(f"  Unique overlap: {comparison['overlap']} "
          f"({comparison['overlap_pct_a']}% of Apriori unique)")
    print(f"  Total time Apriori:     {total_time_apriori:.4f}s")
    print(f"  Total time FP-Growth:   {total_time_fp:.4f}s")

    # Per-support comparison
    per_support_comparisons = {}
    for ms in min_supports:
        subset_a = rules_apriori[rules_apriori["min_support"] == ms] if not rules_apriori.empty else pd.DataFrame()
        subset_f = rules_fp[rules_fp["min_support"] == ms] if not rules_fp.empty else pd.DataFrame()
        per_support_comparisons[ms] = compare_rules(subset_a, subset_f)

    # Save all rules
    out_dir = _get_output_dir()

    if not rules_apriori.empty:
        clean_a = rules_to_dataframe(rules_apriori)
        clean_a.to_csv(os.path.join(out_dir, "rules_apriori.csv"), index=False)
        print(f"\nSaved apriori rules to: {out_dir}/rules_apriori.csv")

    if not rules_fp.empty:
        clean_f = rules_to_dataframe(rules_fp)
        clean_f.to_csv(os.path.join(out_dir, "rules_fp_growth.csv"), index=False)
        print(f"Saved fp-growth rules to: {out_dir}/rules_fp_growth.csv")

    # Save comparison summary
    comp_df = pd.DataFrame([
        {"min_support": ms, **per_support_comparisons[ms]}
        for ms in min_supports
    ])
    comp_df.to_csv(os.path.join(out_dir, "comparison_summary.csv"), index=False)
    print(f"Saved comparison summary to: {out_dir}/comparison_summary.csv")

    # Filter by Sleep Disorder in RHS (both approaches)
    _log_separator("Rules with Sleep Disorder in RHS")
    sd_keywords = ["Sleep Disorder"]
    for name, rules in [("Apriori", rules_apriori), ("FP-Growth", rules_fp)]:
        if rules.empty:
            continue
        sd_rules = rules[
            rules["consequents"].apply(
                lambda x: any(
                    any(kw in str(item) for kw in sd_keywords) for item in x
                )
            )
        ].copy().reset_index(drop=True)
        print(f"  {name}: {len(sd_rules)} rules with Sleep Disorder in RHS")
        if not sd_rules.empty:
            clean_sd = rules_to_dataframe(sd_rules)
            clean_sd.to_csv(
                os.path.join(out_dir, f"rules_{name.lower().replace('-', '_')}_sleep_disorder.csv"),
                index=False,
            )

    results = {
        "rules_apriori": rules_apriori,
        "rules_fp": rules_fp,
        "comparison": comparison,
        "per_support_comparisons": per_support_comparisons,
        "apriori_counts": apriori_counts,
        "fp_counts": fp_counts,
        "total_time_apriori": total_time_apriori,
        "total_time_fp": total_time_fp,
    }

    print("\nAssociation rules mining complete.\n")
    return results
