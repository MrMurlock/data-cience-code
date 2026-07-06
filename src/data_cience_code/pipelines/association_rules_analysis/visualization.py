import os
from typing import Any

import matplotlib.pyplot as plt

from data_cience_code.config import config
from data_cience_code.models.classification.association_rules import (
    filter_rules_by_antecedent,
    filter_rules_by_rhs,
)
from data_cience_code.models.classification.visualization import (
    plot_comparison_bar,
    plot_metric_distribution,
    plot_metrics_heatmap,
    plot_network_graph,
    plot_support_vs_confidence,
    plot_top_rules,
)


def _get_figures_dir() -> str:
    out_dir = config.get_output_path("figures", "association_rules")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _deduplicate_rules(rules):
    if rules is None or rules.empty:
        return rules
    return rules.drop_duplicates(
        subset=["antecedents", "consequents"]
    ).reset_index(drop=True)


def _save_fig(fig, filename: str, out_dir: str):
    if fig is None:
        return
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")


def run(results: dict[str, Any]):
    print("=" * 60)
    print("  Association Rules - Visualization")
    print("=" * 60)

    out_dir = _get_figures_dir()
    rules_apriori = _deduplicate_rules(results.get("rules_apriori"))
    rules_fp = _deduplicate_rules(results.get("rules_fp"))

    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Support vs Confidence scatter
    if rules_apriori is not None and not rules_apriori.empty:
        fig = plot_support_vs_confidence(
            rules_apriori, title="Apriori: Support vs Confidence (color = Lift)"
        )
        _save_fig(fig, "apriori_support_vs_confidence.png", out_dir)

    if rules_fp is not None and not rules_fp.empty:
        fig = plot_support_vs_confidence(
            rules_fp, title="FP-Growth: Support vs Confidence (color = Lift)"
        )
        _save_fig(fig, "fp_growth_support_vs_confidence.png", out_dir)

    # 2. Top rules by lift
    if rules_apriori is not None and not rules_apriori.empty:
        fig = plot_top_rules(rules_apriori, metric="lift", n=10,
                             title="Top 10 Apriori Rules by Lift")
        _save_fig(fig, "top_rules_apriori.png", out_dir)

    if rules_fp is not None and not rules_fp.empty:
        fig = plot_top_rules(rules_fp, metric="lift", n=10,
                             title="Top 10 FP-Growth Rules by Lift")
        _save_fig(fig, "top_rules_fp_growth.png", out_dir)

    # 3. Comparison bar chart
    apriori_counts = results.get("apriori_counts", {})
    fp_counts = results.get("fp_counts", {})
    time_a = results.get("total_time_apriori", 0.0)
    time_f = results.get("total_time_fp", 0.0)

    if apriori_counts and fp_counts:
        fig = plot_comparison_bar(
            apriori_counts, fp_counts, time_a, time_f,
            title="Apriori vs FP-Growth Comparison",
        )
        _save_fig(fig, "comparison_bar.png", out_dir)

    # 4. Rule quality matrix (heatmap)
    if rules_apriori is not None and not rules_apriori.empty:
        fig = plot_metrics_heatmap(
            rules_apriori, n=15,
            title="Rule Quality Matrix (Top 15 by Lift)",
        )
        _save_fig(fig, "rule_quality_matrix.png", out_dir)

    # 5. Network graph
    if rules_apriori is not None and not rules_apriori.empty:
        fig = plot_network_graph(
            rules_apriori, top_n=25,
            title="Association Rules Network (Top 25 by Lift)",
        )
        _save_fig(fig, "network_graph.png", out_dir)

    # 6. Metric distributions
    if rules_apriori is not None and not rules_apriori.empty:
        fig = plot_metric_distribution(
            rules_apriori, metric="support",
            title="Distribution of Support",
        )
        _save_fig(fig, "distribution_support.png", out_dir)

        fig = plot_metric_distribution(
            rules_apriori, metric="confidence",
            title="Distribution of Confidence",
        )
        _save_fig(fig, "distribution_confidence.png", out_dir)

        fig = plot_metric_distribution(
            rules_apriori, metric="lift",
            title="Distribution of Lift",
        )
        _save_fig(fig, "distribution_lift.png", out_dir)

    print("Visualization complete.\n")


_FILTER_CONFIGS = [
    {
        "slug": "insomnio",
        "title": "Insomnio",
        "filter_fn": lambda r: filter_rules_by_rhs(r, "Sleep Disorder_Insomnia"),
    },
    {
        "slug": "apnea",
        "title": "Apnea del Sueño",
        "filter_fn": lambda r: filter_rules_by_rhs(r, "Sleep Disorder_Sleep Apnea"),
    },
    {
        "slug": "sin_trastorno",
        "title": "Sin Trastorno",
        "filter_fn": lambda r: filter_rules_by_rhs(r, "Sleep Disorder_None"),
    },
    {
        "slug": "ocupacion",
        "title": "Ocupación en Antecedente",
        "filter_fn": lambda r: filter_rules_by_antecedent(r, "occupation_group_"),
    },
]


def run_filtered(results: dict[str, Any]) -> dict[str, list[str]]:
    """Generate filtered visualizations for specific rule subsets.

    Returns dict of {slug: [list of saved file paths]}.
    """
    print("=" * 60)
    print("  Association Rules - Filtered Visualizations")
    print("=" * 60)

    rules = _deduplicate_rules(results.get("rules_apriori"))
    out_base = os.path.join(_get_figures_dir(), "filter_by")
    os.makedirs(out_base, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    all_paths: dict[str, list[str]] = {}

    for cfg in _FILTER_CONFIGS:
        slug = cfg["slug"]
        title = cfg["title"]
        filtered = cfg["filter_fn"](rules)

        if filtered.empty:
            print(f"  [SKIP] {slug}: no rules found")
            all_paths[slug] = []
            continue

        print(f"\n  [{slug}] {len(filtered)} rules for '{title}'")

        out_dir = os.path.join(out_base, slug)
        os.makedirs(out_dir, exist_ok=True)
        saved: list[str] = []

        # Top rules
        fig = plot_top_rules(
            filtered, metric="lift", n=10,
            title=f"Top 10 Rules — {title}",
        )
        _save_fig(fig, "top_rules.png", out_dir)
        saved.append(os.path.join(out_dir, "top_rules.png"))

        # Quality matrix
        fig = plot_metrics_heatmap(
            filtered, n=15,
            title=f"Rule Quality Matrix — {title}",
        )
        _save_fig(fig, "quality_matrix.png", out_dir)
        saved.append(os.path.join(out_dir, "quality_matrix.png"))

        # Network graph
        fig = plot_network_graph(
            filtered, top_n=20,
            title=f"Association Network — {title}",
        )
        _save_fig(fig, "network_graph.png", out_dir)
        saved.append(os.path.join(out_dir, "network_graph.png"))

        # Support vs Confidence
        fig = plot_support_vs_confidence(
            filtered,
            title=f"Support vs Confidence — {title}",
        )
        _save_fig(fig, "support_vs_confidence.png", out_dir)
        saved.append(os.path.join(out_dir, "support_vs_confidence.png"))

        # Distribution of lift
        fig = plot_metric_distribution(
            filtered, metric="lift",
            title=f"Distribution of Lift — {title}",
        )
        _save_fig(fig, "distribution_lift.png", out_dir)
        saved.append(os.path.join(out_dir, "distribution_lift.png"))

        all_paths[slug] = saved

    print("\nFiltered visualizations complete.\n")
    return all_paths
