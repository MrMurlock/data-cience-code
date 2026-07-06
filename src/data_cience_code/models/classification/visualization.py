import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns


def plot_support_vs_confidence(
    rules: pd.DataFrame,
    title: str = "Support vs Confidence",
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = None

    scatter = ax.scatter(
        rules["support"],
        rules["confidence"],
        c=rules["lift"],
        cmap="viridis",
        alpha=0.6,
        edgecolors="k",
        linewidth=0.5,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Lift")

    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return fig


def plot_top_rules(
    rules: pd.DataFrame,
    metric: str = "lift",
    n: int = 10,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    if title is None:
        title = f"Top {n} Rules by {metric.title()}"

    top = rules.nlargest(n, metric).copy()
    top["rule"] = top.apply(
        lambda r: (
            f"{', '.join(sorted(r['antecedents']))}"
            f" \u2192 {', '.join(sorted(r['consequents']))}"
        ),
        axis=1,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = None

    colors = sns.color_palette("viridis", n)
    bars = ax.barh(range(len(top)), top[metric], color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["rule"], fontsize=8)
    ax.set_xlabel(metric.title())
    ax.set_title(title)
    ax.invert_yaxis()

    for bar, val in zip(bars, top[metric]):
        ax.text(
            bar.get_width() + bar.get_width() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            fontsize=8,
        )

    return fig


def plot_comparison_bar(
    apriori_counts: dict[float, int],
    fp_counts: dict[float, int],
    time_apriori: float,
    time_fp: float,
    title: str = "Apriori vs FP-Growth",
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    supports = list(apriori_counts.keys())
    apriori_vals = [apriori_counts[s] for s in supports]
    fp_vals = [fp_counts[s] for s in supports]
    x = range(len(supports))
    width = 0.35

    ax1.bar(
        [i - width / 2 for i in x],
        apriori_vals,
        width,
        label="Apriori",
        alpha=0.8,
        color="#2ecc71",
    )
    ax1.bar(
        [i + width / 2 for i in x],
        fp_vals,
        width,
        label="FP-Growth",
        alpha=0.8,
        color="#3498db",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"s={s}" for s in supports], fontsize=9)
    ax1.set_title("Rule Count by min_support")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(
        ["Apriori", "FP-Growth"],
        [time_apriori, time_fp],
        color=["#2ecc71", "#3498db"],
        alpha=0.8,
    )
    ax2.set_title(f"Total Execution Time ({time_apriori + time_fp:.2f}s)")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_metrics_heatmap(
    rules: pd.DataFrame,
    metrics: list[str] | None = None,
    n: int = 15,
    title: str = "Rule Quality Matrix",
) -> plt.Figure:
    if metrics is None:
        metrics = ["support", "confidence", "lift", "leverage", "conviction"]
    top = rules.nlargest(n, "lift").copy()
    top["rule"] = top.apply(
        lambda r: (
            f"{', '.join(sorted(r['antecedents']))}"
            f" \u2192 {', '.join(sorted(r['consequents']))}"
        ),
        axis=1,
    )
    available = [m for m in metrics if m in top.columns]
    matrix = top[available].copy()
    normalized = (matrix - matrix.min()) / (matrix.max() - matrix.min() + 1e-10)

    fig, ax = plt.subplots(figsize=(max(8, len(available) * 2), max(6, n * 0.4)))
    sns.heatmap(
        normalized,
        annot=matrix.round(3),
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Normalized value"},
    )
    ax.set_yticklabels(top["rule"], fontsize=7, rotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Quality Metric")
    ax.set_ylabel("Rule")
    fig.tight_layout()
    return fig


def _item_to_variable(item: str) -> str:
    var = item.rsplit("_", 1)[0] if "_" in item else item
    return var


_VARIABLE_COLORS = {
    "Gender": "#e74c3c",
    "age_group": "#3498db",
    "quality_group": "#9b59b6",
    "sleep_duration_cat": "#1abc9c",
    "stress_cat": "#e67e22",
    "activity_cat": "#2ecc71",
    "heart_rate_cat": "#f1c40f",
    "daily_steps_cat": "#16a085",
    "bp_systolic_cat": "#c0392b",
    "bp_diastolic_cat": "#c0392b",
    "occupation_group": "#8e44ad",
    "BMI Category": "#d35400",
    "Sleep Disorder": "#2c3e50",
}


def _get_color(item: str) -> str:
    var = _item_to_variable(item)
    return _VARIABLE_COLORS.get(var, "#95a5a6")


def _get_variable_label(item: str) -> str:
    var = _item_to_variable(item)
    label_map = {
        "Gender": "Gender",
        "age_group": "Age",
        "quality_group": "Sleep Quality",
        "sleep_duration_cat": "Sleep Duration",
        "stress_cat": "Stress",
        "activity_cat": "Physical Activity",
        "heart_rate_cat": "Heart Rate",
        "daily_steps_cat": "Daily Steps",
        "bp_systolic_cat": "Systolic BP",
        "bp_diastolic_cat": "Diastolic BP",
        "occupation_group": "Occupation",
        "BMI Category": "BMI",
        "Sleep Disorder": "Sleep Disorder",
    }
    return label_map.get(var, var)


def _build_variable_legend(ax: plt.Axes):
    handles = []
    seen = set()
    for var, color in sorted(_VARIABLE_COLORS.items()):
        if var not in seen:
            label = _get_variable_label(var)
            handles.append(
                plt.Line2D(
                    [0], [0], marker="o", color="w",
                    markerfacecolor=color, markersize=10, label=label,
                )
            )
            seen.add(var)
    ax.legend(handles=handles, title="Variable", loc="upper left",
              bbox_to_anchor=(1.02, 1), fontsize=8, title_fontsize=9)


def plot_network_graph(
    rules: pd.DataFrame,
    top_n: int = 20,
    title: str = "Association Rules Network",
) -> plt.Figure:
    top = rules.nlargest(top_n, "lift").copy()
    G = nx.DiGraph()

    for _, row in top.iterrows():
        ant_items = list(row["antecedents"])
        cons_items = list(row["consequents"])
        for a in ant_items:
            for c in cons_items:
                if G.has_edge(a, c):
                    G[a][c]["weight"] = max(G[a][c]["weight"], row["lift"])
                    G[a][c]["lift"] = max(G[a][c]["lift"], row["lift"])
                else:
                    G.add_edge(a, c, weight=row["lift"], lift=row["lift"])

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

    node_colors = [_get_color(n) for n in G.nodes()]
    node_labels_short = {n: n.split("_", 1)[1] if "_" in n else n for n in G.nodes()}

    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    if edge_weights:
        min_w, max_w = min(edge_weights), max(edge_weights)
        edge_widths = [
            0.5 + 3 * (w - min_w) / (max_w - min_w + 1e-10)
            for w in edge_weights
        ]
    else:
        edge_widths = []

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        alpha=0.5,
        edge_color="#34495e",
        arrows=True,
        arrowsize=15,
        arrowstyle="->",
        connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=2000,
        edgecolors="black",
        linewidths=1.5,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels=node_labels_short,
        font_size=7,
        font_color="black",
        font_weight="bold",
    )

    _build_variable_legend(ax)
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_metric_distribution(
    rules: pd.DataFrame,
    metric: str = "lift",
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    if title is None:
        title = f"Distribution of {metric.title()}"

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = None

    sns.histplot(rules[metric], kde=True, ax=ax, color="#3498db", bins=30)
    ax.axvline(
        rules[metric].mean(), color="red", linestyle="--", linewidth=1.5,
        label=f"Mean = {rules[metric].mean():.3f}",
    )
    ax.axvline(
        rules[metric].median(), color="green", linestyle=":", linewidth=1.5,
        label=f"Median = {rules[metric].median():.3f}",
    )
    ax.set_xlabel(metric.title())
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    return fig
