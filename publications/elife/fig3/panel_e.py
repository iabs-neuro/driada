"""
Panel E — Selectivity prevalence violin plot.

Delegated panel: receives a single Axes from the orchestrator.
"""

import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import elife_paths


FEATURES = {
    "locomotion":        "run",
    "place_quad":        "place",
    "bodydirection_harm2": "body dir.",
    "walls":             "walls",
    "centertrue":        "center",
    "rear":              "rearing",
    "walk":              "walk",
    "headdirection_harm2": "head dir.",
    "speed_quad":        "speed",
    "rest":              "rest",
    "freezing":          "freezing",
}

TABLES_DIR = elife_paths.TABLES_DIR


def load_selectivity_table(tables_dir=None):
    """Significant neuron-feature pairs after disentanglement, one row per pair.

    Reads the per-session significance tables the cross-analysis database is
    built from and returns the columns the panels use: mouse, session,
    neuron_idx, feature. A session where the place representation was not
    applied keeps the plain name ``place``; it is counted as ``place_quad``.
    A merged label ``place-<zone>`` is a feature of its own: the panels count
    it neither as place nor as the zone, and the neuron stays selective.
    """
    import ast
    import glob
    import re

    rows = []
    for path in sorted(glob.glob(os.path.join(tables_dir or TABLES_DIR, "*INTENSE significance.csv"))):
        m = re.match(r"^(.+?)_([^_]+)_([^_\s]+)\s+INTENSE", os.path.basename(path))
        mouse, session = m.group(2), m.group(3)
        tab = pd.read_csv(path, index_col=0)
        for feat in tab.columns:
            for idx, cell in tab[feat].items():
                if not isinstance(cell, str) or cell == "{}":
                    continue
                entry = ast.literal_eval(re.sub(r"np\.(True|False)_", r"\1", cell))
                if entry.get("stage2", False):
                    rows.append({"mouse": mouse, "session": session, "neuron_idx": int(idx),
                                 "feature": "place_quad" if feat == "place" else feat})
    return pd.DataFrame(rows)


# Colors must match SEL_COLORS in fig4/panel_c_network.py (grouped by selectivity type)
COLORS = {
    "run":        "#FBC15E",   # locomotion group — amber
    "place":      "#E24A33",   # place group — red
    "body dir.":  "#348ABD",   # HD group — blue
    "walls":      "#2CA02C",   # spatial group — green
    "center":     "#2CA02C",   # spatial group — green
    "rearing":    "#E377C2",   # rear group — pink
    "walk":       "#FBC15E",   # locomotion group — amber
    "head dir.":  "#348ABD",   # HD group — blue
    "speed":      "#FF7F0E",   # speed group — orange
    "rest":       "#17BECF",   # rest group — teal
    "freezing":   "#17BECF",   # rest group — teal
}


def create_panel_E(ax, style, panel_size_cm):
    """Draw selectivity prevalence violin plot on the provided axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Pre-created axes from the master GridSpec.
    style : StylePreset
        Shared style instance.
    panel_size_cm : tuple of (width_cm, height_cm)
        Physical size for style scaling.
    """
    print("  [Panel E] Loading disentangled selectivity tables...")
    df = load_selectivity_table()

    # Compute fraction-of-selective per feature per (mouse, session)
    sel_totals = (df.groupby(["mouse", "session"])["neuron_idx"]
                  .nunique()
                  .reset_index(name="n_sel"))

    sig_counts = (df.groupby(["mouse", "session", "feature"])
                  .size()
                  .reset_index(name="n_sig"))

    records = []
    for _, row in sel_totals.iterrows():
        mouse, session, n_sel = row["mouse"], row["session"], row["n_sel"]
        if n_sel == 0:
            continue
        for feat_key, feat_label in FEATURES.items():
            mask = ((sig_counts["mouse"] == mouse) &
                    (sig_counts["session"] == session) &
                    (sig_counts["feature"] == feat_key))
            n_sig = sig_counts.loc[mask, "n_sig"].sum()
            pct = 100.0 * n_sig / n_sel
            records.append({
                "mouse": mouse,
                "session": session,
                "feature": feat_label,
                "pct_selective": pct,
            })

    pct_df_raw = pd.DataFrame(records)
    pct_df = (pct_df_raw
              .groupby(["mouse", "feature"], as_index=False)["pct_selective"]
              .mean())

    medians = pct_df.groupby("feature")["pct_selective"].median()
    feat_order = medians.sort_values(ascending=False).index.tolist()

    # Draw violin + strip
    positions = np.arange(len(feat_order))
    violin_data = [pct_df.loc[pct_df["feature"] == f, "pct_selective"].values
                   for f in feat_order]

    parts = ax.violinplot(violin_data, positions=positions, showmedians=False,
                          showextrema=False, widths=0.7)

    for pc, feat in zip(parts["bodies"], feat_order):
        color = COLORS.get(feat, "#999999")
        pc.set_facecolor(color)
        pc.set_edgecolor("none")
        pc.set_alpha(0.35)

    # Overlay individual points (jittered)
    rng = np.random.default_rng(42)
    for i, feat in enumerate(feat_order):
        vals = pct_df.loc[pct_df["feature"] == feat, "pct_selective"].values
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        color = COLORS.get(feat, "#999999")
        ax.scatter(positions[i] + jitter, vals, s=12, alpha=0.6,
                   color=color, edgecolors="white", linewidths=0.3, zorder=3)

    # Overlay median markers
    for i, feat in enumerate(feat_order):
        med = medians[feat]
        ax.plot(positions[i], med, marker="_", markersize=14,
                markeredgewidth=2.5, color="k", zorder=4)

    ax.set_ylabel("Selective neurons (%)")
    ax.set_xlim(-0.6, len(feat_order) - 0.4)
    ax.set_ylim(bottom=0)

    ax.set_xticks(positions.tolist())
    ax.set_xticklabels(feat_order, rotation=40, ha="right")

    style.apply_to_axes(ax, panel_size_cm, "cm")

    print(f"  [Panel E] {len(feat_order)} features, {pct_df['mouse'].nunique()} mice")


def create_nsel_bars(ax, style, panel_size_cm):
    """Draw bar chart of % neurons selective to 1, 2, 3+ features.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    style : StylePreset
    panel_size_cm : tuple of (width_cm, height_cm)
    """
    print("  [Panel C-bars] Loading disentangled selectivity tables...")
    df = load_selectivity_table()

    # Count features per neuron per (mouse, session)
    n_total_per_ms = (df.groupby(["mouse", "session"])["neuron_idx"]
                      .nunique().reset_index(name="n_total"))
    feat_per_neuron = (df.groupby(["mouse", "session", "neuron_idx"])["feature"]
                       .nunique().reset_index(name="n_feat"))

    records = []
    for _, row in n_total_per_ms.iterrows():
        mouse, session, n_total = row["mouse"], row["session"], row["n_total"]
        if n_total == 0:
            continue
        sub = feat_per_neuron[(feat_per_neuron["mouse"] == mouse) &
                              (feat_per_neuron["session"] == session)]
        pct_1 = 100.0 * (sub["n_feat"] == 1).sum() / n_total
        pct_2 = 100.0 * (sub["n_feat"] == 2).sum() / n_total
        pct_3 = 100.0 * (sub["n_feat"] >= 3).sum() / n_total
        records.append({"pct_1": pct_1, "pct_2": pct_2, "pct_3": pct_3})

    stats_df = pd.DataFrame(records)
    from scipy import stats as sp_stats
    means = [stats_df["pct_1"].mean(), stats_df["pct_2"].mean(),
             stats_df["pct_3"].mean()]
    n = len(stats_df)
    # 95% CI: mean ± t * SEM
    t_crit = sp_stats.t.ppf(0.975, df=n - 1)
    ci95 = [t_crit * stats_df[c].sem() for c in ["pct_1", "pct_2", "pct_3"]]

    colors = ["#4daf4a", "#377eb8", "#e41a1c"]
    ax.bar([1, 2, 3], means, color=colors, edgecolor="black",
           linewidth=0.3, width=0.65)
    ax.errorbar([1, 2, 3], means, yerr=ci95, fmt="none", ecolor="black",
                capsize=2.5, linewidth=0.6, capthick=0.5)

    ax.set_xlabel("# of features")
    ax.set_ylabel("Selective neurons (%)")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["1", "2", "3+"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    style.apply_to_axes(ax, panel_size_cm, "cm")
    print(f"  [Panel C-bars] 1={means[0]:.1f}%, 2={means[1]:.1f}%, 3+={means[2]:.1f}%")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from driada.utils.publication.style import StylePreset

    fig, ax = plt.subplots(figsize=(14 / 2.54, 5 / 2.54))
    style = StylePreset.publication_default().copy(
        base_spine_width=1.5, base_tick_width=1.5, base_tick_length=4,
        base_tick_pad=5, base_tick_labelsize=9, base_label_size=10,
        base_title_size=11, tight_layout=False,
    )
    create_panel_E(ax, style, (14, 5))
    fig.subplots_adjust(bottom=0.30, left=0.10, right=0.97, top=0.95)
    fig.savefig(elife_paths.out_path("panel_e_standalone.png"), dpi=300,
                facecolor="white", edgecolor="none")
    print("Saved standalone panel_e")
