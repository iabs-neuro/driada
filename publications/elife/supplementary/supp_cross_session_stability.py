"""
Supplementary Figure: Cross-session selectivity stability.

For CellReg-matched neurons across 3 sessions per mouse, computes the
fraction that retain their primary INTENSE selectivity label. Uses the
retention code of tools/neuron_database, whose loader maps the type-based
feature names (place_quad, speed_quad, *_harm2) back to the plain ones
(place, speed, *_2d).

Protocol
--------
1. Build the FOF database (13 mice × 3 sessions) from the significance
   tables after disentanglement.
2. For each feature, compute retention_count_table: how many neurons
   are selective in 1, 2, or all 3 sessions.
3. Compute retention_enrichment (shuffle test) for significance.
4. Plot: (A) stacked bar per feature showing 1/2/3-session retention,
         (B) enrichment ratio (observed/expected) for 3-session retention.

Output: supp_cross_session_stability.pdf / .png in the output directory
"""

import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import elife_paths
elife_paths.add_tools_to_path()

FOF_DATA_DIR = elife_paths.DATA_ROOT
OUTPUT_DIR = elife_paths.OUTPUT_DIR

# Features to analyze (skip multivariate _2d features)
SKIP_FEATURES = {'bodydirection_2d', 'headdirection_2d', 'speed_2d'}

# Retention enrichment shuffles
N_ENRICHMENT_SHUFFLES = 1000
# A neuron-feature pair counts as selective on significance alone (p < 0.001), as in the
# rest of the paper; no threshold on the size of the mutual information.
MI_THRESHOLD = None


def load_fof_database():
    """Load FOF NeuronDatabase from INTENSE results."""
    import dataclasses
    from neuron_database.configs import EXPERIMENT_CONFIGS
    from neuron_database.loaders import load_experiment
    # the tables of the type-based INTENSE run are not in the default subfolder
    config = dataclasses.replace(EXPERIMENT_CONFIGS['FOF'],
                                 tables_subdir=elife_paths.TABLES_DIR)
    db = load_experiment('FOF', FOF_DATA_DIR, config=config)
    db.summary()
    return db


def compute_retention(db):
    """Compute retention tables and enrichment for all features.

    Returns
    -------
    results : dict
        {feature: {
            'retention_table': pd.DataFrame (mice × k-sessions),
            'enrichment': dict with observed, expected, ratio, p-value
        }}
    """
    from neuron_database.tables import retention_count_table, retention_enrichment

    features = sorted(set(db.features) - SKIP_FEATURES)
    print(f"\nComputing retention for {len(features)} features...")

    results = {}
    for feat in features:
        print(f"  {feat}...", end=" ")
        rt = retention_count_table(db, feat, mi_threshold=MI_THRESHOLD)
        total_row = rt.loc['total']
        print(f"  1-session={total_row.get(1, 0)}, "
              f"2-session={total_row.get(2, 0)}, "
              f"3-session={total_row.get(3, 0)}")

        # Enrichment: is 3-session retention more than chance?
        try:
            enrich = retention_enrichment(
                db, feat, n_shuffles=N_ENRICHMENT_SHUFFLES, mi_threshold=MI_THRESHOLD
            )
        except Exception as e:
            print(f"    Enrichment failed: {e}")
            enrich = None

        results[feat] = {
            'retention_table': rt,
            'enrichment': enrich,
        }

    return results


def plot_results(results, db):
    """Two-panel figure for cross-session stability."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features = sorted(results.keys())
    n_sessions = len(db.sessions)  # 3

    # Extract counts from total row
    counts_1only = []  # selective in exactly 1 session
    counts_2only = []  # selective in exactly 2
    counts_3 = []      # selective in all 3
    for feat in features:
        total = results[feat]['retention_table'].loc['total']
        # retention_count_table gives cumulative: k = at least k sessions
        # So: exactly_k = total[k] - total[k+1]
        at_least_1 = int(total.get(1, 0))
        at_least_2 = int(total.get(2, 0))
        at_least_3 = int(total.get(3, 0))
        counts_1only.append(at_least_1 - at_least_2)
        counts_2only.append(at_least_2 - at_least_3)
        counts_3.append(at_least_3)

    counts_1only = np.array(counts_1only)
    counts_2only = np.array(counts_2only)
    counts_3 = np.array(counts_3)
    totals = counts_1only + counts_2only + counts_3

    # Sort by total count (descending)
    order = np.argsort(-totals)
    features_sorted = [features[i] for i in order]
    c1 = counts_1only[order]
    c2 = counts_2only[order]
    c3 = counts_3[order]

    # --- Panel A: Stacked bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={'width_ratios': [2, 1]})

    ax = axes[0]
    x = np.arange(len(features_sorted))
    width = 0.7

    bars3 = ax.bar(x, c3, width, label=f'All {n_sessions} sessions',
                   color='#2a9d8f', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x, c2, width, bottom=c3, label='2 sessions',
                   color='#e9c46a', edgecolor='white', linewidth=0.5)
    bars1 = ax.bar(x, c1, width, bottom=c3 + c2, label='1 session only',
                   color='#e76f51', edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(features_sorted, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Number of selective neurons\n(summed across 13 mice)', fontsize=11)
    ax.set_title('A  Cross-session selectivity retention', fontsize=12,
                 loc='left', fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')

    # Annotate fraction retained in all 3
    for i, (total, n3) in enumerate(zip(c1 + c2 + c3, c3)):
        if total > 0:
            pct = n3 / total * 100
            ax.text(i, total + 1, f'{pct:.0f}%', ha='center', va='bottom',
                    fontsize=7, color='#2a9d8f', fontweight='bold')

    # --- Panel B: Enrichment ratio for 3-session retention ---
    ax = axes[1]
    enrichment_feats = []
    enrichment_ratios = []
    enrichment_pvals = []

    for feat in features_sorted:
        enrich = results[feat].get('enrichment')
        if enrich is not None and hasattr(enrich, 'loc'):
            # DataFrame with index=k, columns: observed, enrichment, -log10p, etc.
            k_max = n_sessions  # 3
            if k_max in enrich.index:
                row = enrich.loc[k_max]
                enrichment_feats.append(feat)
                enrichment_ratios.append(row.get('enrichment', np.nan))
                # Convert -log10p back to p-value for thresholding
                neg_log10p = row.get('-log10p', 0)
                pval = 10 ** (-neg_log10p) if not np.isnan(neg_log10p) else 1.0
                enrichment_pvals.append(pval)

    if enrichment_feats:
        y = np.arange(len(enrichment_feats))
        colors = ['#2a9d8f' if p < 0.05 else '#cccccc'
                  for p in enrichment_pvals]
        ratios_arr = np.array(enrichment_ratios)
        ratios_plot = np.where(np.isnan(ratios_arr), 0, ratios_arr)
        ax.barh(y, ratios_plot, color=colors, edgecolor='black',
                linewidth=0.5, height=0.6)
        ax.axvline(1.0, color='gray', linestyle='--', linewidth=1,
                   label='Chance level')
        ax.set_yticks(y)
        ax.set_yticklabels(enrichment_feats, fontsize=9)
        ax.set_xlabel(f'Enrichment ratio (observed / expected)\n'
                      f'for {n_sessions}-session retention', fontsize=11)
        ax.set_title('B  Retention enrichment', fontsize=12,
                     loc='left', fontweight='bold')
        ax.legend(fontsize=9)

        # Mark significant
        for i, (r, p) in enumerate(zip(ratios_plot, enrichment_pvals)):
            if p < 0.05 and r > 0:
                ax.text(r + 0.05, i, '*', fontsize=14, va='center',
                        fontweight='bold', color='#2a9d8f')
    else:
        ax.text(0.5, 0.5, 'No enrichment data', transform=ax.transAxes,
                ha='center', va='center')

    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTPUT_DIR,
                    f"supp_cross_session_stability.{ext}"), dpi=200)
    plt.close(fig)
    print(f"\nSaved figure to {OUTPUT_DIR}")


def main():
    db = load_fof_database()
    results = compute_retention(db)

    # Print summary
    print("\n" + "=" * 60)
    print("CROSS-SESSION STABILITY SUMMARY (FOF, 3 sessions)")
    print("=" * 60)
    n_sessions = len(db.sessions)
    print(f"{'Feature':<20} {'1-sess':<8} {'2-sess':<8} {'3-sess':<8} "
          f"{'Total':<8} {'%retained':<10}")
    print("-" * 60)

    for feat in sorted(results.keys()):
        total_row = results[feat]['retention_table'].loc['total']
        at1 = int(total_row.get(1, 0))
        at2 = int(total_row.get(2, 0))
        at3 = int(total_row.get(3, 0))
        pct = at3 / at1 * 100 if at1 > 0 else 0
        print(f"{feat:<20} {at1:<8} {at2:<8} {at3:<8} {at1:<8} {pct:<10.1f}")

    print("\nENRICHMENT (index k = at least k sessions)")
    for feat in sorted(results.keys()):
        print("==", feat)
        print(results[feat]['enrichment'])

    plot_results(results, db)


if __name__ == "__main__":
    main()
