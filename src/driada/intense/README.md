# INTENSE Module

**I**nformation-**T**heoretic **E**valuation of **N**euronal **SE**lectivity

## Overview

INTENSE is DRIADA's core module for analyzing how individual neurons encode behavioral and environmental variables using information theory. It quantifies neuronal selectivity through mutual information (MI) analysis with rigorous statistical testing.

## Key features

- **Mutual Information Analysis**: Quantify how much information neurons carry about features
- **Statistical Significance Testing**: Two-stage approach with multiple comparison correction
- **Delay Optimization**: Find optimal temporal delays between neural activity and features
- **Mixed Selectivity Analysis**: Disentangle neurons responding to multiple features
- **Cell-Cell Connectivity**: Analyze functional relationships between neurons

## Main functions

### Pipeline functions
- `compute_cell_feat_significance()` - Analyze neuron-feature selectivity
- `compute_feat_feat_significance()` - Analyze feature-feature relationships
- `compute_cell_cell_significance()` - Analyze neuron-neuron connectivity
- `compute_embedding_selectivity()` - Analyze selectivity to embedding components

### Disentanglement
- `disentangle_pair()` — decompose mixed selectivity for a neuron-feature-feature triplet
- `disentangle_all_selectivities()` — run disentanglement across all significant pairs
- `create_multifeature_map()` — build a multi-feature selectivity summary

### Visualization
- `plot_neuron_feature_pair()` — single neuron-feature relationship
- `plot_neuron_feature_density()` — density plot of neural activity vs feature
- `plot_selectivity_heatmap()` — overview of all selectivities
- `plot_disentanglement_heatmap()` — mixed selectivity analysis
- `plot_disentanglement_summary()` — summary of disentanglement results
- `plot_pc_activity()` — place cell spatial activity maps
- `plot_shadowed_groups()` — highlight neuron groups on a shared axis

## Example usage

```python
from driada.intense import compute_cell_feat_significance

# Analyze neuronal selectivity
stats, significance, info, results, disent = compute_cell_feat_significance(
    experiment,
    n_shuffles_stage1=100,   # Quick screening
    n_shuffles_stage2=1000,  # Detailed validation
    ds=5,                    # Downsample for speed
    with_disentanglement=True,
    remove_anti_selective=True,
    verbose=True
)

# Get neurons selective to any feature
significant_neurons = experiment.get_significant_neurons()
```

## Technical details

### Two-stage significance testing

INTENSE uses a two-stage approach:
1. **Stage 1**: Quick screening with fewer shuffles to identify candidates
2. **Stage 2**: Rigorous validation with many shuffles for selected pairs

Statistical significance is determined by comparing actual MI values against null distributions generated through temporal shuffling.

### Anti-selectivity filtering

After significance testing, INTENSE computes a **signal ratio** for each significant neuron-feature pair to detect anti-selective neurons (neurons suppressed when a feature is active). Anti-selective pairs are removed before disentanglement.

Signal ratio is computed as `mean(Ca|feature active) / mean(Ca|feature inactive)`:
- **Binary discrete features** (locomotion, freezing, rest, etc.): active = feature value 1, inactive = feature value 0
- **Linear continuous features** (speed): active = above median, inactive = at or below median
- **Circular and multivariate features**: not applicable (signal_ratio = None, these pass through)

Pairs with signal_ratio ≤ 1.0 are marked as non-significant (stage2 = False). Controlled by the `remove_anti_selective` parameter (default: True).

### Disentanglement

When a neuron is significant for multiple features, disentanglement determines whether the neuron truly encodes each feature independently or is redundantly driven by correlated features. It uses conditional mutual information (CMI) to test whether a neuron's selectivity to feature A persists after conditioning on feature B.

The disentanglement pipeline supports **pre-filters** (priority rules that resolve known feature hierarchies before CMI computation) and **post-filters** (tie-breaking rules applied after CMI results). See `tools/selectivity_dynamics/filters.py` for filter construction utilities.

## Supported metrics

While mutual information is the primary metric, INTENSE also supports:
- **Pearson correlation** - Linear relationships
- **Spearman correlation** - Monotonic relationships

These can be selected using the `metric` parameter in `compute_cell_feat_significance()` and other analysis functions. Each metric uses the same statistical testing framework for significance assessment.