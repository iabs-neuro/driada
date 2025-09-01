# INTENSE Module

**I**nformation-**T**heoretic **E**valuation of **N**euronal **SE**lectivity

## Overview

INTENSE is DRIADA's core module for analyzing how individual neurons encode behavioral and environmental variables using information theory. It quantifies neuronal selectivity through mutual information (MI) analysis with rigorous statistical testing.

## Key Features

- **Mutual Information Analysis**: Quantify how much information neurons carry about features
- **Statistical Significance Testing**: Two-stage approach with multiple comparison correction
- **Delay Optimization**: Find optimal temporal delays between neural activity and features
- **Mixed Selectivity Analysis**: Disentangle neurons responding to multiple features
- **Cell-Cell Connectivity**: Analyze functional relationships between neurons

## Main Functions

### Pipeline Functions
- `compute_cell_feat_significance()` - Analyze neuron-feature selectivity
- `compute_feat_feat_significance()` - Analyze feature-feature relationships
- `compute_cell_cell_significance()` - Analyze neuron-neuron connectivity
- `compute_embedding_selectivity()` - Analyze selectivity to embedding components

### Visualization
- `plot_neuron_feature_pair()` - Visualize single neuron-feature relationship
- `plot_selectivity_heatmap()` - Overview of all selectivities
- `plot_disentanglement_heatmap()` - Mixed selectivity analysis

## Example Usage

```python
from driada.intense import compute_cell_feat_significance

# Analyze neuronal selectivity
stats, significance, info, results = compute_cell_feat_significance(
    experiment,
    n_shuffles_stage1=100,   # Quick screening
    n_shuffles_stage2=1000,  # Detailed validation
    ds=5,                    # Downsample for speed
    verbose=True
)

# Get neurons selective to any feature
significant_neurons = experiment.get_significant_neurons()
```

## Technical Details

INTENSE uses a two-stage approach:
1. **Stage 1**: Quick screening with fewer shuffles to identify candidates
2. **Stage 2**: Rigorous validation with many shuffles for selected pairs

Statistical significance is determined by comparing actual MI values against null distributions generated through temporal shuffling.

## Supported Metrics

While mutual information is the primary metric, INTENSE also supports:
- **Pearson correlation** - Linear relationships
- **Spearman correlation** - Monotonic relationships

These can be selected using the `metric` parameter in `compute_cell_feat_significance()` and other analysis functions. Each metric uses the same statistical testing framework for significance assessment.