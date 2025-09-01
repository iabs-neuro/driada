# Representational Similarity Analysis (RSA) Module

## Overview

The RSA module implements Representational Similarity Analysis for comparing neural population representations. RSA characterizes the geometry of neural representations by computing distances between population activity patterns.

## Key Concepts

### RDM (Representational Dissimilarity Matrix)
A symmetric matrix where each element represents the dissimilarity between neural population patterns for two conditions/stimuli/time points.

### Distance Metrics
- Correlation distance (default)
- Euclidean distance  
- Cosine distance
- Mahalanobis distance

## Main Functions

### Computing RDMs
- `compute_rdm()` - Basic RDM from data matrix
- `compute_rdm_from_trials()` - Average within conditions first
- `compute_experiment_rdm()` - Direct from Experiment objects
- `compute_rdm_unified()` - Flexible interface with bootstrapping

### Comparing RDMs
- `compare_rdms()` - Correlate two RDMs
- `bootstrap_rdm_comparison()` - Statistical significance
- `rsa_between_experiments()` - Cross-experiment comparison
- `rsa_compare()` - New simplified comparison API

### Visualization
- `plot_rdm()` - Heatmap visualization
- `plot_rdm_comparison()` - Side-by-side comparison

## Example Usage

```python
from driada.rsa import compute_experiment_rdm, compare_rdms, plot_rdm

# Compute RDM from experiment based on conditions
rdm = compute_experiment_rdm(
    exp,
    condition_feature='trial_type',  # Group by trial type
    method='correlation',
    average_conditions=True
)

# Visualize
plot_rdm(rdm, labels=exp.get_unique_conditions('trial_type'))

# Compare with another experiment
rdm2 = compute_experiment_rdm(exp2, condition_feature='trial_type')
similarity = compare_rdms(rdm, rdm2, method='spearman')

# Bootstrap comparison for significance
from driada.rsa import bootstrap_rdm_comparison
p_value, null_dist = bootstrap_rdm_comparison(
    rdm, rdm2,
    n_bootstrap=1000,
    method='spearman'
)
```

## Advanced Usage

### Within-Condition Bootstrap
```python
from driada.rsa import compute_rdm_unified

# Bootstrap within each condition for error bars
rdm_mean, rdm_ci = compute_rdm_unified(
    data_dict,  # {condition: trials}
    bootstrap_within_condition=True,
    n_bootstrap=100
)
```

### Cross-Validation
Split data to avoid overfitting when comparing RDMs from the same dataset.

## Applications

- Compare representations across brain areas
- Track representational changes over time
- Compare neural representations to model predictions
- Identify representational geometry of task variables