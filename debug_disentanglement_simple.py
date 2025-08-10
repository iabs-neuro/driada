#!/usr/bin/env python
"""Debug why disentanglement doesn't find mixed selectivity pairs."""

import numpy as np
from driada.experiment.synthetic import generate_synthetic_exp_with_mixed_selectivity
from driada.intense.pipelines import compute_cell_feat_significance

# Generate simple experiment with guaranteed mixed selectivity
exp, selectivity_info = generate_synthetic_exp_with_mixed_selectivity(
    n_discrete_feats=2,        # Just 2 features
    n_continuous_feats=0,      # No continuous
    n_neurons=4,               # Just 4 neurons
    duration=300,              # 5 minutes
    fps=20,
    selectivity_prob=1.0,      # All selective
    multi_select_prob=1.0,     # All mixed
    weights_mode='dominant',   # Asymmetric weights
    skip_prob=0.0,
    rate_0=0.5,
    rate_1=10.0,
    ampl_range=(0.5, 2.0),
    noise_std=0.005,
    seed=42,
    verbose=False
)

print("Selectivity matrix (features x neurons):")
print(selectivity_info['matrix'])
print("\nFeatures:", list(exp.dynamic_features.keys()))

# All neurons should have mixed selectivity
mixed_neurons = np.where(np.sum(selectivity_info['matrix'] > 0, axis=0) >= 2)[0]
print(f"\nMixed selectivity neurons: {mixed_neurons}")

# Run significance testing with disentanglement
print("\nRunning significance testing with disentanglement...")
stats, significance, info, results, disent_results = compute_cell_feat_significance(
    exp,
    cell_bunch=None,  # Use all neurons
    feat_bunch=None,  # Use all features
    mode='stage1',    # Simple mode
    n_shuffles_stage1=10,
    metric='mi',
    pval_thr=0.1,     # Very lenient
    multicomp_correction=None,
    with_disentanglement=True,
    save_computed_stats=True,  # Ensure stats are saved
    verbose=True,
    seed=42
)

print("\n" + "="*60)
print("DISENTANGLEMENT RESULTS:")
print("="*60)

if 'summary' in disent_results:
    summary = disent_results['summary']
    if summary.get('overall_stats'):
        print(f"Total pairs analyzed: {summary['overall_stats']['total_neuron_pairs']}")
    else:
        print("No overall stats found")
        
print(f"\nCount matrix:\n{disent_results['count_matrix']}")
print(f"\nDisentanglement matrix:\n{disent_results['disent_matrix']}")

# Check what get_significant_neurons returns
print("\n" + "="*60)
print("CHECKING get_significant_neurons:")
print("="*60)

try:
    sig_neurons = exp.get_significant_neurons(min_nspec=2)
    print(f"Neurons with >=2 significant features: {sig_neurons}")
except Exception as e:
    print(f"Error calling get_significant_neurons: {type(e).__name__}: {e}")
    
# Check significance tables structure
print("\nChecking significance_tables structure:")
if hasattr(exp, 'significance_tables'):
    print(f"Modes: {list(exp.significance_tables.keys())}")
    if 'calcium' in exp.significance_tables:
        print(f"Features in calcium: {list(exp.significance_tables['calcium'].keys())}")
        # Check one feature
        if exp.significance_tables['calcium']:
            feat = list(exp.significance_tables['calcium'].keys())[0]
            print(f"Neurons for feature '{feat}': {list(exp.significance_tables['calcium'][feat].keys())}")
            # Check one neuron
            if exp.significance_tables['calcium'][feat]:
                neuron = list(exp.significance_tables['calcium'][feat].keys())[0]
                print(f"Significance for neuron {neuron}, feature '{feat}':")
                print(exp.significance_tables['calcium'][feat][neuron])
else:
    print("No significance_tables attribute found!")