"""Quick check of mixed selectivity detection."""
import numpy as np
from driada.experiment import generate_mixed_population_exp
from driada.intense import compute_cell_feat_significance

# Generate the same mixed population as in extract_task_variables.py
print("Generating mixed population...")
exp, info = generate_mixed_population_exp(
    n_neurons=200,
    manifold_type='2d_spatial',
    manifold_fraction=0.6,
    n_discrete_features=1,
    n_continuous_features=2,
    duration=300,
    fps=20,
    correlation_mode='spatial_correlated',
    seed=42,
    manifold_params={
        'grid_arrangement': True,
        'field_sigma': 0.15,
        'noise_std': 0.1,
        'baseline_rate': 0.1,
        'peak_rate': 2.0,
        'decay_time': 2.0,
        'calcium_noise_std': 0.1
    },
    feature_params={
        'selectivity_prob': 0.8,
        'multi_select_prob': 0.6,
        'rate_0': 0.5,
        'rate_1': 3.0,
        'noise_std': 0.1,
        'hurst': 0.3,
        'skip_prob': 0.0,
        'ampl_range': (1.5, 3.5),
        'decay_time': 2.0
    }
)

print("Running INTENSE with original parameters...")
# Run with original parameters from extract_task_variables.py
stats, significance, info_result, intense_results = compute_cell_feat_significance(
    exp,
    feat_bunch=['position_2d', 'c_feat_0', 'c_feat_1', 'd_feat_0'],
    mode='two_stage',
    n_shuffles_stage1=50,
    n_shuffles_stage2=500,
    pval_thr=0.05,
    find_optimal_delays=False,
    allow_mixed_dimensions=True,
    with_disentanglement=False
)

# Check results
significant_neurons = exp.get_significant_neurons()
mixed_selectivity_neurons = exp.get_significant_neurons(min_nspec=2)

print(f"\nRESULTS:")
print(f"Total selective neurons: {len(significant_neurons)}")
print(f"Mixed selectivity neurons: {len(mixed_selectivity_neurons)}")

# Ground truth check
if 'feature_selectivity' in info:
    gt = info['feature_selectivity']
    n_features_per_neuron = np.sum(gt > 0, axis=0)
    gt_mixed = np.sum(n_features_per_neuron > 1)
    print(f"Ground truth mixed selectivity: {gt_mixed}/80 feature neurons")