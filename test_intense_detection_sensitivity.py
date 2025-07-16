"""Test INTENSE detection sensitivity with relaxed parameters."""
import numpy as np
from driada.experiment import generate_mixed_population_exp
from driada.intense import compute_cell_feat_significance

# Generate the same mixed population as in extract_task_variables.py
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

# Run INTENSE analysis with relaxed parameters
print("=== INTENSE DETECTION WITH RELAXED PARAMETERS ===")
print("- Disabling multiple comparisons")
print("- Setting p_value threshold to 0.001")
print("- Using stage1 only mode for faster testing")

# Test with relaxed parameters
task_features = ['position_2d', 'c_feat_0', 'c_feat_1', 'd_feat_0']
available_features = [f for f in task_features if f in exp.dynamic_features]

print(f"Analyzing {exp.n_cells} neurons × {len(available_features)} features")

# Run with relaxed parameters
stats, significance, info, intense_results = compute_cell_feat_significance(
    exp,
    feat_bunch=available_features,
    mode='stage1',  # Use stage1 only for faster testing
    n_shuffles_stage1=100,
    pval_thr=0.001,  # Relaxed threshold
    verbose=True,
    find_optimal_delays=False,
    allow_mixed_dimensions=True,
    with_disentanglement=False,
    multicomp_correction=None  # Disable multiple comparisons
)

# Analyze results
significant_neurons = exp.get_significant_neurons()
mixed_selectivity_neurons = exp.get_significant_neurons(min_nspec=2)

print(f"\nResults with relaxed parameters:")
print(f"  - Total selective neurons: {len(significant_neurons)}/{exp.n_cells}")
print(f"  - Mixed selectivity neurons: {len(mixed_selectivity_neurons)}")

# Count selectivity by feature
feature_counts = {}
for neuron_id, features in significant_neurons.items():
    for feat in features:
        feature_counts[feat] = feature_counts.get(feat, 0) + 1

print(f"\nSelectivity by feature:")
for feat, count in sorted(feature_counts.items()):
    print(f"  - {feat}: {count} neurons")

# Compare with ground truth
print(f"\n=== COMPARISON WITH GROUND TRUTH ===")
print(f"Ground truth mixed selectivity: 37/80 feature neurons (46.2%)")
print(f"INTENSE detected mixed selectivity: {len(mixed_selectivity_neurons)} neurons")

# Show improvement
print(f"\nImprovement analysis:")
print(f"  - Previous detection (default params): 6 mixed neurons")
print(f"  - Current detection (relaxed params): {len(mixed_selectivity_neurons)} mixed neurons")
improvement = len(mixed_selectivity_neurons) - 6
print(f"  - Improvement: {improvement} additional mixed neurons detected")

# Show some examples of mixed selectivity
if mixed_selectivity_neurons:
    print(f"\nExamples of detected mixed selectivity (first 5):")
    for i, (neuron_id, features) in enumerate(list(mixed_selectivity_neurons.items())[:5]):
        print(f"  Neuron {neuron_id}: {', '.join(features)}")