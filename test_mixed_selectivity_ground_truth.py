"""Test to verify that generate_mixed_population_exp creates proper mixed selectivity in ground truth."""
import numpy as np
from driada.experiment import generate_mixed_population_exp

# Generate a mixed population like in extract_task_variables.py
exp, info = generate_mixed_population_exp(
    n_neurons=200,
    manifold_type='2d_spatial',
    manifold_fraction=0.6,  # 120 manifold cells, 80 feature cells
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

# Analyze ground truth
print("=== GROUND TRUTH ANALYSIS ===")
print(f"Total neurons: {exp.n_cells}")
print(f"Total features: {len(exp.dynamic_features)}")
print(f"Feature names: {list(exp.dynamic_features.keys())}")

# Check info dictionary
print(f"\nPopulation composition from info:")
print(f"  Info keys: {list(info.keys())}")
if 'n_manifold_cells' in info:
    print(f"  Manifold cells: {info['n_manifold_cells']}")
if 'n_feature_cells' in info:
    print(f"  Feature-selective cells: {info['n_feature_cells']}")
    
# Check feature_selectivity info
if 'feature_selectivity' in info:
    fs = info['feature_selectivity']
    if isinstance(fs, np.ndarray):
        print(f"  Feature selectivity shape: {fs.shape}")
        gt = fs  # This might be the ground truth
    else:
        print(f"  Feature selectivity type: {type(fs)}")
# Calculate from fraction
n_manifold = int(200 * 0.6)  # 120
n_feature = 200 - n_manifold  # 80
print(f"  Calculated manifold cells: {n_manifold}")
print(f"  Calculated feature cells: {n_feature}")

# Get ground truth from experiment - try different locations
gt = None
if hasattr(exp, 'ground_truth'):
    gt = exp.ground_truth
elif hasattr(exp, 'static'):
    gt = exp.static.get('ground_truth', None)

# If not found in experiment, use from info dictionary
if gt is None and 'feature_selectivity' in info:
    # feature_selectivity is the ground truth matrix for feature-selective neurons
    gt = info['feature_selectivity']
if gt is not None:
    print(f"\nGround truth matrix shape: {gt.shape}")
    
    # feature_selectivity should be 3x80 for the feature neurons
    # Rows are: d_feat_0, c_feat_0, c_feat_1
    feature_gt = gt
    
    print(f"\nAnalyzing {feature_gt.shape[1]} feature-selective neurons:")
    
    # Count neurons selective to each feature
    feat_names = ['d_feat_0', 'c_feat_0', 'c_feat_1']
    
    selective_counts = []
    for i, feat_name in enumerate(feat_names):
        selective = np.sum(feature_gt[i, :] > 0)
        selective_counts.append(selective)
        print(f"  Neurons selective to {feat_name}: {selective}")
    
    # Count mixed selectivity
    n_features_per_neuron = np.sum(feature_gt > 0, axis=0)
    mixed_selective = np.sum(n_features_per_neuron > 1)
    single_selective = np.sum(n_features_per_neuron == 1)
    non_selective = np.sum(n_features_per_neuron == 0)
    
    print(f"\nSelectivity distribution among feature-selective neurons:")
    print(f"  Non-selective: {non_selective} ({100*non_selective/feature_gt.shape[1]:.1f}%)")
    print(f"  Single-selective: {single_selective} ({100*single_selective/feature_gt.shape[1]:.1f}%)")
    print(f"  Mixed-selective: {mixed_selective} ({100*mixed_selective/feature_gt.shape[1]:.1f}%)")
    
    # Show some examples of mixed selectivity
    if mixed_selective > 0:
        print(f"\nExamples of mixed selectivity (first 5):")
        mixed_neurons = np.where(n_features_per_neuron > 1)[0]
        for i, neuron_idx in enumerate(mixed_neurons[:5]):
            selective_features = []
            for j, feat_name in enumerate(feat_names):
                if feature_gt[j, neuron_idx] > 0:
                    weight = feature_gt[j, neuron_idx]
                    selective_features.append(f"{feat_name}({weight:.2f})")
            print(f"  Neuron {n_manifold + neuron_idx}: {', '.join(selective_features)}")
else:
    print("\nNo ground truth found in experiment!")

print("\n=== EXPECTED VS ACTUAL ===")
print(f"Expected selective neurons: ~{int(80 * 0.8)} (80% of 80)")
print(f"Expected mixed selective: ~{int(64 * 0.6)} (60% of selective)")
print(f"Actual mixed selective in ground truth: {mixed_selective if gt is not None else 'N/A'}")