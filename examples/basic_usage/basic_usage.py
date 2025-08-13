#!/usr/bin/env python3
"""
Basic INTENSE Usage Example

This example demonstrates the minimal workflow for using DRIADA's INTENSE module:
1. Generate synthetic neural data
2. Analyze neuronal selectivity
3. Extract significant results
4. Visualize findings

This is a self-contained example that runs without external data files.
"""

import sys
import os

# Add the src directory to the path to import driada
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import driada
import matplotlib.pyplot as plt


def main():
    """Run basic INTENSE analysis example."""
    print("=" * 60)
    print("DRIADA INTENSE - Basic Usage Example")
    print("=" * 60)

    # Step 1: Generate synthetic experiment
    print("\n1. Generating synthetic experiment...")
    print("   - 20 neurons")
    print("   - 2 discrete + 2 continuous features")
    print("   - 5 minutes recording")

    exp = driada.generate_synthetic_exp(
        n_dfeats=2,  # discrete features (e.g., trial type)
        n_cfeats=2,  # continuous features (e.g., x, y position)
        nneurons=20,  # number of neurons
        duration=300,  # 5 minutes recording
        seed=42,  # reproducible results
    )

    print(
        f"   ✓ Created experiment with {exp.n_cells} neurons and {exp.n_frames} timepoints"
    )
    print(f"   ✓ Features: {list(exp.dynamic_features.keys())}")

    # Step 2: Analyze neuronal selectivity
    print("\n2. Running INTENSE analysis...")
    print("   - Two-stage statistical testing")
    print("   - Mutual information metric")
    print("   - Multiple comparison correction")

    stats, significance, info, results = driada.compute_cell_feat_significance(
        exp,
        mode="two_stage",
        n_shuffles_stage1=50,  # preliminary screening
        n_shuffles_stage2=1000,  # validation (use 10000+ for publication)
        verbose=False,  # suppress detailed output for cleaner demo
    )

    print("   ✓ Analysis complete")

    # Step 3: Extract significant results
    print("\n3. Extracting significant results...")

    significant_neurons = exp.get_significant_neurons()
    total_pairs = sum(len(features) for features in significant_neurons.values())

    print(f"   ✓ Found {len(significant_neurons)} neurons with significant selectivity")
    print(f"   ✓ Total significant neuron-feature pairs: {total_pairs}")

    # Step 4: Display results
    print("\n4. Results summary:")

    if significant_neurons:
        print("   Significant neuron-feature relationships:")
        for cell_id in list(significant_neurons.keys())[:3]:  # Show first 3
            for feat_name in significant_neurons[cell_id]:
                pair_stats = exp.get_neuron_feature_pair_stats(cell_id, feat_name)

                print(f"   - Neuron {cell_id} ↔ Feature '{feat_name}':")
                print(f"     • Mutual Information: {pair_stats['pre_rval']:.4f}")
                if "pval" in pair_stats:
                    print(f"     • P-value: {pair_stats['pval']:.2e}")
                print(f"     • Optimal delay: {pair_stats.get('shift_used', 0):.2f}s")

        if len(significant_neurons) > 3:
            remaining = len(significant_neurons) - 3
            print(f"   ... and {remaining} more significant neurons")
    else:
        print("   No significant relationships found with current parameters.")
        print("   Try increasing n_shuffles_stage2 or using different synthetic data.")

    # Step 5: Create visualization
    print("\n5. Creating visualization...")

    if significant_neurons:
        # Plot first significant neuron-feature pair
        cell_id = list(significant_neurons.keys())[0]
        feat_name = significant_neurons[cell_id][0]

        fig, ax = plt.subplots(figsize=(10, 6))
        driada.intense.plot_neuron_feature_pair(exp, cell_id, feat_name, ax=ax)
        plt.title(f"Neuron {cell_id} selectivity to {feat_name}")
        plt.tight_layout()

        # Save plot
        output_path = os.path.join(os.path.dirname(__file__), "basic_usage_result.png")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"   ✓ Visualization saved to: {output_path}")

        # Display plot
        plt.show()
    else:
        print("   No visualization created (no significant relationships found)")

    print("\n" + "=" * 60)
    print("BASIC USAGE EXAMPLE COMPLETE")
    print("=" * 60)
    print("\nNext steps:")
    print("- Try full_pipeline.py for comprehensive analysis")
    print("- Try mixed_selectivity.py for advanced disentanglement")
    print("- Modify parameters above to explore different scenarios")


if __name__ == "__main__":
    main()
