#!/usr/bin/env python3
"""
Mixed Selectivity Analysis Example

This example demonstrates advanced disentanglement analysis using DRIADA's INTENSE module:
1. Generate synthetic data with mixed selectivity patterns and MultiTimeSeries features
2. Run standard INTENSE analysis to find significant relationships
3. Apply disentanglement analysis to separate mixed selectivity 
4. Visualize disentanglement results and interpret findings
5. Demonstrate redundancy vs synergy detection

This showcases INTENSE's advanced capabilities for analyzing neurons that respond 
to multiple correlated behavioral variables, including multivariate features.
"""

import sys
import os

# Add the src directory to the path to import driada
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import driada
import matplotlib.pyplot as plt
import numpy as np


def generate_mixed_selectivity_data():
    """Generate synthetic data with known mixed selectivity patterns and MultiTimeSeries."""
    print("\n=== GENERATING MIXED SELECTIVITY DATA ===")
    
    # Create synthetic experiment with mixed selectivity and multifeatures
    exp, selectivity_info = driada.experiment.generate_synthetic_exp_with_mixed_selectivity(
        n_discrete_feats=2,      # discrete features (e.g., task context, trial type)
        n_continuous_feats=4,    # continuous features (e.g., x, y, speed, direction)
        n_neurons=30,            # moderate population size
        n_multifeatures=2,       # create MultiTimeSeries features (e.g., (x,y), (speed,direction))
        selectivity_prob=0.8,    # high probability of selectivity
        multi_select_prob=0.5,   # 50% chance of mixed selectivity
        weights_mode='random',   # random selectivity weights
        duration=600,            # 10 minutes for good statistics
        seed=42,                 # reproducible results
        fps=20,                  # sampling rate
        verbose=False
    )
    
    print(f"Generated experiment:")
    print(f"  • {exp.n_cells} neurons")
    print(f"  • {len(exp.dynamic_features)} total features")
    
    # Separate regular and multifeatures
    regular_features = [f for f in exp.dynamic_features.keys() if isinstance(f, str)]
    multifeatures = [f for f in exp.dynamic_features.keys() if isinstance(f, tuple)]
    
    print(f"  • Regular features: {regular_features}")
    print(f"  • MultiTimeSeries features: {multifeatures}")
    print(f"  • {exp.n_frames} timepoints ({exp.n_frames/exp.fps:.1f}s recording)")
    
    return exp, selectivity_info


def run_intense_analysis(exp):
    """Run INTENSE analysis to identify significant relationships."""
    print("\n=== RUNNING INTENSE ANALYSIS ===")
    
    # Run comprehensive analysis including multifeatures
    # Get features that should skip delay optimization
    # Check both tuple names (multifeatures) and MultiTimeSeries instances
    skip_delays = []
    for feat_name, feat_data in exp.dynamic_features.items():
        if isinstance(feat_name, tuple):  # Tuple name indicates multifeature
            skip_delays.append(feat_name)
        elif hasattr(feat_data, '__class__') and feat_data.__class__.__name__ == 'MultiTimeSeries':
            skip_delays.append(feat_name)
    
    stats, significance, info, results = driada.compute_cell_feat_significance(
        exp,
        mode='two_stage',
        n_shuffles_stage1=50,
        n_shuffles_stage2=1500,  # Higher precision for mixed selectivity
        allow_mixed_dimensions=True,  # Enable MultiTimeSeries analysis
        skip_delays=skip_delays if skip_delays else None,  # Skip delay optimization for MultiTimeSeries
        verbose=False
    )
    
    # Extract significant relationships
    significant_neurons = exp.get_significant_neurons()
    
    print(f"INTENSE analysis results:")
    print(f"  • Significant neurons: {len(significant_neurons)}")
    print(f"  • Total significant pairs: {sum(len(features) for features in significant_neurons.values())}")
    
    # Count multifeature relationships
    from driada.information.info_base import MultiTimeSeries
    multifeature_count = 0
    for cell_id, features in significant_neurons.items():
        for feat in features:
            if feat in exp.dynamic_features and isinstance(exp.dynamic_features[feat], MultiTimeSeries):
                multifeature_count += 1
    
    print(f"  • Significant multifeature relationships: {multifeature_count}")
    
    # Identify neurons with multiple significant features (mixed selectivity candidates)
    mixed_candidates = {cell_id: features for cell_id, features in significant_neurons.items() 
                       if len(features) > 1}
    
    print(f"  • Mixed selectivity candidates: {len(mixed_candidates)} neurons")
    
    return stats, significance, info, results, significant_neurons, mixed_candidates


def analyze_disentanglement(exp, mixed_candidates):
    """Apply disentanglement analysis to mixed selectivity neurons."""
    print("\n=== DISENTANGLEMENT ANALYSIS ===")
    
    if not mixed_candidates:
        print("No mixed selectivity candidates found for disentanglement.")
        return None, None
    
    # Use the built-in multifeature map
    multifeature_map = driada.intense.DEFAULT_MULTIFEATURE_MAP
    
    # Add any additional multifeatures from the experiment
    for feat in exp.dynamic_features.keys():
        if isinstance(feat, tuple) and len(feat) == 2:
            # Create a descriptive name for the multifeature
            if feat not in multifeature_map:
                if 'x' in feat[0].lower() and 'y' in feat[1].lower():
                    multifeature_map[feat] = 'spatial_location'
                elif 'speed' in feat[0].lower() or 'speed' in feat[1].lower():
                    multifeature_map[feat] = 'movement'
                else:
                    multifeature_map[feat] = f'{feat[0]}_{feat[1]}_combined'
    
    print(f"Using {len(multifeature_map)} multifeature combinations:")
    for mf_combo, mf_name in list(multifeature_map.items())[:5]:  # Show first 5
        print(f"  • {mf_combo} → {mf_name}")
    if len(multifeature_map) > 5:
        print(f"  ... and {len(multifeature_map) - 5} more combinations")
    
    # Run disentanglement analysis
    print("\nRunning disentanglement analysis...")
    
    disentanglement_results = driada.intense.disentangle_all_selectivities(
        exp,
        multifeature_map=multifeature_map,
        cell_bunch=list(mixed_candidates.keys()),  # Focus on mixed selectivity neurons
        verbose=False
    )
    
    print(f"Disentanglement completed for {len(mixed_candidates)} neurons")
    
    return disentanglement_results, multifeature_map


def interpret_disentanglement_results(exp, disentanglement_results, mixed_candidates):
    """Interpret and summarize disentanglement analysis results."""
    print("\n=== INTERPRETING DISENTANGLEMENT RESULTS ===")
    
    if disentanglement_results is None:
        return [], [], []
    
    redundancy_cases = []
    synergy_cases = []
    independence_cases = []
    
    for cell_id in mixed_candidates.keys():
        cell_results = disentanglement_results.get(cell_id, {})
        
        for feature_pair, result in cell_results.items():
            if not isinstance(feature_pair, tuple) or len(feature_pair) != 2:
                continue
                
            disentanglement_value = result.get('disentanglement', None)
            
            if disentanglement_value is not None:
                if disentanglement_value < 0.3:  # Redundancy
                    redundancy_cases.append((cell_id, feature_pair, disentanglement_value))
                elif disentanglement_value > 0.7:  # Synergy
                    synergy_cases.append((cell_id, feature_pair, disentanglement_value))
                else:  # Independence
                    independence_cases.append((cell_id, feature_pair, disentanglement_value))
    
    # Summary statistics
    print(f"Disentanglement analysis summary:")
    print(f"  • Redundancy cases (< 0.3): {len(redundancy_cases)}")
    print(f"  • Independence cases (0.3-0.7): {len(independence_cases)}")
    print(f"  • Synergy cases (> 0.7): {len(synergy_cases)}")
    
    # Detailed examples
    if redundancy_cases:
        print(f"\nRedundancy examples (overlapping information):")
        for cell_id, feature_pair, value in redundancy_cases[:3]:
            feat1_name = feature_pair[0]
            feat2_name = feature_pair[1]
            print(f"  • Neuron {cell_id}: {feat1_name} ↔ {feat2_name} (redundancy: {value:.3f})")
            
            # Get mutual information values for context
            try:
                stats1 = exp.get_neuron_feature_pair_stats(cell_id, feature_pair[0])
                stats2 = exp.get_neuron_feature_pair_stats(cell_id, feature_pair[1])
                print(f"    MI({feat1_name}): {stats1['pre_rval']:.4f}, MI({feat2_name}): {stats2['pre_rval']:.4f}")
            except:
                pass
    
    if synergy_cases:
        print(f"\nSynergy examples (combined information > individual):")
        for cell_id, feature_pair, value in synergy_cases[:3]:
            feat1_name = feature_pair[0]
            feat2_name = feature_pair[1]
            print(f"  • Neuron {cell_id}: {feat1_name} + {feat2_name} (synergy: {value:.3f})")
            
            try:
                stats1 = exp.get_neuron_feature_pair_stats(cell_id, feature_pair[0])
                stats2 = exp.get_neuron_feature_pair_stats(cell_id, feature_pair[1])
                print(f"    MI({feat1_name}): {stats1['pre_rval']:.4f}, MI({feat2_name}): {stats2['pre_rval']:.4f}")
            except:
                pass
    
    if independence_cases:
        print(f"\nIndependence examples (separate information sources):")
        for cell_id, feature_pair, value in independence_cases[:3]:
            feat1_name = feature_pair[0]
            feat2_name = feature_pair[1]
            print(f"  • Neuron {cell_id}: {feat1_name} | {feat2_name} (independence: {value:.3f})")
    
    return redundancy_cases, synergy_cases, independence_cases


def create_disentanglement_visualization(exp, disentanglement_results, mixed_candidates, output_dir):
    """Create comprehensive visualization of disentanglement results."""
    print("\n=== CREATING DISENTANGLEMENT VISUALIZATIONS ===")
    
    if not disentanglement_results or not mixed_candidates:
        print("No disentanglement results to visualize")
        return
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Disentanglement heatmap
    ax1 = plt.subplot(2, 3, 1)
    try:
        driada.intense.plot_disentanglement_heatmap(
            disentanglement_results, 
            cell_bunch=list(mixed_candidates.keys())[:10],  # First 10 mixed neurons
            ax=ax1
        )
        ax1.set_title("Disentanglement Heatmap\n(Red=Redundancy, Blue=Synergy)")
    except Exception as e:
        ax1.text(0.5, 0.5, f'Heatmap error:\n{str(e)[:50]}...', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title("Disentanglement Heatmap (Error)")
    
    # Plot 2: Disentanglement value distribution
    ax2 = plt.subplot(2, 3, 2)
    all_disentanglement_values = []
    
    for cell_results in disentanglement_results.values():
        for feature_pair, result in cell_results.items():
            if isinstance(feature_pair, tuple) and len(feature_pair) == 2:
                disentanglement_value = result.get('disentanglement', None)
                if disentanglement_value is not None:
                    all_disentanglement_values.append(disentanglement_value)
    
    if all_disentanglement_values:
        ax2.hist(all_disentanglement_values, bins=15, alpha=0.7, edgecolor='black')
        ax2.axvline(0.3, color='red', linestyle='--', label='Redundancy threshold')
        ax2.axvline(0.7, color='blue', linestyle='--', label='Synergy threshold')
        ax2.set_xlabel('Disentanglement Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Disentanglement Distribution')
        ax2.legend()
    
    # Plot 3: Example neuron-feature relationship
    ax3 = plt.subplot(2, 3, 3)
    if mixed_candidates:
        example_cell = list(mixed_candidates.keys())[0]
        example_features = mixed_candidates[example_cell]
        
        # Try to find a multifeature example
        from driada.information.info_base import MultiTimeSeries
        multifeature_example = None
        for feat in example_features:
            if feat in exp.dynamic_features and isinstance(exp.dynamic_features[feat], MultiTimeSeries):
                multifeature_example = feat
                break
        
        example_feature = multifeature_example if multifeature_example else example_features[0]
        
        try:
            driada.intense.plot_neuron_feature_pair(exp, example_cell, example_feature, ax=ax3)
            from driada.information.info_base import MultiTimeSeries
            is_multi = example_feature in exp.dynamic_features and isinstance(exp.dynamic_features[example_feature], MultiTimeSeries)
            feat_name = f"MultiFeature_{example_feature}" if is_multi else example_feature
            ax3.set_title(f"Example: Neuron {example_cell} ↔ {feat_name}")
        except Exception as e:
            ax3.text(0.5, 0.5, f'Plot error:\n{str(e)[:50]}...', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title("Example Relationship (Error)")
    
    # Plot 4: Mixed selectivity statistics
    ax4 = plt.subplot(2, 3, 4)
    
    # Count features per neuron
    features_per_neuron = [len(features) for features in mixed_candidates.values()]
    
    if features_per_neuron:
        unique_counts, frequencies = np.unique(features_per_neuron, return_counts=True)
        ax4.bar(unique_counts, frequencies, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Number of Significant Features')
        ax4.set_ylabel('Number of Neurons')
        ax4.set_title('Mixed Selectivity Distribution')
        ax4.set_xticks(unique_counts)
    
    # Plot 5: MI comparison for mixed neurons
    ax5 = plt.subplot(2, 3, 5)
    
    all_mi_values = []
    multifeature_mi_values = []
    regular_mi_values = []
    
    for cell_id, features in mixed_candidates.items():
        for feature in features:
            try:
                stats = exp.get_neuron_feature_pair_stats(cell_id, feature)
                mi_value = stats['pre_rval']
                all_mi_values.append(mi_value)
                
                if feature in exp.dynamic_features and isinstance(exp.dynamic_features[feature], MultiTimeSeries):
                    multifeature_mi_values.append(mi_value)
                else:
                    regular_mi_values.append(mi_value)
            except:
                pass
    
    if all_mi_values:
        bins = np.linspace(min(all_mi_values), max(all_mi_values), 12)
        ax5.hist(regular_mi_values, bins=bins, alpha=0.5, label='Regular features', edgecolor='black')
        if multifeature_mi_values:
            ax5.hist(multifeature_mi_values, bins=bins, alpha=0.5, label='MultiTimeSeries', edgecolor='black')
        ax5.set_xlabel('Mutual Information')
        ax5.set_ylabel('Frequency')
        ax5.set_title('MI Values for Mixed Neurons')
        ax5.legend()
    
    # Plot 6: Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate summary statistics
    total_analyzed = len(mixed_candidates)
    total_pairs = sum(len(features) for features in mixed_candidates.values())
    avg_features = np.mean([len(features) for features in mixed_candidates.values()]) if mixed_candidates else 0
    
    # Count multifeature relationships
    multifeature_count = sum(1 for features in mixed_candidates.values() 
                           for feat in features if isinstance(feat, tuple))
    
    redundancy_count = sum(1 for cell_results in disentanglement_results.values() 
                          for result in cell_results.values() 
                          if result.get('disentanglement', 1) < 0.3)
    
    synergy_count = sum(1 for cell_results in disentanglement_results.values() 
                       for result in cell_results.values() 
                       if result.get('disentanglement', 0) > 0.7)
    
    summary_text = f"""Mixed Selectivity Summary:

Neurons analyzed: {total_analyzed}
Total feature pairs: {total_pairs}
MultiTimeSeries pairs: {multifeature_count}
Avg features/neuron: {avg_features:.1f}

Disentanglement Results:
• Redundancy cases: {redundancy_count}
• Synergy cases: {synergy_count}
• Independence cases: {len(all_disentanglement_values) - redundancy_count - synergy_count}

MI Statistics:
• Mean MI: {np.mean(all_mi_values) if all_mi_values else 0:.4f}
• Mean MI (regular): {np.mean(regular_mi_values) if regular_mi_values else 0:.4f}
• Mean MI (multi): {np.mean(multifeature_mi_values) if multifeature_mi_values else 0:.4f}
"""
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save visualization
    output_path = os.path.join(output_dir, 'mixed_selectivity_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Disentanglement visualization saved to: {output_path}")
    
    plt.show()


def main():
    """Run mixed selectivity analysis example."""
    print("=" * 80)
    print("DRIADA INTENSE - Mixed Selectivity Analysis Example")
    print("=" * 80)
    print("\nThis example demonstrates advanced disentanglement analysis for neurons")
    print("that respond to multiple correlated behavioral variables, including")
    print("MultiTimeSeries features that combine multiple signals.")
    
    output_dir = os.path.dirname(__file__)
    
    # Step 1: Generate mixed selectivity data with MultiTimeSeries
    exp, selectivity_info = generate_mixed_selectivity_data()
    
    # Step 2: Run INTENSE analysis
    stats, significance, info, results, significant_neurons, mixed_candidates = run_intense_analysis(exp)
    
    # Step 3: Apply disentanglement analysis
    disentanglement_results, multifeature_map = analyze_disentanglement(exp, mixed_candidates)
    
    # Step 4: Interpret results
    redundancy_cases, synergy_cases, independence_cases = interpret_disentanglement_results(
        exp, disentanglement_results, mixed_candidates
    )
    
    # Step 5: Create visualizations
    create_disentanglement_visualization(exp, disentanglement_results, mixed_candidates, output_dir)
    
    # Final summary and interpretation guide
    print("\n" + "=" * 80)
    print("MIXED SELECTIVITY ANALYSIS COMPLETE")
    print("=" * 80)
    
    print(f"\nResults interpretation guide:")
    print(f"• Redundancy (< 0.3): Features provide overlapping information about neuron")
    print(f"• Independence (0.3-0.7): Features provide separate information")
    print(f"• Synergy (> 0.7): Combined features provide more info than sum of parts")
    
    print(f"\nMultiTimeSeries interpretation:")
    print(f"• Spatial MultiTimeSeries (x,y): Neurons encoding 2D position")
    print(f"• Movement MultiTimeSeries (speed,direction): Neurons encoding motion")
    print(f"• These combined features often show higher MI than individual components")
    
    print(f"\nBiological interpretations:")
    print(f"• Redundancy: Multiple correlated behavioral variables (e.g., speed & distance)")
    print(f"• Independence: Neuron encodes multiple independent variables")
    print(f"• Synergy: Neuron requires combination of variables (e.g., place cells need X+Y)")
    
    print(f"\nAnalysis summary:")
    print(f"• Dataset: {exp.n_cells} neurons, {len(exp.dynamic_features)} features")
    print(f"• Mixed selectivity neurons: {len(mixed_candidates)}")
    
    # Count multifeatures in results
    from driada.information.info_base import MultiTimeSeries
    multifeature_neurons = sum(1 for features in significant_neurons.values() 
                             for feat in features 
                             if feat in exp.dynamic_features and isinstance(exp.dynamic_features[feat], MultiTimeSeries))
    print(f"• Neurons selective to MultiTimeSeries: {multifeature_neurons}")
    
    print(f"• Redundancy cases: {len(redundancy_cases)}")
    print(f"• Synergy cases: {len(synergy_cases)}")
    
    print(f"\nOutput files:")
    print(f"• mixed_selectivity_analysis.png - Comprehensive disentanglement visualization")
    
    print(f"\nNext steps:")
    print(f"• Modify synthetic data parameters to explore different selectivity patterns")
    print(f"• Apply to real neural data with known behavioral correlations")
    print(f"• Use results to understand neural coding principles")


if __name__ == "__main__":
    main()