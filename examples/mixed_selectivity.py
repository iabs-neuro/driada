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
        n_neurons=50,            # larger population for more mixed selectivity
        n_multifeatures=2,       # create MultiTimeSeries features (e.g., (x,y), (speed,direction))
        selectivity_prob=0.9,    # very high probability of selectivity
        multi_select_prob=0.8,   # 80% chance of mixed selectivity
        weights_mode='random',   # random selectivity weights
        duration=600,            # 10 minutes for good statistics
        seed=123,                # different seed for better results
        fps=20,                  # sampling rate
        verbose=False
    )
    
    print(f"Generated experiment: {exp.n_cells} neurons, {len(exp.dynamic_features)} features, {exp.n_frames/exp.fps:.1f}s recording")
    
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
    
    # Count multifeature relationships
    from driada.information.info_base import MultiTimeSeries
    multifeature_count = 0
    for cell_id, features in significant_neurons.items():
        for feat in features:
            if feat in exp.dynamic_features and isinstance(exp.dynamic_features[feat], MultiTimeSeries):
                multifeature_count += 1
    
    # Identify neurons with multiple significant features (mixed selectivity candidates)
    mixed_candidates = {cell_id: features for cell_id, features in significant_neurons.items() 
                       if len(features) > 1}
    
    print(f"INTENSE found {len(significant_neurons)} significant neurons, {len(mixed_candidates)} with mixed selectivity")
    
    return stats, significance, info, results, significant_neurons, mixed_candidates


def analyze_disentanglement(exp, mixed_candidates):
    """Apply disentanglement analysis to mixed selectivity neurons."""
    print("\n=== DISENTANGLEMENT ANALYSIS ===")
    
    if not mixed_candidates:
        print("No mixed selectivity candidates found for disentanglement.")
        return None, None, None
    
    # Get all feature names from the experiment (string features only, excluding MultiTimeSeries)
    from driada.information.info_base import MultiTimeSeries
    feat_names = [f for f in exp.dynamic_features.keys() 
                  if isinstance(f, str) and not isinstance(exp.dynamic_features[f], MultiTimeSeries)]
    
    if len(feat_names) < 2:
        print("Not enough features for disentanglement analysis.")
        return None, None, None
    
    # Use the built-in multifeature map
    multifeature_map = driada.intense.DEFAULT_MULTIFEATURE_MAP.copy()
    
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
    
    # Run disentanglement analysis with correct parameters
    
    try:
        disent_matrix, count_matrix = driada.intense.disentangle_all_selectivities(
            exp,
            feat_names,
            multifeature_map=multifeature_map,
            cell_bunch=list(mixed_candidates.keys())  # Focus on mixed selectivity neurons
        )
        
        print(f"Disentanglement analysis completed")
        
        return disent_matrix, count_matrix, feat_names
        
    except Exception as e:
        print(f"Error in disentanglement analysis: {str(e)}")
        return None, None, None


def interpret_disentanglement_results(exp, disent_matrix, count_matrix, feat_names, mixed_candidates):
    """Interpret and summarize disentanglement analysis results from matrices."""
    print("\n=== INTERPRETING DISENTANGLEMENT RESULTS ===")
    
    if disent_matrix is None or count_matrix is None:
        return [], [], []
    
    redundancy_cases = []
    synergy_cases = []
    independence_cases = []
    
    # Calculate relative disentanglement matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_disent_matrix = np.divide(disent_matrix, count_matrix) * 100
        rel_disent_matrix[count_matrix == 0] = np.nan
    
    # Extract disentanglement cases based on matrix values
    for i in range(len(feat_names)):
        for j in range(i + 1, len(feat_names)):  # Only upper triangle to avoid duplicates
            if count_matrix[i, j] > 0:  # Only consider pairs with data
                feat1 = feat_names[i]
                feat2 = feat_names[j]
                
                # Get disentanglement score (percentage)
                disent_score = rel_disent_matrix[i, j]
                
                if not np.isnan(disent_score):
                    # Classify based on disentanglement score
                    if disent_score < 30:  # Redundancy: feat2 dominates
                        redundancy_cases.append((f"{feat1}-{feat2}", (feat1, feat2), disent_score/100))
                    elif disent_score > 70:  # Synergy: feat1 dominates  
                        synergy_cases.append((f"{feat1}-{feat2}", (feat1, feat2), disent_score/100))
                    else:  # Independence: balanced
                        independence_cases.append((f"{feat1}-{feat2}", (feat1, feat2), disent_score/100))
    
    # Summary statistics
    total_pairs = len(redundancy_cases) + len(synergy_cases) + len(independence_cases)
    print(f"Found {len(redundancy_cases)} redundancy, {len(independence_cases)} independence, {len(synergy_cases)} synergy cases")
    
    # Show a few examples if available
    if redundancy_cases and len(redundancy_cases) > 0:
        feat1, feat2 = redundancy_cases[0][1]
        print(f"Example redundancy: {feat1} ↔ {feat2}")
    elif synergy_cases and len(synergy_cases) > 0:
        feat1, feat2 = synergy_cases[0][1]
        print(f"Example synergy: {feat1} + {feat2}")
    
    return redundancy_cases, synergy_cases, independence_cases


def create_disentanglement_visualization(exp, disent_matrix, count_matrix, feat_names, mixed_candidates, output_dir):
    """Create comprehensive visualization using production-grade visual.py functions."""
    print("\n=== CREATING DISENTANGLEMENT VISUALIZATIONS ===")
    
    if disent_matrix is None or count_matrix is None or not mixed_candidates:
        print("No disentanglement results to visualize")
        return
    
    try:
        # Create disentanglement heatmap using visual.py
        fig1, ax1 = driada.intense.plot_disentanglement_heatmap(
            disent_matrix, 
            count_matrix, 
            feat_names,
            title="Feature Disentanglement Analysis",
            figsize=(10, 8)
        )
        
        # Save heatmap
        heatmap_path = os.path.join(output_dir, 'disentanglement_heatmap.png')
        fig1.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        
    except Exception as e:
        print(f"Error creating disentanglement heatmap: {str(e)}")
        fig1 = None
    
    try:
        # Create comprehensive summary using visual.py
        fig2 = driada.intense.plot_disentanglement_summary(
            disent_matrix,
            count_matrix, 
            feat_names,
            title_prefix="Mixed Selectivity - ",
            figsize=(14, 10)
        )
        
        # Save summary
        summary_path = os.path.join(output_dir, 'disentanglement_summary.png')
        fig2.savefig(summary_path, dpi=150, bbox_inches='tight')
        
    except Exception as e:
        print(f"Error creating disentanglement summary: {str(e)}")
        fig2 = None
    
    # Create an additional plot showing example neuron-feature relationship
    if mixed_candidates:
        try:
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
            
            fig3 = driada.intense.plot_neuron_feature_pair(
                exp, example_cell, example_feature, 
                title=f"Example Mixed Selectivity: Neuron {example_cell} ↔ {example_feature}"
            )
            
            # Save example
            example_path = os.path.join(output_dir, 'example_mixed_selectivity.png')
            fig3.savefig(example_path, dpi=150, bbox_inches='tight')
            
        except Exception as e:
            print(f"Error creating example plot: {str(e)}")
    
    print(f"\nVisualizations saved: disentanglement_heatmap.png, disentanglement_summary.png")
    
    # Show plots if successful
    if fig1 is not None:
        plt.figure(fig1.number)
        plt.show()
    if fig2 is not None:
        plt.figure(fig2.number)
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
    disentanglement_results = analyze_disentanglement(exp, mixed_candidates)
    
    # Step 4: Interpret results
    redundancy_cases, synergy_cases, independence_cases = interpret_disentanglement_results(
        exp, disentanglement_results[0], disentanglement_results[1], disentanglement_results[2], mixed_candidates
    )
    
    # Step 5: Create visualizations
    if disentanglement_results[0] is not None:
        create_disentanglement_visualization(
            exp, disentanglement_results[0], disentanglement_results[1], disentanglement_results[2], 
            mixed_candidates, output_dir
        )
    
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
    
    print(f"\nAnalysis complete: {exp.n_cells} neurons, {len(mixed_candidates)} with mixed selectivity")
    print(f"Results: {len(redundancy_cases)} redundancy, {len(synergy_cases)} synergy cases")


if __name__ == "__main__":
    main()