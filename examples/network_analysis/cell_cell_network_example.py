"""
Example: Neural Network Analysis with Cell-Cell Significance

This example demonstrates:
1. Computing cell-cell functional connectivity using INTENSE
2. Creating networks from significance results
3. Analyzing network properties using spectral methods
4. Visualizing functional modules and network structure

Performance: ~2 min for 120 neurons (ds=5, 10k shuffles)
File size: 13 KB sparse network format
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp

# DRIADA imports
import driada
from driada.network import Network
from driada.network.drawing import draw_degree_distr, draw_spectrum
from driada.intense import compute_cell_cell_significance
from driada.utils.plot import create_default_figure


def create_modular_experiment(duration=300, seed=42):
    """
    Create synthetic experiment with hierarchical modular structure.

    Creates 120 neurons in 6 functional groups:
    - 3 single-feature modules (30 neurons each): respond to event_0, event_1, or event_2
    - 3 dual-feature modules (10 neurons each): respond to pairs of events in OR mode
      (event_0 OR event_1, event_0 OR event_2, event_1 OR event_2)

    This creates a realistic hierarchical network with both specialized
    and multi-selective neurons.
    """
    from driada.experiment.synthetic import generate_tuned_selectivity_exp

    # Create population with mixed selectivity
    population = [
        # Single-feature modules (30 neurons each)
        {
            "name": "event_0_cells",
            "count": 30,
            "features": ["event_0"],
        },
        {
            "name": "event_1_cells",
            "count": 30,
            "features": ["event_1"],
        },
        {
            "name": "event_2_cells",
            "count": 30,
            "features": ["event_2"],
        },
        # Dual-feature modules (10 neurons each, OR combination)
        {
            "name": "event_0_or_1_cells",
            "count": 10,
            "features": ["event_0", "event_1"],
            "combination": "or",
        },
        {
            "name": "event_0_or_2_cells",
            "count": 10,
            "features": ["event_0", "event_2"],
            "combination": "or",
        },
        {
            "name": "event_1_or_2_cells",
            "count": 10,
            "features": ["event_1", "event_2"],
            "combination": "or",
        },
    ]

    # Generate experiment with hierarchical structure
    exp = generate_tuned_selectivity_exp(
        population=population,
        n_discrete_features=3,  # Three distinct events
        duration=duration,
        fps=20.0,
        baseline_rate=0.05,
        peak_rate=2.0,
        decay_time=2.0,
        calcium_noise=0.02,
        seed=seed,
        verbose=True
    )

    # Return info about true module structure
    n_modules = 6
    module_sizes = [30, 30, 30, 10, 10, 10]

    return exp, n_modules, module_sizes


def analyze_cell_cell_network(exp, data_type='calcium', pval_thr=0.01):
    """Compute and analyze cell-cell functional network."""
    
    print("=" * 60)
    print("Computing Cell-Cell Functional Connectivity")
    print("=" * 60)
    
    # Compute cell-cell significance
    sim_mat, sig_mat, pval_mat, cells, info = compute_cell_cell_significance(
        exp,
        data_type=data_type,
        ds=5,                      # Downsample by 5x for speed (~5x faster)
        n_shuffles_stage1=100,     # Stage 1 screening
        n_shuffles_stage2=10000,   # FFT makes high shuffle counts fast!
        pval_thr=pval_thr,
        multicomp_correction='holm',
        verbose=True
    )
    
    print(f"\nAnalysis complete!")
    print(f"Total neuron pairs: {len(cells) * (len(cells)-1) / 2:.0f}")
    print(f"Significant connections: {np.sum(sig_mat) / 2:.0f}")  # Divide by 2 for symmetry
    print(f"Connection density: {np.sum(sig_mat) / (len(cells) * (len(cells)-1)):.3f}")
    
    return sim_mat, sig_mat, pval_mat, cells, info


def create_functional_network(sig_mat, sim_mat, cells):
    """Create Network object from significance results."""
    
    print("\n" + "=" * 60)
    print("Creating Functional Network")
    print("=" * 60)
    
    # Create binary network from significant connections
    sig_sparse = sp.csr_matrix(sig_mat)
    net_binary = Network(
        adj=sig_sparse, 
        preprocessing='giant_cc',
        name='Neural Functional Network (Binary)'
    )
    
    # Create weighted network (similarity values for significant edges only)
    weighted_mat = sim_mat * sig_mat  # Zero out non-significant
    weighted_sparse = sp.csr_matrix(weighted_mat)
    net_weighted = Network(
        adj=weighted_sparse,
        preprocessing='giant_cc', 
        name='Neural Functional Network (Weighted)'
    )
    
    print(f"Binary network: {net_binary.n} nodes, {net_binary.graph.number_of_edges()} edges")
    print(f"Weighted network: {net_weighted.n} nodes, {net_weighted.graph.number_of_edges()} edges")
    
    return net_binary, net_weighted


def detect_functional_modules(net, cells):
    """Detect functional modules using community detection."""
    
    print("\n" + "=" * 60)
    print("Detecting Functional Modules")
    print("=" * 60)
    
    # Use Louvain community detection
    import networkx.algorithms.community as nx_comm
    
    # For weighted network, use weight attribute
    if net.weighted:
        communities = nx_comm.louvain_communities(net.graph, weight='weight', seed=42)
    else:
        communities = nx_comm.louvain_communities(net.graph, seed=42)
    
    print(f"Found {len(communities)} functional modules:")
    for i, community in enumerate(communities):
        print(f"  Module {i+1}: {len(community)} neurons")
    
    # Create module assignment dictionary
    module_assignment = {}
    for module_idx, community in enumerate(communities):
        for node in community:
            module_assignment[node] = module_idx
    
    return communities, module_assignment


def analyze_network_properties(net):
    """Analyze various network properties."""
    
    print("\n" + "=" * 60)
    print("Network Properties Analysis")
    print("=" * 60)
    
    # Basic properties
    print(f"Network type: {'Directed' if net.directed else 'Undirected'}, "
          f"{'Weighted' if net.weighted else 'Binary'}")
    print(f"Number of nodes: {net.n}")
    print(f"Number of edges: {net.graph.number_of_edges()}")
    
    # Degree statistics
    degrees = [d for n, d in net.graph.degree()]
    print(f"Average degree: {np.mean(degrees):.2f} +- {np.std(degrees):.2f}")
    print(f"Max degree: {np.max(degrees)}")
    
    # Clustering coefficient
    if not net.directed:
        clustering = nx.average_clustering(net.graph)
        print(f"Average clustering coefficient: {clustering:.3f}")
    
    # Connected components
    if net.directed:
        n_weak = nx.number_weakly_connected_components(net.graph)
        n_strong = nx.number_strongly_connected_components(net.graph)
        print(f"Weakly connected components: {n_weak}")
        print(f"Strongly connected components: {n_strong}")
    else:
        n_cc = nx.number_connected_components(net.graph)
        print(f"Connected components: {n_cc}")
    
    # Spectral properties
    print("\nSpectral Analysis:")
    adj_spectrum = net.get_spectrum('adj')
    print(f"Largest eigenvalue (spectral radius): {np.max(np.abs(adj_spectrum)):.3f}")
    
    if not net.directed:
        lap_spectrum = net.get_spectrum('lap')
        # Second smallest eigenvalue (algebraic connectivity)
        sorted_lap = np.sort(np.real(lap_spectrum))
        if len(sorted_lap) > 1:
            print(f"Algebraic connectivity: {sorted_lap[1]:.3f}")
    
    return degrees


def visualize_results(sim_mat, sig_mat, cells, net, module_assignment, module_sizes_true=None):
    """Create comprehensive visualization of results with consistent module colors."""

    fig = plt.figure(figsize=(16, 12))

    # Get unique modules and create consistent color mapping
    unique_modules = sorted(set(module_assignment.values()))
    n_modules = len(unique_modules)
    module_colors = plt.cm.tab10(np.linspace(0, 1, n_modules))
    module_to_color = {mod: module_colors[i] for i, mod in enumerate(unique_modules)}

    # 1. Similarity matrix
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(sim_mat, cmap='hot', aspect='auto')
    ax1.set_title('Similarity Matrix (MI)', fontsize=12)
    ax1.set_xlabel('Neuron ID')
    ax1.set_ylabel('Neuron ID')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # Add module boundaries if known (for 90/90/90/10/10/10 structure)
    if module_sizes_true is not None:
        cumsum = np.cumsum([0] + module_sizes_true)
        for boundary in cumsum[1:-1]:
            ax1.axhline(boundary - 0.5, color='cyan', linewidth=2)
            ax1.axvline(boundary - 0.5, color='cyan', linewidth=2)

    # 2. Significance matrix
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(sig_mat, cmap='RdBu_r', aspect='auto', vmin=0, vmax=1)
    ax2.set_title('Significance Matrix', fontsize=12)
    ax2.set_xlabel('Neuron ID')
    ax2.set_ylabel('Neuron ID')

    # 3. Network visualization with modules using spring layout
    ax3 = plt.subplot(2, 3, 3)

    # Use spring layout with parameters optimized for clustering
    pos = nx.spring_layout(
        net.graph,
        k=1.5/np.sqrt(len(net.graph)),  # Optimal distance
        iterations=100,  # More iterations for better convergence
        seed=42
    )

    # Draw nodes colored by detected module
    node_colors = [module_to_color[module_assignment[node]] for node in net.graph.nodes()]
    nx.draw_networkx_nodes(
        net.graph, pos,
        node_color=node_colors,
        node_size=20,  # Smaller for 300 nodes
        ax=ax3,
        alpha=0.8
    )

    # Draw edges
    if net.weighted:
        edges = net.graph.edges()
        weights = [net.graph[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(net.graph, pos, alpha=0.1, width=weights, ax=ax3)
    else:
        nx.draw_networkx_edges(net.graph, pos, alpha=0.1, ax=ax3)

    ax3.set_title('Functional Network Modules', fontsize=12)
    ax3.axis('off')

    # 4. Degree distribution
    ax4 = plt.subplot(2, 3, 4)
    degrees = [d for n, d in net.graph.degree()]
    ax4.hist(degrees, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax4.set_xlabel('Degree')
    ax4.set_ylabel('Count')
    ax4.set_title('Degree Distribution', fontsize=12)
    ax4.grid(True, alpha=0.3)

    # 5. Module size distribution
    ax5 = plt.subplot(2, 3, 5)
    detected_sizes = {}
    for node, module in module_assignment.items():
        detected_sizes[module] = detected_sizes.get(module, 0) + 1

    modules = sorted(detected_sizes.keys())
    sizes = [detected_sizes[m] for m in modules]
    colors = [module_to_color[m] for m in modules]

    ax5.bar(modules, sizes, color=colors)
    ax5.set_xlabel('Module ID')
    ax5.set_ylabel('Number of Neurons')
    ax5.set_title('Module Sizes', fontsize=12)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. P-value distribution
    ax6 = plt.subplot(2, 3, 6)
    # Get upper triangle p-values (excluding diagonal)
    triu_indices = np.triu_indices_from(sig_mat, k=1)
    pvals_upper = sim_mat[triu_indices]
    
    ax6.hist(pvals_upper, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax6.set_xlabel('Similarity (MI)')
    ax6.set_ylabel('Count')
    ax6.set_title('Similarity Distribution', fontsize=12)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def save_network(filename, sim_mat, sig_mat, pval_mat, cells, info):
    """
    Save network as sparse adjacency matrix with essential metadata only.

    Saves only:
    - Sparse adjacency matrix (significant connections)
    - Similarity/p-values for significant edges only
    - Cell IDs and scalar metadata
    - Excludes large shuffle arrays (random_shifts, me_total)

    Parameters
    ----------
    filename : str
        Output filename (without extension). Will save as .npz file.
    sim_mat : np.ndarray
        Similarity matrix
    sig_mat : np.ndarray
        Significance matrix
    pval_mat : np.ndarray
        P-value matrix
    cells : list
        List of cell IDs
    info : dict
        Metadata dictionary from INTENSE
    """
    import scipy.sparse as sp

    # Get indices of significant connections
    sig_indices = np.where(sig_mat > 0)

    # Extract only significant edges (sparse format)
    sparse_data = {
        'sig_mat_data': sig_mat[sig_indices],  # Binary significance
        'sim_values': sim_mat[sig_indices],    # Similarity for significant edges
        'pval_values': pval_mat[sig_indices],  # P-values for significant edges
        'sig_indices_i': sig_indices[0],       # Row indices
        'sig_indices_j': sig_indices[1],       # Column indices
        'matrix_shape': np.array(sig_mat.shape),  # Shape for reconstruction
        'cells': np.array(cells),
    }

    # Add optimal delays if available (only for significant edges)
    if 'optimal_delays' in info:
        sparse_data['optimal_delays'] = info['optimal_delays'][sig_indices]

    # Add only scalar metadata (exclude huge shuffle arrays!)
    exclude_keys = {'random_shifts1', 'me_total1', 'random_shifts2', 'me_total2'}
    for k, v in info.items():
        if k not in exclude_keys and isinstance(v, (int, float, str, np.integer, np.floating)):
            sparse_data[f'info_{k}'] = v

    # Save as compressed NPZ
    filename = filename if filename.endswith('.npz') else f"{filename}.npz"
    np.savez_compressed(filename, **sparse_data)

    # Report size savings
    n_sig = len(sig_indices[0])
    n_total = sig_mat.shape[0] * sig_mat.shape[1]
    density = n_sig / n_total
    print(f"Network saved to {filename}")
    print(f"  Significant edges: {n_sig}/{n_total} ({density*100:.2f}% density)")
    print(f"  Sparse storage: ~{n_sig * 32 / 1024:.1f} KB (vs {n_total * 8 / 1024:.1f} KB full matrix)")


def load_network(filename):
    """
    Load network from sparse format.

    Parameters
    ----------
    filename : str
        Input filename (with or without .npz extension)

    Returns
    -------
    sim_mat : np.ndarray
        Similarity matrix (reconstructed from sparse)
    sig_mat : np.ndarray
        Significance matrix (reconstructed from sparse)
    pval_mat : np.ndarray
        P-value matrix (reconstructed from sparse)
    cells : list
        List of cell IDs
    info : dict
        Metadata dictionary
    """
    if not filename.endswith('.npz'):
        filename = f"{filename}.npz"

    data = np.load(filename, allow_pickle=True)

    # Load sparse format
    shape = tuple(data['matrix_shape'])
    sig_indices_i = data['sig_indices_i']
    sig_indices_j = data['sig_indices_j']

    # Initialize full matrices
    sig_mat = np.zeros(shape)
    sim_mat = np.zeros(shape)
    pval_mat = np.ones(shape)  # Initialize with 1.0 (non-significant)

    # Fill in significant edges
    sig_mat[sig_indices_i, sig_indices_j] = data['sig_mat_data']
    sim_mat[sig_indices_i, sig_indices_j] = data['sim_values']
    pval_mat[sig_indices_i, sig_indices_j] = data['pval_values']

    cells = data['cells'].tolist()

    # Reconstruct info dict (with optimal delays if present)
    info = {}
    if 'optimal_delays' in data:
        info['optimal_delays'] = np.zeros(shape)
        info['optimal_delays'][sig_indices_i, sig_indices_j] = data['optimal_delays']

    # Add scalar metadata
    for k, v in data.items():
        if k.startswith('info_'):
            info[k.replace('info_', '')] = v

    print(f"Network loaded from {filename}")
    print(f"  Neurons: {len(cells)}")
    print(f"  Significant connections: {len(sig_indices_i) // 2:.0f} (undirected)")

    return sim_mat, sig_mat, pval_mat, cells, info


def main():
    """Run complete cell-cell network analysis example with save/load support."""
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Neural network analysis with INTENSE cell-cell connectivity'
    )
    parser.add_argument(
        '--load',
        type=str,
        default=None,
        help='Load pre-computed network from .npz file instead of computing'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Save computed network to .npz file'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=300,
        help='Recording duration in seconds (default: 300)'
    )
    parser.add_argument(
        '--pval',
        type=float,
        default=0.001,
        help='P-value threshold for significance (default: 0.001)'
    )
    args = parser.parse_args()

    # Either load network or compute it
    if args.load:
        print(f"Loading pre-computed network from {args.load}...")
        sim_mat, sig_mat, pval_mat, cells, info = load_network(args.load)
        module_sizes_true = [30, 30, 30, 10, 10, 10]  # Known structure
    else:
        # Create synthetic experiment with hierarchical modular structure
        print("Creating synthetic experiment with hierarchical modular structure...")
        print("  120 neurons: 30+30+30 (single-feature) + 10+10+10 (dual-feature)")
        exp, n_modules_true, module_sizes_true = create_modular_experiment(
            duration=args.duration
        )
        print(f"Created {len(exp.neurons)} neurons in {n_modules_true} functional groups")

        # Compute cell-cell significance
        sim_mat, sig_mat, pval_mat, cells, info = analyze_cell_cell_network(
            exp,
            data_type='calcium',
            pval_thr=args.pval
        )

        # Save network if requested
        if args.save:
            save_network(args.save, sim_mat, sig_mat, pval_mat, cells, info)

    # Create functional networks
    net_binary, net_weighted = create_functional_network(sig_mat, sim_mat, cells)

    # Detect functional modules
    communities, module_assignment = detect_functional_modules(net_weighted, cells)

    # Analyze network properties
    degrees = analyze_network_properties(net_weighted)

    # Visualize results
    print("\nCreating visualizations...")
    fig = visualize_results(
        sim_mat, sig_mat, cells, net_weighted,
        module_assignment, module_sizes_true
    )
    plt.savefig('cell_cell_network_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Additional network visualizations
    print("\nCreating network property visualizations...")

    # Degree distribution with production-quality styling
    fig1, ax1 = create_default_figure(figsize=(10, 8), dpi=300)
    ax1 = draw_degree_distr(net_weighted, ax=ax1)
    ax1.set_title("Degree Distribution")
    plt.savefig('network_degree_distribution.png', dpi=300, bbox_inches='tight')

    # Laplacian spectrum with production-quality styling
    fig2, ax2 = create_default_figure(figsize=(10, 8), dpi=300)
    ax2 = draw_spectrum(net_weighted, mode='lap', ax=ax2)
    ax2.set_title("Laplacian Spectrum")
    plt.savefig('network_spectrum.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nAnalysis complete! Check the generated plots.")
    if args.save and not args.load:
        print(f"Network saved to {args.save}.npz for future use.")
        print(f"  Re-run with: python {__file__} --load {args.save}")

    return net_weighted, communities


if __name__ == "__main__":
    net, communities = main()