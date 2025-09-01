"""
Example: Neural Network Analysis with Cell-Cell Significance

This example demonstrates:
1. Computing cell-cell functional connectivity using INTENSE
2. Creating networks from significance results
3. Analyzing network properties using spectral methods
4. Visualizing functional modules and network structure
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp
from matplotlib.patches import Rectangle

# DRIADA imports
import driada
from driada import TimeSeries
from driada.network import Network
from driada.network.drawing import draw_degree_distr, draw_spectrum
from driada.intense import compute_cell_cell_significance


def create_modular_experiment(n_modules=3, neurons_per_module=20, duration=300, seed=42):
    """Create synthetic experiment with modular structure."""
    np.random.seed(seed)
    
    # Total neurons
    n_neurons = n_modules * neurons_per_module
    
    # Generate base experiment
    exp = driada.generate_synthetic_exp(
        n_dfeats=3,  # Dynamic features
        n_cfeats=0,  # No categorical features needed
        nneurons=n_neurons,
        duration=duration,
        seed=seed
    )
    
    # Create modular structure in neural activity
    # Neurons within modules have correlated activity
    time_points = int(duration * 20)  # 20 Hz
    
    for module_idx in range(n_modules):
        # Get neurons in this module
        start_idx = module_idx * neurons_per_module
        end_idx = (module_idx + 1) * neurons_per_module
        
        # Create shared signal for this module
        module_signal = np.random.randn(time_points) * 0.5
        module_signal = np.convolve(module_signal, np.ones(10)/10, mode='same')  # Smooth
        
        # Add module signal to each neuron with some noise
        for neuron_idx in range(start_idx, end_idx):
            neuron = exp.neurons[neuron_idx]
            # Add module signal + individual noise
            neuron.ca.data += module_signal + np.random.randn(time_points) * 0.3
    
    # Add some inter-module connections (weaker correlations)
    for i in range(5):  # 5 inter-module pairs
        n1 = np.random.randint(0, n_neurons)
        n2 = np.random.randint(0, n_neurons)
        if n1 // neurons_per_module != n2 // neurons_per_module:  # Different modules
            shared = np.random.randn(time_points) * 0.2
            shared = np.convolve(shared, np.ones(10)/10, mode='same')
            exp.neurons[n1].ca.data += shared
            exp.neurons[n2].ca.data += shared
    
    return exp, n_modules, neurons_per_module


def analyze_cell_cell_network(exp, data_type='calcium', pval_thr=0.01):
    """Compute and analyze cell-cell functional network."""
    
    print("=" * 60)
    print("Computing Cell-Cell Functional Connectivity")
    print("=" * 60)
    
    # Compute cell-cell significance
    sim_mat, sig_mat, pval_mat, cells, info = compute_cell_cell_significance(
        exp,
        data_type=data_type,
        n_shuffles_stage1=50,     # Reduced for example
        n_shuffles_stage2=1000,   # Reduced for example
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
    print(f"Average degree: {np.mean(degrees):.2f} ± {np.std(degrees):.2f}")
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


def visualize_results(sim_mat, sig_mat, cells, net, module_assignment, n_modules_true=None):
    """Create comprehensive visualization of results."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Similarity matrix
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(sim_mat, cmap='hot', aspect='auto')
    ax1.set_title('Similarity Matrix (MI)', fontsize=12)
    ax1.set_xlabel('Neuron ID')
    ax1.set_ylabel('Neuron ID')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    # Add module boundaries if known
    if n_modules_true is not None:
        neurons_per_module = len(cells) // n_modules_true
        for i in range(1, n_modules_true):
            ax1.axhline(i * neurons_per_module - 0.5, color='cyan', linewidth=2)
            ax1.axvline(i * neurons_per_module - 0.5, color='cyan', linewidth=2)
    
    # 2. Significance matrix
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(sig_mat, cmap='RdBu_r', aspect='auto', vmin=0, vmax=1)
    ax2.set_title('Significance Matrix', fontsize=12)
    ax2.set_xlabel('Neuron ID')
    ax2.set_ylabel('Neuron ID')
    
    # 3. Network visualization with modules
    ax3 = plt.subplot(2, 3, 3)
    
    # Create layout emphasizing modules
    pos = nx.spring_layout(net.graph, k=2, iterations=50, seed=42)
    
    # Draw nodes colored by module
    node_colors = [module_assignment.get(node, 0) for node in net.graph.nodes()]
    nx.draw_networkx_nodes(net.graph, pos, node_color=node_colors, 
                          cmap='tab10', node_size=100, ax=ax3)
    
    # Draw edges
    if net.weighted:
        edges = net.graph.edges()
        weights = [net.graph[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(net.graph, pos, alpha=0.3, width=weights, ax=ax3)
    else:
        nx.draw_networkx_edges(net.graph, pos, alpha=0.3, ax=ax3)
    
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
    module_sizes = {}
    for node, module in module_assignment.items():
        module_sizes[module] = module_sizes.get(module, 0) + 1
    
    modules = list(module_sizes.keys())
    sizes = list(module_sizes.values())
    colors = plt.cm.tab10(np.arange(len(modules)))
    
    ax5.bar(modules, sizes, color=colors[:len(modules)])
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


def main():
    """Run complete cell-cell network analysis example."""
    
    # Create synthetic experiment with modular structure
    print("Creating synthetic experiment with modular structure...")
    exp, n_modules_true, neurons_per_module = create_modular_experiment(
        n_modules=3, 
        neurons_per_module=20, 
        duration=300
    )
    print(f"Created {len(exp.neurons)} neurons in {n_modules_true} true modules")
    
    # Compute cell-cell significance
    sim_mat, sig_mat, pval_mat, cells, info = analyze_cell_cell_network(
        exp, 
        data_type='calcium',
        pval_thr=0.01
    )
    
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
        module_assignment, n_modules_true
    )
    plt.savefig('cell_cell_network_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Additional network visualizations
    print("\nCreating network property visualizations...")
    
    # Degree distribution (log-log)
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Standard degree distribution
    ax1.set_title("Degree Distribution")
    draw_degree_distr(net_weighted, ax=ax1)
    
    # Spectral analysis
    ax2.set_title("Laplacian Spectrum")
    draw_spectrum(net_weighted, mode='lap', ax=ax2)
    
    plt.tight_layout()
    plt.savefig('network_properties.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nAnalysis complete! Check the generated plots.")
    
    return exp, net_weighted, communities


if __name__ == "__main__":
    exp, net, communities = main()