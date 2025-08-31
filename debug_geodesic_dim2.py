"""Further debug analysis of geodesic dimension issue."""

import numpy as np
import matplotlib.pyplot as plt
from driada.dimensionality import geodesic_dimension, nn_dimension, correlation_dimension
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path


def analyze_2d_plane_issue():
    """Analyze why 2D plane is detected as 1D."""
    
    print("=== Testing different data generation methods ===\n")
    
    # Method 1: Original test data
    print("1. Original test method (QR basis):")
    np.random.seed(42)
    basis = np.random.randn(10, 2)
    basis = np.linalg.qr(basis)[0][:, :2]
    coeffs = np.random.randn(200, 2)
    data1 = coeffs @ basis.T + 1e-4 * np.random.randn(200, 10)
    
    U, S, _ = np.linalg.svd(data1 - data1.mean(axis=0))
    print(f"   Singular values: {S[:5]}")
    print(f"   Effective rank: {np.sum(S > 1e-10)}")
    
    # Method 2: Direct 2D plane
    print("\n2. Direct 2D plane in first 2 coords:")
    x = np.random.randn(200)
    y = np.random.randn(200)
    data2 = np.zeros((200, 10))
    data2[:, 0] = x
    data2[:, 1] = y
    data2 += 1e-4 * np.random.randn(200, 10)
    
    U, S, _ = np.linalg.svd(data2 - data2.mean(axis=0))
    print(f"   Singular values: {S[:5]}")
    print(f"   Effective rank: {np.sum(S > 1e-10)}")
    
    # Method 3: Spread out 2D data
    print("\n3. Well-spread 2D data:")
    theta = np.linspace(0, 2*np.pi, 100)
    r = np.linspace(0.1, 2, 100)
    np.random.shuffle(r)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points_2d = np.column_stack([x, y])
    # Embed in 10D
    basis3 = np.random.randn(10, 2)
    basis3 = np.linalg.qr(basis3)[0][:, :2]
    data3 = points_2d @ basis3.T + 1e-4 * np.random.randn(100, 10)
    
    U, S, _ = np.linalg.svd(data3 - data3.mean(axis=0))
    print(f"   Singular values: {S[:5]}")
    print(f"   Effective rank: {np.sum(S > 1e-10)}")
    
    # Test all methods with different estimators
    print("\n=== Dimension estimates ===")
    
    methods = [
        ("Original (random)", data1),
        ("Direct 2D", data2),
        ("Structured 2D", data3)
    ]
    
    for name, data in methods:
        print(f"\n{name}:")
        
        # Test different k values for geodesic
        for k in [7, 10, 15, 20, 30]:
            try:
                dim = geodesic_dimension(data, k=k)
                print(f"  Geodesic (k={k:2d}): {dim:.3f}")
            except Exception as e:
                print(f"  Geodesic (k={k:2d}): Failed - {str(e)}")
        
        # Compare with other methods
        dim_nn = nn_dimension(data, k=10)
        dim_corr = correlation_dimension(data, n_bins=15)
        print(f"  NN dimension:     {dim_nn:.3f}")
        print(f"  Correlation dim:  {dim_corr:.3f}")
    
    # Analyze connectivity
    print("\n=== Graph connectivity analysis ===")
    for k in [5, 7, 10, 15, 20]:
        graph = kneighbors_graph(data1, n_neighbors=k, mode='distance', include_self=False)
        graph_sym = graph.minimum(graph.T)
        
        # Check connectivity
        spmatrix = shortest_path(graph_sym, method='D', directed=False)
        n_inf = np.sum(np.isinf(spmatrix))
        print(f"k={k:2d}: {n_inf} infinite distances out of {200*199} = {100*n_inf/(200*199):.1f}% disconnected")
    
    # Visualize the data distribution
    print("\n=== Creating visualization ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, (name, data) in enumerate(methods):
        # Project to 2D using PCA
        data_centered = data - data.mean(axis=0)
        U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
        data_2d = U[:, :2] @ np.diag(S[:2])
        
        ax = axes[0, i]
        ax.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.6)
        ax.set_title(f'{name} - PCA projection')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.axis('equal')
        
        # Plot k-NN graph for k=7
        ax = axes[1, i]
        graph = kneighbors_graph(data, n_neighbors=7, mode='connectivity', include_self=False)
        
        # Draw edges
        rows, cols = graph.nonzero()
        for row, col in zip(rows, cols):
            if row < col:  # Draw each edge once
                ax.plot([data_2d[row, 0], data_2d[col, 0]], 
                       [data_2d[row, 1], data_2d[col, 1]], 
                       'b-', alpha=0.1, linewidth=0.5)
        
        ax.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.8, s=20, c='red', zorder=10)
        ax.set_title(f'{name} - k-NN graph (k=7)')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('geodesic_debug2.png', dpi=150)
    plt.close()
    
    print("\nVisualization saved to geodesic_debug2.png")


if __name__ == "__main__":
    analyze_2d_plane_issue()