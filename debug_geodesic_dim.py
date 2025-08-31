"""Debug script to analyze geodesic dimension test failure."""

import numpy as np
import matplotlib.pyplot as plt
from driada.dimensionality import geodesic_dimension
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path


def test_linear_subspace_debug():
    """Debug version of the failing test."""
    # Create 2D linear subspace in 10D
    np.random.seed(42)
    n_samples = 200
    
    # Generate basis vectors for 2D subspace
    basis = np.random.randn(10, 2)
    basis = np.linalg.qr(basis)[0][:, :2]  # Orthonormalize
    
    # Generate random coefficients in 2D
    coeffs = np.random.randn(n_samples, 2)
    
    # Project to 10D space
    data = coeffs @ basis.T
    
    # Add small noise to avoid degeneracy
    data += 1e-4 * np.random.randn(*data.shape)
    
    print(f"Data shape: {data.shape}")
    print(f"Data range: [{data.min():.4f}, {data.max():.4f}]")
    
    # Check intrinsic dimensionality using SVD
    U, S, Vt = np.linalg.svd(data - data.mean(axis=0))
    print(f"\nSingular values: {S[:5]}")
    print(f"Effective rank (S > 1e-10): {np.sum(S > 1e-10)}")
    
    # Build k-NN graph
    k = 7
    graph = kneighbors_graph(data, n_neighbors=k, mode='distance', include_self=False)
    graph_min = graph.minimum(graph.T)
    graph_sym = graph_min.tocsr()
    
    # Compute geodesic distances
    spmatrix = shortest_path(graph_sym, method='D', directed=False)
    all_dists = spmatrix.flatten()
    all_dists = all_dists[all_dists != 0]  # Remove self-distances
    all_dists = all_dists[np.isfinite(all_dists)]  # Remove infinite distances
    
    print(f"\nNumber of finite distances: {len(all_dists)}")
    print(f"Distance range: [{all_dists.min():.4f}, {all_dists.max():.4f}]")
    
    # Analyze distance distribution
    nbins = 500
    hist, bin_edges = np.histogram(all_dists, bins=nbins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find maximum
    dmax_idx = np.argmax(hist)
    dmax = bin_centers[dmax_idx]
    print(f"\nMaximum of distribution at: {dmax:.4f}")
    
    # Plot distance distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(all_dists, bins=50, density=True, alpha=0.7)
    plt.axvline(dmax, color='r', linestyle='--', label=f'Max at {dmax:.3f}')
    plt.xlabel('Geodesic distance')
    plt.ylabel('Density')
    plt.title('Distance distribution')
    plt.legend()
    
    # Normalize and analyze near maximum
    hist_norm, bin_edges_norm = np.histogram(all_dists / dmax, bins=nbins, density=True)
    bin_centers_norm = (bin_edges_norm[:-1] + bin_edges_norm[1:]) / 2
    
    plt.subplot(1, 3, 2)
    plt.plot(bin_centers_norm, hist_norm)
    plt.axvline(1.0, color='r', linestyle='--', label='x=1 (dmax)')
    plt.xlabel('Normalized distance')
    plt.ylabel('Density')
    plt.title('Normalized distribution')
    plt.xlim(0, 2)
    plt.legend()
    
    # Analyze left side near maximum
    std_norm = np.std(all_dists / dmax)
    mask = (
        (bin_centers_norm > 1 - 2 * std_norm)
        & (bin_centers_norm <= 1)
        & (hist_norm > 1e-6)
    )
    x_left = bin_centers_norm[mask]
    y_left = np.log(hist_norm[mask] / np.max(hist_norm))
    
    print(f"\nStd of normalized distances: {std_norm:.4f}")
    print(f"Analysis window: [{1 - 2*std_norm:.4f}, 1.0]")
    print(f"Number of points in analysis window: {len(x_left)}")
    
    if len(x_left) > 0:
        plt.subplot(1, 3, 3)
        plt.plot(x_left, y_left, 'bo', label='Data')
        
        # Test theoretical curves for different dimensions
        for D in [1.0, 2.0, 3.0]:
            x_theory = np.linspace(x_left.min(), x_left.max(), 100)
            x_clipped = np.clip(x_theory, 1e-10, 1 - 1e-10)
            y_theory = D * np.log(np.sin(x_clipped * np.pi / 2))
            plt.plot(x_theory, y_theory, '--', label=f'D={D}')
        
        plt.xlabel('Normalized distance')
        plt.ylabel('Log density')
        plt.title('Fit near maximum')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('geodesic_debug.png', dpi=150)
    plt.close()
    
    # Run geodesic dimension estimation
    dim7 = geodesic_dimension(data, k=7)
    dim10 = geodesic_dimension(data, k=10)
    dim15 = geodesic_dimension(data, k=15)
    
    print(f"\nGeodesic dimension estimates:")
    print(f"  k=7:  {dim7:.3f}")
    print(f"  k=10: {dim10:.3f}")
    print(f"  k=15: {dim15:.3f}")
    
    # Test with different parameters
    dim_step = geodesic_dimension(data, k=7, dim_step=0.05)
    print(f"\nWith finer step (0.05): {dim_step:.3f}")
    
    # Analyze k-NN graph connectivity
    print(f"\nGraph analysis (k={k}):")
    n_components = 0
    visited = np.zeros(n_samples, dtype=bool)
    for i in range(n_samples):
        if not visited[i]:
            # BFS to find component
            queue = [i]
            visited[i] = True
            component_size = 1
            while queue:
                node = queue.pop(0)
                neighbors = graph_sym[node].nonzero()[1]
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
                        component_size += 1
            if component_size > 1:
                n_components += 1
    
    print(f"Number of components: {n_components}")
    
    return dim7


if __name__ == "__main__":
    test_linear_subspace_debug()