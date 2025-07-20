"""
Extract circular manifold from head direction cells using dimensionality reduction.

This example demonstrates how to:
1. Generate a population of head direction cells with circular tuning
2. Estimate the intrinsic dimensionality of the neural population
3. Apply various dimensionality reduction methods to extract the underlying circular structure
4. Validate the reconstruction against ground truth head direction
5. Visualize the extracted manifold

The key insight is that head direction cells form a ring attractor network,
and their population activity should lie on a circular (1D) manifold embedded in
high-dimensional neural space.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
import umap

# Import DRIADA modules
from driada.experiment import generate_circular_manifold_exp
from driada.dimensionality import (
    eff_dim, nn_dimension, correlation_dimension,
    pca_dimension, pca_dimension_profile, effective_rank
)


def estimate_dimensionality(neural_data, methods=None):
    """
    Estimate intrinsic dimensionality using multiple methods from DRIADA.
    
    Parameters
    ----------
    neural_data : ndarray
        Neural activity matrix (n_neurons x n_timepoints)
    
    methods : list of str, optional
        List of dimensionality estimation methods to use. If None, uses all available.
        Available methods:
        - 'pca_90': PCA dimension for 90% variance explained
        - 'pca_95': PCA dimension for 95% variance explained
        - 'effective_rank': Effective rank based on eigenvalue entropy
        - 'participation_ratio': Participation ratio (quadratic Renyi entropy)
        - 'nn_dimension': k-NN based intrinsic dimension estimator
        - 'correlation_dim': Correlation dimension (Grassberger-Procaccia)
    
    Returns
    -------
    dim_estimates : dict
        Dictionary of dimensionality estimates from each method
    """
    # Default to all methods if none specified
    if methods is None:
        methods = ['pca_90', 'pca_95', 'effective_rank', 'participation_ratio', 
                  'nn_dimension', 'correlation_dim']
    
    dim_estimates = {}
    
    # Transpose data for methods that expect (n_samples, n_features)
    data_transposed = neural_data.T
    
    # Linear methods
    if 'pca_90' in methods:
        dim_estimates['pca_90'] = pca_dimension(data_transposed, threshold=0.90)
    
    if 'pca_95' in methods:
        dim_estimates['pca_95'] = pca_dimension(data_transposed, threshold=0.95)
    
    if 'effective_rank' in methods:
        dim_estimates['effective_rank'] = effective_rank(data_transposed)
    
    # Nonlinear methods
    if 'nn_dimension' in methods:
        try:
            dim_estimates['nn_dimension'] = nn_dimension(data_transposed, k=2)
        except Exception as e:
            print(f"  Warning: nn_dimension failed: {e}")
            dim_estimates['nn_dimension'] = np.nan
    
    if 'correlation_dim' in methods:
        try:
            dim_estimates['correlation_dim'] = correlation_dimension(data_transposed)
        except Exception as e:
            print(f"  Warning: correlation_dimension failed: {e}")
            dim_estimates['correlation_dim'] = np.nan
    
    # Effective dimensionality (participation ratio)
    if 'participation_ratio' in methods:
        dim_estimates['participation_ratio'] = eff_dim(neural_data, enable_correction=False, q=2)
    
    return dim_estimates


def plot_eigenspectrum(neural_data):
    """Plot eigenvalue spectrum of correlation matrix."""
    # Compute correlation matrix
    data_centered = neural_data - np.mean(neural_data, axis=1, keepdims=True)
    corr_mat = np.corrcoef(data_centered)
    
    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(corr_mat)[::-1]  # Descending order
    eigenvalues = eigenvalues[eigenvalues > 0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Eigenvalue spectrum
    ax1.plot(eigenvalues, 'o-', markersize=4)
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('Eigenvalue Spectrum')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative variance explained
    cumvar = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    ax2.plot(cumvar, 'o-', markersize=4)
    ax2.axhline(0.9, color='r', linestyle='--', label='90% variance')
    ax2.axhline(0.95, color='orange', linestyle='--', label='95% variance')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance Explained')
    ax2.set_title('Cumulative Variance Explained')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def extract_manifold_pca(neural_data, n_components=2):
    """Extract manifold using Principal Component Analysis."""
    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(neural_data.T)
    explained_var = pca.explained_variance_ratio_
    return embedding, explained_var


def extract_manifold_isomap(neural_data, n_components=2, n_neighbors=10):
    """Extract manifold using Isomap (geodesic distances)."""
    isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    embedding = isomap.fit_transform(neural_data.T)
    return embedding


def extract_manifold_umap(neural_data, n_components=2, n_neighbors=15):
    """Extract manifold using UMAP (Uniform Manifold Approximation)."""
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, 
                        min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(neural_data.T)
    return embedding


def extract_manifold_tsne(neural_data, n_components=2, perplexity=30):
    """Extract manifold using t-SNE."""
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                random_state=42)
    embedding = tsne.fit_transform(neural_data.T)
    return embedding


def compute_circular_coordinates(embedding):
    """Convert 2D embedding to circular coordinates (angles)."""
    # Center the embedding
    centered = embedding - np.mean(embedding, axis=0)
    
    # Compute angles
    angles = np.arctan2(centered[:, 1], centered[:, 0])
    angles = np.mod(angles, 2 * np.pi)  # Ensure [0, 2π]
    
    return angles


def evaluate_reconstruction(true_angles, reconstructed_angles):
    """Evaluate how well the reconstruction preserves circular structure."""
    # Handle circular distance
    diff = np.abs(true_angles - reconstructed_angles)
    circular_diff = np.minimum(diff, 2*np.pi - diff)
    
    # Compute circular correlation
    # Using complex representation for circular data
    true_complex = np.exp(1j * true_angles)
    recon_complex = np.exp(1j * reconstructed_angles)
    
    # Circular correlation coefficient
    r = np.abs(np.mean(true_complex * np.conj(recon_complex)))
    
    # Mean circular error
    mean_error = np.mean(circular_diff)
    
    return r, mean_error


def visualize_manifold_extraction(embeddings, true_angles, method_names):
    """Visualize extracted manifolds from different methods."""
    n_methods = len(embeddings)
    fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 10))
    
    if n_methods == 1:
        axes = axes.reshape(2, 1)
    
    for i, (embedding, method) in enumerate(zip(embeddings, method_names)):
        # Plot 2D embedding colored by true angle
        ax = axes[0, i]
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], 
                           c=true_angles, cmap='hsv', 
                           s=20, alpha=0.7)
        ax.set_title(f'{method} Embedding')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        
        # Add colorbar
        if i == n_methods - 1:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('True head direction (rad)')
        
        # Extract angles and evaluate
        recon_angles = compute_circular_coordinates(embedding)
        r, error = evaluate_reconstruction(true_angles, recon_angles)
        
        # Plot true vs reconstructed angles
        ax = axes[1, i]
        ax.scatter(true_angles, recon_angles, alpha=0.5, s=10)
        ax.plot([0, 2*np.pi], [0, 2*np.pi], 'r--', alpha=0.5)
        ax.set_xlabel('True angle (rad)')
        ax.set_ylabel('Reconstructed angle (rad)')
        ax.set_title(f'r = {r:.3f}, error = {error:.3f} rad')
        ax.set_xlim([0, 2*np.pi])
        ax.set_ylim([0, 2*np.pi])
    
    plt.tight_layout()
    return fig


def main():
    """Main demonstration of circular manifold extraction."""
    print("=" * 70)
    print("CIRCULAR MANIFOLD EXTRACTION FROM HEAD DIRECTION CELLS")
    print("=" * 70)
    
    print("\n1. Generating head direction cell population...")
    
    # Generate synthetic head direction cells
    exp, info = generate_circular_manifold_exp(
        n_neurons=100,
        duration=300,  # 5 minutes
        kappa=4.0,     # Tuning width
        seed=42,
        verbose=True
    )
    
    # Extract neural activity and true head directions
    neural_data = exp.calcium  # Shape: (n_neurons, n_timepoints)
    true_angles = info['head_direction']  # Ground truth angles
    
    print(f"\nGenerated {neural_data.shape[0]} neurons, {neural_data.shape[1]} timepoints")
    print(f"Neural activity shape: {neural_data.shape}")
    
    # Estimate intrinsic dimensionality
    print("\n2. Estimating intrinsic dimensionality of neural population...")
    print("-" * 50)
    
    dim_methods = ['participation_ratio', 'renyi_q1', 'renyi_q2', 
                   'var_explained_90', 'corrected_pr']
    dim_estimates = estimate_dimensionality(neural_data, methods=dim_methods)
    
    print("Dimensionality estimates:")
    for method, estimate in dim_estimates.items():
        print(f"  {method:20s}: {estimate:.2f}")
    
    print("\nNote: Head direction cells should have intrinsic dimensionality ≈ 1")
    print("      (circular manifold), but finite sampling may increase estimates")
    
    # Plot eigenspectrum
    print("\n3. Plotting eigenvalue spectrum...")
    eigen_fig = plot_eigenspectrum(neural_data)
    plt.savefig('circular_manifold_eigenspectrum.png', dpi=150, bbox_inches='tight')
    
    # Apply different dimensionality reduction methods
    print("\n4. Applying dimensionality reduction methods...")
    print("-" * 50)
    
    methods = ['PCA', 'Isomap', 'UMAP']
    embeddings = []
    
    # PCA
    print("- PCA...")
    pca_embedding, explained_var = extract_manifold_pca(neural_data)
    embeddings.append(pca_embedding)
    print(f"  First 2 PCs explain {100*sum(explained_var[:2]):.1f}% of variance")
    
    # Isomap
    print("- Isomap...")
    isomap_embedding = extract_manifold_isomap(neural_data, n_neighbors=50)
    embeddings.append(isomap_embedding)
    
    # UMAP
    print("- UMAP...")
    umap_embedding = extract_manifold_umap(neural_data, n_neighbors=50)
    embeddings.append(umap_embedding)
    
    # Visualize results
    print("\n5. Visualizing extracted manifolds...")
    from driada.utils.visual import plot_embedding_comparison, DEFAULT_DPI
    
    # Prepare embeddings dict and features
    embeddings_dict = {method: emb for method, emb in zip(methods, embeddings)}
    features = {'angle': true_angles}
    feature_names = {'angle': 'True head direction (rad)'}
    
    # Create embedding comparison using visual utility
    fig1 = plot_embedding_comparison(
        embeddings=embeddings_dict,
        features=features,
        feature_names=feature_names,
        with_trajectory=False,
        compute_metrics=True,
        figsize=(15, 5),
        save_path='circular_manifold_extraction.png',
        dpi=DEFAULT_DPI
    )
    
    # Keep the custom reconstruction analysis
    fig2 = visualize_manifold_extraction(embeddings, true_angles, methods)
    plt.savefig('circular_manifold_reconstruction.png', dpi=DEFAULT_DPI, bbox_inches='tight')
    
    # Additional analysis: temporal continuity
    print("\n6. Analyzing temporal continuity of extracted manifolds...")
    from driada.utils.visual import plot_trajectories
    
    # Use only first 1000 timepoints for trajectory visualization
    traj_len = min(1000, embeddings[0].shape[0])
    trajectories_dict = {method: emb[:traj_len] for method, emb in zip(methods, embeddings)}
    
    fig3 = plot_trajectories(
        embeddings=trajectories_dict,
        trajectory_kwargs={
            'arrow_spacing': 50,
            'linewidth': 0.5,
            'alpha': 0.5
        },
        figsize=(15, 5),
        save_path='circular_manifold_trajectories.png',
        dpi=DEFAULT_DPI
    )
    
    # Summary statistics
    print("\n7. Summary of manifold extraction quality:")
    print("-" * 60)
    print(f"{'Method':10s} | {'Correlation':12s} | {'Mean Error':12s} | {'Quality':8s}")
    print("-" * 60)
    
    for i, method in enumerate(methods):
        recon_angles = compute_circular_coordinates(embeddings[i])
        r, error = evaluate_reconstruction(true_angles, recon_angles)
        
        # Quality assessment
        if r > 0.95:
            quality = "Excellent"
        elif r > 0.85:
            quality = "Good"
        elif r > 0.70:
            quality = "Fair"
        else:
            quality = "Poor"
        
        print(f"{method:10s} | {r:12.3f} | {error:9.3f} rad | {quality:8s}")
    
    print("\n" + "="*70)
    print("CONCLUSIONS:")
    print("- Head direction cells have low intrinsic dimensionality (~1-2)")
    print("- Nonlinear methods (Isomap, UMAP) better preserve circular topology")
    print("- PCA captures variance but may distort circular structure")
    print("- Higher n_neighbors helps preserve global structure")
    print("="*70)
    
    print("\nResults saved to:")
    print("- circular_manifold_eigenspectrum.png")
    print("- circular_manifold_extraction.png")
    print("- circular_manifold_reconstruction.png")
    print("- circular_manifold_trajectories.png")
    
    plt.show()


if __name__ == "__main__":
    main()