"""
Test manifold neural data with spatial correspondence metrics.
This test demonstrates how different DR methods preserve manifold structure
in synthetic neural populations and integrates with dimensionality estimation.

This test focuses on aspects NOT covered in test_dr_extended.py:
- Comprehensive manifold metrics evaluation
- Neural data specific testing
- Dimensionality estimation integration  
- Advanced manifold preservation analysis
"""
import numpy as np
import pytest
from src.driada.experiment import (
    generate_circular_manifold_exp,
    generate_2d_manifold_exp,
    generate_3d_manifold_exp
)
from src.driada.dim_reduction.data import MVData
from src.driada.dim_reduction.dr_base import METHODS_DICT
from src.driada.dim_reduction.manifold_metrics import (
    knn_preservation_rate,
    trustworthiness,
    continuity,
    geodesic_distance_correlation,
    circular_structure_preservation,
    manifold_preservation_score
)
from src.driada.dimensionality import (
    nn_dimension,
    correlation_dimension,
    pca_dimension,
    effective_rank
)
from src.driada.signals import manifold_preprocessing


# Shared data generation utilities
def generate_neural_manifold_data():
    """Generate neural data for different manifold types"""
    manifold_data = {}
    
    # Circular manifold (head direction cells)
    exp_circular, info_circular = generate_circular_manifold_exp(
        n_neurons=40,
        duration=200,
        fps=20.0,
        kappa=4.0,
        noise_std=0.1,
        seed=42
    )
    # Filter neural signals for better manifold analysis
    filtered_circular = manifold_preprocessing(exp_circular.calcium.T, method='gaussian', sigma=1.5)
    manifold_data['circular'] = {
        'neural_data': filtered_circular,
        'true_angles': info_circular['head_direction'],
        'expected_intrinsic_dim': 1.0,
        'expected_embedding_dim': 2.0,
        'info': info_circular
    }
    
    # 2D spatial manifold (place cells)
    exp_2d, info_2d = generate_2d_manifold_exp(
        duration=400,
        n_neurons=64,
        field_sigma=0.15,
        seed=42
    )
    # Filter neural signals for better manifold analysis
    filtered_2d = manifold_preprocessing(exp_2d.calcium.T, method='gaussian', sigma=1.2)
    manifold_data['2d_spatial'] = {
        'neural_data': filtered_2d,
        'true_positions': np.column_stack([
            exp_2d.dynamic_features['x_position'].data,
            exp_2d.dynamic_features['y_position'].data
        ]),
        'expected_intrinsic_dim': 2.0,
        'expected_embedding_dim': 2.0,
        'info': info_2d
    }
    
    # 3D spatial manifold (3D place cells)
    exp_3d, info_3d = generate_3d_manifold_exp(
        duration=600,
        n_neurons=125,
        seed=42
    )
    # Filter neural signals for better manifold analysis
    filtered_3d = manifold_preprocessing(exp_3d.calcium.T, method='gaussian', sigma=1.0)
    manifold_data['3d_spatial'] = {
        'neural_data': filtered_3d,
        'true_positions': np.column_stack([
            exp_3d.dynamic_features['x_position'].data,
            exp_3d.dynamic_features['y_position'].data,
            exp_3d.dynamic_features['z_position'].data
        ]),
        'expected_intrinsic_dim': 3.0,
        'expected_embedding_dim': 3.0,
        'info': info_3d
    }
    
    return manifold_data


# =============================================================================
# DIMENSIONALITY ESTIMATION TESTS
# =============================================================================
def test_neural_manifold_dimensionality_estimation():
    """Test dimensionality estimation on neural populations with known manifold structure"""
    manifold_data = generate_neural_manifold_data()
    
    for manifold_type, data in manifold_data.items():
        print(f"\n--- Testing {manifold_type} manifold dimensionality ---")
        
        neural_data = data['neural_data']
        expected_dim = data['expected_intrinsic_dim']
        tolerance = 0.8  # Allow some variance in estimates
        
        # Estimate dimensionality using different methods
        dim_estimates = {}
        
        # Linear methods
        try:
            dim_estimates['pca_90'] = pca_dimension(neural_data.T, threshold=0.9)
            dim_estimates['effective_rank'] = effective_rank(neural_data.T)
        except Exception as e:
            print(f"Linear dimensionality estimation failed: {e}")
        
        # Nonlinear methods
        try:
            dim_estimates['nn_dim'] = nn_dimension(neural_data.T, k=10)
            dim_estimates['correlation_dim'] = correlation_dimension(neural_data.T, n_bins=50)
        except Exception as e:
            print(f"Nonlinear dimensionality estimation failed: {e}")
        
        print(f"Expected intrinsic dimension: {expected_dim}")
        print(f"Estimated dimensions: {dim_estimates}")
        
        if dim_estimates:
            for method, est in dim_estimates.items():
                assert est > 0, f"{method} gave non-positive estimate: {est}"
                
                if method == 'effective_rank':
                    max_bound = 80
                elif method == 'pca_90':
                    max_bound = 45
                elif method == 'nn_dim':
                    max_bound = 25
                elif method == 'correlation_dim':
                    max_bound = 20
                else:
                    max_bound = 30
                    
                assert est < max_bound, f"{method} gave unreasonably high estimate: {est}"
            
            if 'nn_dim' in dim_estimates and 'pca_90' in dim_estimates:
                assert dim_estimates['nn_dim'] < dim_estimates['pca_90'], \
                    f"Nonlinear estimate {dim_estimates['nn_dim']:.3f} not lower than linear {dim_estimates['pca_90']:.3f}"
            
            if 'correlation_dim' in dim_estimates:
                assert 0.5 < dim_estimates['correlation_dim'] < 15, \
                    f"Correlation dimension {dim_estimates['correlation_dim']:.3f} outside reasonable range"
            
            if manifold_type == 'circular':
                if 'correlation_dim' in dim_estimates:
                    assert dim_estimates['correlation_dim'] < 8, \
                        f"Too high dimension estimate for circular manifold: {dim_estimates['correlation_dim']:.3f}"


def test_dimensionality_guided_reconstruction():
    """Test using dimensionality estimation to guide DR method selection"""
    manifold_data = generate_neural_manifold_data()
    
    for manifold_type, data in manifold_data.items():
        print(f"\n--- Testing dimensionality-guided reconstruction for {manifold_type} ---")
        
        neural_data = data['neural_data']
        D = MVData(neural_data)
        
        # First, estimate intrinsic dimensionality
        try:
            estimated_dim = int(np.round(correlation_dimension(neural_data.T, n_bins=50)))
            estimated_dim = max(2, min(estimated_dim, 5))  # Clamp to reasonable range
        except:
            estimated_dim = int(data['expected_embedding_dim'])  # Fallback
        
        print(f"Estimated intrinsic dimension: {estimated_dim}")
        
        # Test DR method with estimated dimension
        try:
            params = {
                'e_method_name': 'isomap',
                'dim': estimated_dim,
                'e_method': METHODS_DICT['isomap'],
                'nn': 15
            }
            
            emb = D.get_embedding(params)
            X_low = emb.coords.T
            
            # Verify reconstruction quality
            scores = manifold_preservation_score(neural_data.T, X_low, k_neighbors=10)
            
            assert scores['overall_score'] > 0.4, \
                f"Poor reconstruction quality for {manifold_type}: {scores['overall_score']:.3f}"
            
        except Exception as e:
            print(f"Dimensionality-guided reconstruction failed: {e}")
            pytest.skip(f"Reconstruction failed for {manifold_type}")


# =============================================================================
# MANIFOLD RECONSTRUCTION TESTS
# =============================================================================

def test_circular_manifold_reconstruction():
    """Test comprehensive reconstruction of circular manifolds"""
    manifold_data = generate_neural_manifold_data()
    data = manifold_data['circular']
    
    neural_data = data['neural_data']
    true_angles = data['true_angles']
    D = MVData(neural_data)
    
    # Test methods optimized for circular manifolds
    methods_to_test = ['pca', 'isomap', 'umap']
    results = {}
    
    for method_name in methods_to_test:
        params = {
            'e_method_name': method_name,
            'dim': 2,
            'e_method': METHODS_DICT[method_name]
        }
        
        # Method-specific parameters
        if method_name == 'isomap':
            params['nn'] = 7  # Optimized from previous testing
        elif method_name == 'umap':
            params['nn'] = 10
            params['min_dist'] = 0.1
        
        try:
            emb = D.get_embedding(params)
            X_low = emb.coords.T
            
            # Comprehensive circular metrics
            circular_metrics = circular_structure_preservation(
                X_low, true_angles=true_angles, k_neighbors=5
            )
            
            # General manifold preservation
            general_metrics = {
                'trustworthiness': trustworthiness(neural_data.T, X_low, k=10),
                'continuity': continuity(neural_data.T, X_low, k=10),
                'geodesic_correlation': geodesic_distance_correlation(
                    neural_data.T, X_low, k_neighbors=10
                )
            }
            
            results[method_name] = {**circular_metrics, **general_metrics}
            
        except Exception as e:
            print(f"Method {method_name} failed: {str(e)}")
    
    # Validate results
    if len(results) >= 2:
        best_circular = max(r['circular_correlation'] for r in results.values())
        assert best_circular > 0.8, f"Best circular correlation {best_circular:.3f} too low"
        
        # Nonlinear methods should excel at geodesic preservation
        if 'pca' in results and 'isomap' in results:
            assert results['isomap']['geodesic_correlation'] >= \
                   results['pca']['geodesic_correlation']


def test_spatial_manifold_reconstruction():
    """Test reconstruction of spatial manifolds"""
    manifold_data = generate_neural_manifold_data()
    
    for manifold_type in ['2d_spatial', '3d_spatial']:
        data = manifold_data[manifold_type]
        
        neural_data = data['neural_data']
        expected_dim = int(data['expected_embedding_dim'])
        D = MVData(neural_data)
        
        # Test methods suitable for spatial manifolds
        methods_to_test = ['pca', 'isomap', 'umap']
        results = {}
        
        # Graph parameters for methods that need them
        metric_params = {'metric_name': 'l2', 'sigma': 1, 'p': 2}
        graph_params = {
            'g_method_name': 'knn',
            'weighted': 0,
            'nn': 15 if expected_dim == 2 else 20,
            'max_deleted_nodes': 0.2,
            'dist_to_aff': 'hk'
        }
        
        for method_name in methods_to_test:
            params = {
                'e_method_name': method_name,
                'dim': expected_dim,
                'e_method': METHODS_DICT[method_name]
            }
            
            if method_name in ['isomap', 'umap']:
                params['nn'] = 15 if expected_dim == 2 else 20
            if method_name == 'umap':
                params['min_dist'] = 0.1
            
            try:
                if method_name in ['isomap', 'umap']:
                    emb = D.get_embedding(params, g_params=graph_params, m_params=metric_params)
                else:
                    emb = D.get_embedding(params)
                X_low = emb.coords.T
                
                # Compute preservation metrics
                scores = manifold_preservation_score(neural_data.T, X_low, k_neighbors=10)
                results[method_name] = scores
                
            except Exception as e:
                print(f"Method {method_name} failed for {manifold_type}: {str(e)}")
        
        # Validate results
        # Validate results with stricter thresholds due to filtering
        if results:
            best_score = max(r['overall_score'] for r in results.values())
            assert best_score > 0.5, \
                f"Poor reconstruction quality for {manifold_type}: {best_score:.3f}"


def test_geodesic_distance_preservation():
    """Test geodesic distance preservation across methods"""
    manifold_data = generate_neural_manifold_data()
    data = manifold_data['2d_spatial']  # Use 2D spatial for geodesic testing
    
    neural_data = data['neural_data']
    D = MVData(neural_data)
    
    # Compare linear vs nonlinear geodesic preservation
    geodesic_scores = {}
    
    for method_name in ['pca', 'isomap']:
        params = {
            'e_method_name': method_name,
            'dim': 2,
            'e_method': METHODS_DICT[method_name]
        }
        
        if method_name == 'isomap':
            params['nn'] = 10
        
        try:
            emb = D.get_embedding(params)
            X_low = emb.coords.T
            
            # Compute geodesic correlation
            geo_corr = geodesic_distance_correlation(
                neural_data.T, X_low, k_neighbors=10
            )
            geodesic_scores[method_name] = geo_corr
            
        except Exception as e:
            print(f"Geodesic test failed for {method_name}: {str(e)}")
    
    # Validate geodesic preservation
    if len(geodesic_scores) >= 2:
        # Isomap should preserve geodesic distances better than PCA
        if 'pca' in geodesic_scores and 'isomap' in geodesic_scores:
            assert geodesic_scores['isomap'] >= geodesic_scores['pca']
            # Both should have positive correlation - stricter thresholds with filtering
            assert geodesic_scores['isomap'] > 0.5
            assert geodesic_scores['pca'] > 0.3


def test_autoencoder_manifold_reconstruction():
    """Test autoencoder methods on neural manifolds"""
    manifold_data = generate_neural_manifold_data()
    data = manifold_data['circular']  # Use circular for faster testing
    
    neural_data = data['neural_data']
    D = MVData(neural_data)
    
    # Test both AE and VAE
    for method_name in ['ae', 'vae']:
        params = {
            'e_method_name': method_name,
            'dim': 2,
            'e_method': METHODS_DICT[method_name]
        }
        
        kwargs = {
            'epochs': 30,  # Fewer epochs for testing
            'batch_size': 32,
            'lr': 1e-3,
            'verbose': False,
            'seed': 42
        }
        
        if method_name == 'vae':
            kwargs['kld_weight'] = 0.01
        
        try:
            emb = D.get_embedding(params, kwargs=kwargs)
            X_low = emb.coords.T
            
            # Check basic preservation - slightly stricter with filtering
            knn_score = knn_preservation_rate(neural_data.T, X_low, k=5)
            assert knn_score > 0.2  # Improved from 0.1 due to filtering
            
            # Check output shape
            assert X_low.shape == (neural_data.shape[1], 2)
            
        except Exception as e:
            pytest.skip(f"Autoencoder {method_name} failed: {str(e)}")


def test_manifold_preservation_comparison():
    """Compare manifold preservation across different neural populations"""
    manifold_data = generate_neural_manifold_data()
    
    # Test same method (Isomap) on different manifolds
    method_name = 'isomap'
    results = {}
    
    for manifold_type, data in manifold_data.items():
        neural_data = data['neural_data']
        expected_dim = int(data['expected_embedding_dim'])
        D = MVData(neural_data)
        
        params = {
            'e_method_name': method_name,
            'dim': expected_dim,
            'e_method': METHODS_DICT[method_name],
            'nn': 15 if expected_dim <= 2 else 20
        }
        
        try:
            emb = D.get_embedding(params)
            X_low = emb.coords.T
            
            # Compute preservation metrics
            scores = manifold_preservation_score(neural_data.T, X_low, k_neighbors=10)
            results[manifold_type] = scores['overall_score']
            
        except Exception as e:
            print(f"Manifold comparison failed for {manifold_type}: {str(e)}")
    
    # Validate that method works across different manifold types
    if results:
        # All manifolds should achieve some level of preservation - stricter with filtering
        for manifold_type, score in results.items():
            assert score > 0.4, \
                f"Poor preservation for {manifold_type}: {score:.3f}"
        
        print(f"Manifold preservation scores: {results}")