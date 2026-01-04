"""Test autoencoder data loading functionality."""

import pytest
import numpy as np
from driada.dim_reduction.data import MVData
from driada.dim_reduction.embedding import Embedding


class TestAutoencoderDataLoading:
    """Test suite for autoencoder data loading refactoring."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 50  # Small for faster testing
        n_features = 10
        data = np.random.randn(n_features, n_samples)
        return data
    
    @pytest.fixture
    def mvdata(self, sample_data):
        """Create MVData object from sample data."""
        return MVData(data=sample_data, labels=None)
    
    @pytest.fixture
    def embedding(self, mvdata):
        """Create embedding object."""
        from driada.dim_reduction.dr_base import METHODS_DICT
        
        params = {"e_method_name": "ae", "dim": 2}
        params["e_method"] = METHODS_DICT["ae"]
        
        return Embedding(
            init_data=mvdata.data,
            init_distmat=None,
            labels=mvdata.labels,
            params=params
        )
    
    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch not installed"),
        reason="PyTorch required for this test"
    )
    def test_prepare_data_loaders(self, embedding):
        """Test _prepare_data_loaders method functionality."""
        batch_size = 16
        train_size = 0.8
        seed = 42
        
        # Call the method
        train_loader, test_loader, device = embedding._prepare_data_loaders(
            batch_size=batch_size,
            train_size=train_size,
            seed=seed
        )
        
        # Basic checks
        assert train_loader is not None
        assert test_loader is not None
        assert device is not None
        
        # Check data split
        n_samples = embedding.init_data.shape[1]
        expected_train = int(n_samples * train_size)
        expected_test = n_samples - expected_train
        
        # Count actual samples
        train_count = sum(len(batch[0]) for batch in train_loader)
        test_count = sum(len(batch[0]) for batch in test_loader)
        
        assert train_count == expected_train
        assert test_count == expected_test
    
    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch not installed"),
        reason="PyTorch required for this test"
    )
    def test_ae_embedding_uses_helper(self, embedding):
        """Test that create_ae_embedding_ actually works with the refactored code."""
        # Create a minimal AE embedding with tiny parameters
        try:
            embedding.create_ae_embedding_(
                epochs=1,  # Just 1 epoch
                batch_size=16,
                train_size=0.8,
                seed=42,
                lr=1e-3,
                inter_dim=5,  # Very small
                verbose=False,
                log_every=1
            )
            
            # Check that embedding was created
            assert hasattr(embedding, 'coords')
            assert embedding.coords is not None
            assert hasattr(embedding, 'nnmodel')
            assert embedding.nnmodel is not None
            
        except ImportError:
            pytest.skip("PyTorch not available")
    
    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch not installed"),
        reason="PyTorch required for this test"
    )
    def test_vae_embedding_uses_helper(self, embedding):
        """Test that create_vae_embedding_ actually works with the refactored code."""
        # Update method name for VAE
        embedding.e_method_name = "vae"
        
        # Create a minimal VAE embedding with tiny parameters
        try:
            embedding.create_vae_embedding_(
                epochs=1,  # Just 1 epoch
                batch_size=16,
                train_size=0.8,
                seed=42,
                lr=1e-3,
                inter_dim=5,  # Very small
                verbose=False,
                log_every=1
            )
            
            # Check that embedding was created
            assert hasattr(embedding, 'coords')
            assert embedding.coords is not None
            assert hasattr(embedding, 'nnmodel')
            assert embedding.nnmodel is not None
            
        except ImportError:
            pytest.skip("PyTorch not available")
