"""Import tests for wavelet-related modules in experiment package."""

import os
import pytest
import numpy as np


class TestWaveletModuleImports:
    """Test imports for wavelet processing modules."""

    def test_import_wavelet_ridge(self):
        """Test importing wavelet ridge detection module."""
        from driada.experiment import wavelet_ridge

        assert hasattr(wavelet_ridge, "__file__")

    def test_import_wavelet_event_detection(self):
        """Test importing wavelet event detection module."""
        from driada.experiment import wavelet_event_detection

        assert hasattr(wavelet_event_detection, "__file__")

    def test_wavelet_modules_have_functions(self):
        """Test that wavelet modules have expected attributes."""
        from driada.experiment import wavelet_ridge
        from driada.experiment import wavelet_event_detection

        # These modules should have some content
        assert dir(wavelet_ridge)  # Should not be empty
        assert dir(wavelet_event_detection)  # Should not be empty


class TestRidgeClass:
    """Test Ridge class functionality in both numba and python modes."""
    
    def test_ridge_class_python_mode(self):
        """Test Ridge class functionality in python mode."""
        # Force python mode
        os.environ['DRIADA_DISABLE_NUMBA'] = '1'
        
        # Reload modules
        import importlib
        from driada.utils import jit as jit_module
        from driada.experiment import wavelet_ridge as ridge_module
        
        importlib.reload(jit_module)
        importlib.reload(ridge_module)
        
        from driada.experiment.wavelet_ridge import Ridge
        
        # Create a ridge
        ridge = Ridge(start_index=100, ampl=1.5, start_scale=10.0, wvt_time=5.0)
        
        # Test initial state
        assert ridge.indices == [100]
        assert ridge.ampls == [1.5]
        assert ridge.birth_scale == 10.0
        assert ridge.scales == [10.0]
        assert ridge.wvt_times == [5.0]
        assert ridge.terminated == False
        
        # Test extending ridge
        ridge.extend(index=110, ampl=2.0, scale=12.0, wvt_time=6.0)
        assert ridge.indices == [100, 110]
        assert ridge.ampls == [1.5, 2.0]
        assert ridge.scales == [10.0, 12.0]
        assert ridge.wvt_times == [5.0, 6.0]
        
        # Test tip
        assert ridge.tip() == 110
        
        # Test termination
        ridge.terminate()
        assert ridge.terminated == True
        assert ridge.end_scale == 12.0
        assert ridge.length == 2
        assert ridge.max_scale == 12.0  # Scale at max amplitude
        assert ridge.max_ampl == 2.0
        assert ridge.start == 100
        assert ridge.end == 110
        assert ridge.duration == 10
        
        # Test cannot extend after termination
        with pytest.raises(ValueError):
            ridge.extend(120, 1.0, 14.0, 7.0)
        
        # Reset environment
        os.environ.pop('DRIADA_DISABLE_NUMBA', None)
    
    def test_ridge_class_numba_mode(self):
        """Test Ridge class functionality in numba mode."""
        # Force numba mode
        os.environ['DRIADA_DISABLE_NUMBA'] = '0'
        
        # Reload modules
        import importlib
        from driada.utils import jit as jit_module
        from driada.experiment import wavelet_ridge as ridge_module
        
        importlib.reload(jit_module)
        importlib.reload(ridge_module)
        
        from driada.experiment.wavelet_ridge import Ridge
        from driada.utils.jit import is_jit_enabled
        
        # Skip if numba not available
        if not is_jit_enabled():
            pytest.skip("Numba not available")
        
        # Create a ridge (numba jitclass doesn't support keyword args)
        ridge = Ridge(100, 1.5, 10.0, 5.0)
        
        # Test extending ridge
        ridge.extend(110, 2.0, 12.0, 6.0)
        ridge.extend(115, 1.8, 14.0, 7.0)
        
        # Test tip
        assert ridge.tip() == 115
        
        # Test termination
        ridge.terminate()
        assert ridge.terminated == True
        assert ridge.end_scale == 14.0
        assert ridge.length == 3
        assert ridge.max_ampl == 2.0
        assert ridge.start == 100
        assert ridge.end == 115
        assert ridge.duration == 15
        
    def test_ridge_container(self):
        """Test RidgeInfoContainer functionality."""
        from driada.experiment.wavelet_ridge import RidgeInfoContainer
        
        # Create container with sample data
        indices = [100, 110, 120, 130]
        ampls = [1.0, 1.5, 2.0, 1.8]
        scales = [10.0, 12.0, 14.0, 16.0]
        wvt_times = [5.0, 6.0, 7.0, 8.0]
        
        container = RidgeInfoContainer(indices, ampls, scales, wvt_times)
        
        # Test attributes
        assert np.array_equal(container.indices, indices)
        assert np.array_equal(container.ampls, ampls)
        assert np.array_equal(container.scales, scales)
        assert np.array_equal(container.wvt_times, wvt_times)
        
        # Test computed properties
        assert container.birth_scale == 10.0
        assert container.end_scale == 16.0
        assert container.length == 4
        assert container.max_scale == 14.0  # Scale at max amplitude (2.0)
        assert container.max_ampl == 2.0
        assert container.start == 100
        assert container.end == 130
        assert container.duration == 30
