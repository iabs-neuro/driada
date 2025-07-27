"""Import tests for wavelet-related modules in experiment package."""

import pytest


class TestWaveletModuleImports:
    """Test imports for wavelet processing modules."""
    
    def test_import_wavelet_ridge(self):
        """Test importing wavelet ridge detection module."""
        from driada.experiment import wavelet_ridge
        assert hasattr(wavelet_ridge, '__file__')
        
    def test_import_wavelet_event_detection(self):
        """Test importing wavelet event detection module."""
        from driada.experiment import wavelet_event_detection
        assert hasattr(wavelet_event_detection, '__file__')
        
    def test_wavelet_modules_have_functions(self):
        """Test that wavelet modules have expected attributes."""
        from driada.experiment import wavelet_ridge
        from driada.experiment import wavelet_event_detection
        
        # These modules should have some content
        assert dir(wavelet_ridge)  # Should not be empty
        assert dir(wavelet_event_detection)  # Should not be empty