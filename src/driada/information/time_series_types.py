"""
Advanced type detection for time series data.

This module provides comprehensive detection of time series types including:
- Discrete vs Continuous classification
- Circular/periodic signal detection
- Probabilistic type inference with confidence scores
- Detection of various data patterns (binary, categorical, count, etc.)
"""

import numpy as np
from typing import Dict, Tuple, Optional, Literal, Union
from dataclasses import dataclass
from scipy import stats
from scipy.signal import find_peaks, correlate
import warnings


@dataclass
class TimeSeriesType:
    """Result of time series type detection."""
    primary_type: Literal['discrete', 'continuous', 'ambiguous']
    subtype: Optional[Literal['binary', 'categorical', 'count', 'timeline', 'linear', 'circular']]
    confidence: float
    is_circular: bool
    circular_period: Optional[float]
    periodicity: Optional[float]  # For autocorrelation detection
    metadata: Dict[str, float]
    
    @property
    def is_discrete(self) -> bool:
        return self.primary_type == 'discrete'
    
    @property
    def is_continuous(self) -> bool:
        return self.primary_type == 'continuous'
    
    @property
    def is_ambiguous(self) -> bool:
        return self.primary_type == 'ambiguous'
    
    @property
    def is_periodic(self) -> bool:
        return self.periodicity is not None


def analyze_time_series_type(
    data: np.ndarray,
    name: Optional[str] = None,
    confidence_threshold: float = 0.7,
    min_samples: int = 30,
    verbose: bool = False
) -> TimeSeriesType:
    """
    Analyze and detect the type of a time series using comprehensive statistical analysis.
    
    Parameters
    ----------
    data : np.ndarray
        1D array of time series values
    name : str, optional
        Name of the time series (used for context-aware detection)
    confidence_threshold : float
        Minimum confidence for definitive classification
    min_samples : int
        Minimum samples for reliable detection
    verbose : bool
        Print detection details
        
    Returns
    -------
    TimeSeriesType
        Comprehensive type detection results
    """
    data = np.asarray(data).ravel()
    
    if len(data) < min_samples:
        warnings.warn(f"Only {len(data)} samples available. Detection may be unreliable.")
    
    # Extract comprehensive statistical properties
    properties = _extract_statistical_properties(data)
    
    # Check for periodicity patterns (including circular)
    periodicity_result = _detect_periodicity(data, properties, name)
    circular_result = _detect_circular(data, properties, name)
    
    # If strongly circular, that determines the type
    if circular_result['is_circular'] and circular_result['confidence'] > confidence_threshold:
        return TimeSeriesType(
            primary_type='continuous',
            subtype='circular',
            confidence=circular_result['confidence'],
            is_circular=True,
            circular_period=circular_result['period'],
            periodicity=circular_result['period'],
            metadata=properties
        )
    
    # Detect primary type (discrete vs continuous)
    primary_result = _detect_primary_type(data, properties)
    
    # Detect subtype
    if primary_result['type'] == 'discrete':
        subtype_result = _detect_discrete_subtype(data, properties)
    else:
        subtype_result = _detect_continuous_subtype(data, properties, circular_result, periodicity_result)
    
    if verbose:
        _print_detection_details(data, properties, primary_result, subtype_result, circular_result, periodicity_result)
    
    # Add detection scores to metadata
    metadata = properties.copy()
    metadata['discrete_score'] = primary_result.get('discrete_score', None)
    metadata['continuous_score'] = primary_result.get('continuous_score', None)
    
    return TimeSeriesType(
        primary_type=primary_result['type'],
        subtype=subtype_result.get('subtype') if primary_result['type'] != 'ambiguous' else None,
        confidence=min(primary_result['confidence'], subtype_result.get('confidence', 1.0)),
        is_circular=circular_result['is_circular'],
        circular_period=circular_result.get('period'),
        periodicity=periodicity_result.get('period'),
        metadata=metadata
    )


def _extract_statistical_properties(data: np.ndarray) -> Dict[str, float]:
    """Extract comprehensive statistical properties from time series."""
    n = len(data)
    unique_vals = np.unique(data)
    n_unique = len(unique_vals)
    
    # Basic statistics
    features = {
        'n_samples': n,
        'n_unique': n_unique,
        'uniqueness_ratio': n_unique / n,
        'mean': np.mean(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'range': np.ptp(data),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data),
    }
    
    # Entropy measures
    if n_unique > 1:
        hist, _ = np.histogram(data, bins=min(n_unique, int(np.sqrt(n))))
        hist = hist[hist > 0]  # Remove zero bins
        features['entropy'] = stats.entropy(hist)
        features['normalized_entropy'] = features['entropy'] / np.log(len(hist))
    else:
        features['entropy'] = 0
        features['normalized_entropy'] = 0
    
    # Gap statistics (spacing between consecutive unique values)
    if n_unique > 1:
        sorted_unique = np.sort(unique_vals)
        gaps = np.diff(sorted_unique)
        features['mean_gap'] = np.mean(gaps)
        features['std_gap'] = np.std(gaps)
        features['cv_gap'] = features['std_gap'] / (features['mean_gap'] + 1e-10)
        features['max_gap_ratio'] = np.max(gaps) / (np.min(gaps) + 1e-10)
    else:
        features['mean_gap'] = 0
        features['std_gap'] = 0
        features['cv_gap'] = 0
        features['max_gap_ratio'] = 1
    
    # Distribution tests
    if n >= 8:  # Minimum for statistical tests
        # Test for uniformity
        _, features['uniform_pvalue'] = stats.kstest(
            data, lambda x: (x - features['min']) / (features['range'] + 1e-10)
        )
        # Test for normality
        _, features['normal_pvalue'] = stats.normaltest(data)
    else:
        features['uniform_pvalue'] = 0.5
        features['normal_pvalue'] = 0.5
    
    # Autocorrelation for periodicity detection
    if n >= 20:
        autocorr = correlate(data - features['mean'], data - features['mean'], mode='same')
        autocorr = autocorr / autocorr[n//2]  # Normalize
        features['max_autocorr'] = np.max(np.abs(autocorr[n//2+1:]))
    else:
        features['max_autocorr'] = 0
    
    # Integer check
    features['fraction_integers'] = np.mean(np.abs(data - np.round(data)) < 1e-10)
    
    return features


def _detect_periodicity(
    data: np.ndarray,
    properties: Dict[str, float],
    name: Optional[str] = None
) -> Dict[str, Union[float, None]]:
    """Detect general periodicity in time series (not necessarily circular)."""
    result = {'period': None, 'confidence': 0.0}
    
    if len(data) < 20:
        return result
    
    # Autocorrelation analysis
    if properties['max_autocorr'] > 0.7:
        # Find the lag with maximum autocorrelation
        data_centered = data - np.mean(data)
        autocorr = correlate(data_centered, data_centered, mode='same')
        autocorr = autocorr / autocorr[len(data)//2]
        
        # Find peaks in autocorrelation
        peaks, peak_props = find_peaks(autocorr[len(data)//2+1:], height=0.5)
        if len(peaks) > 0:
            # First significant peak gives the period
            result['period'] = peaks[0] + 1
            result['confidence'] = peak_props['peak_heights'][0]
    
    # Fourier analysis for strong periodicity
    if len(data) >= 50:
        fft = np.fft.fft(data - np.mean(data))
        power = np.abs(fft[:len(data)//2])**2
        freqs = np.fft.fftfreq(len(data))[:len(data)//2]
        
        # Find dominant frequency (excluding DC component)
        peak_idx = np.argmax(power[1:]) + 1
        if power[peak_idx] > 10 * np.median(power[1:]):  # Strong peak
            dominant_freq = freqs[peak_idx]
            if dominant_freq > 0:
                result['period'] = 1.0 / dominant_freq
                result['confidence'] = max(result['confidence'], 0.8)
    
    return result


def _detect_circular(
    data: np.ndarray, 
    properties: Dict[str, float],
    name: Optional[str] = None
) -> Dict[str, Union[bool, float, None]]:
    """Detect if the time series represents circular/angular data."""
    result = {'is_circular': False, 'confidence': 0.0, 'period': None}
    
    # Context-based detection from name
    if name:
        circular_keywords = [
            'angle', 'phase', 'direction', 'heading', 'azimuth', 
            'orientation', 'rotation', 'bearing', 'circular', 'theta',
            'phi', 'degrees', 'radians', 'compass'
        ]
        if any(keyword in name.lower() for keyword in circular_keywords):
            result['confidence'] += 0.3
    
    # Range-based detection (only for continuous-looking data)
    # Binary data [0,1] should NOT be detected as circular!
    if properties['uniqueness_ratio'] > 0.1 and properties['n_unique'] > 2:  # Not binary or too discrete
        common_circular_ranges = [
            (0, 2*np.pi, 2*np.pi),      # [0, 2π]
            (-np.pi, np.pi, 2*np.pi),   # [-π, π]
            (0, 360, 360),              # [0, 360]
            (-180, 180, 360),           # [-180, 180]
        ]
        
        for range_min, range_max, period in common_circular_ranges:
            if (np.abs(properties['min'] - range_min) < 0.1 * period and 
                np.abs(properties['max'] - range_max) < 0.1 * period):
                result['confidence'] += 0.4
                result['period'] = period
                break
    
    # Wraparound detection
    if len(data) > 1:
        diffs = np.diff(data)
        large_jumps = np.abs(diffs) > 0.8 * properties['range']
        if np.any(large_jumps):
            # Check if jumps are consistent with wraparound
            jump_indices = np.where(large_jumps)[0]
            for idx in jump_indices:
                if (data[idx] > properties['mean'] and data[idx+1] < properties['mean']) or \
                   (data[idx] < properties['mean'] and data[idx+1] > properties['mean']):
                    result['confidence'] += 0.2
                    break
    
    # Periodicity detection (but not for binary/categorical data)
    if properties['max_autocorr'] > 0.7 and properties['n_unique'] > 5:
        result['confidence'] += 0.3
    
    # Von Mises distribution test (for circular data)
    # Skip for discrete data with few unique values
    if properties['range'] > 0 and len(data) >= 50 and properties['n_unique'] > 10:
        # Normalize to [0, 2π]
        normalized = 2 * np.pi * (data - properties['min']) / properties['range']
        try:
            # Fit Von Mises distribution
            kappa, loc, scale = stats.vonmises.fit(normalized)
            if kappa > 0.5:  # Significant concentration parameter
                result['confidence'] += 0.2
        except:
            pass
    
    result['is_circular'] = result['confidence'] >= 0.6  # Higher threshold to avoid false positives
    return result


def _detect_primary_type(data: np.ndarray, properties: Dict[str, float]) -> Dict[str, Union[str, float]]:
    """Detect primary type: discrete vs continuous."""
    # Strong discrete indicators (binary, small categorical)
    if properties['n_unique'] == 2:
        # Binary data - check if truly binary (allowing small numerical errors)
        if properties['fraction_integers'] > 0.98:
            return {'type': 'discrete', 'confidence': 1.0}
    
    # Strong categorical indicator: few unique values relative to sample size
    if properties['uniqueness_ratio'] < 0.05 and properties['n_unique'] <= 20:
        return {'type': 'discrete', 'confidence': 0.95}
    
    # Integer data with reasonable number of categories
    if properties['fraction_integers'] > 0.99 and properties['uniqueness_ratio'] < 0.3:
        return {'type': 'discrete', 'confidence': 0.9}
    
    # Monotonic integer data (count data) - always discrete
    if properties['fraction_integers'] > 0.99 and len(data) > 1:
        # Check if monotonically non-decreasing
        diffs = np.diff(data)
        if np.all(diffs >= 0):
            return {'type': 'discrete', 'confidence': 0.95}
    
    # Timeline data: values with very regular gaps (even if high uniqueness)
    # Can be non-integer (e.g., 0, 0.5, 1, 1.5, ...)
    if properties['cv_gap'] < 0.05 and properties['uniqueness_ratio'] > 0.8:
        # Very regular gaps with high uniqueness indicate discrete timeline
        return {'type': 'discrete', 'confidence': 0.85}
    
    # Now do weighted scoring for ambiguous cases
    discrete_score = 0.0
    continuous_score = 0.0
    
    # Uniqueness ratio analysis
    ur = properties['uniqueness_ratio']
    if ur < 0.1:
        discrete_score += 0.3
    elif ur > 0.5:
        continuous_score += 0.3
    
    # Normalized entropy analysis
    ne = properties['normalized_entropy']
    if ne < 0.3:
        discrete_score += 0.2
    elif ne > 0.7:
        continuous_score += 0.2
    
    # Integer fraction
    if properties['fraction_integers'] > 0.95:
        discrete_score += 0.2
    elif properties['fraction_integers'] < 0.1:
        continuous_score += 0.2
    
    # Gap analysis
    if properties['cv_gap'] < 0.1:  # Regular gaps
        discrete_score += 0.1
    elif properties['cv_gap'] > 1.0:  # Irregular gaps
        continuous_score += 0.1
    
    # Number of unique values
    if properties['n_unique'] <= 30:
        discrete_score += 0.1
    elif properties['n_unique'] > 0.7 * properties['n_samples']:
        continuous_score += 0.1
    
    # Distribution tests
    if properties['normal_pvalue'] > 0.05:
        continuous_score += 0.1
    if properties['uniform_pvalue'] > 0.05:
        continuous_score += 0.1
    
    # Normalize scores
    total = discrete_score + continuous_score
    if total > 0:
        discrete_score /= total
        continuous_score /= total
    else:
        discrete_score = continuous_score = 0.5
    
    # Store scores in result for transparency
    # Check if the decision is ambiguous
    if abs(discrete_score - continuous_score) < 0.2:  # Too close to call
        return {
            'type': 'ambiguous', 
            'confidence': max(discrete_score, continuous_score),
            'discrete_score': discrete_score,
            'continuous_score': continuous_score
        }
    elif discrete_score > continuous_score:
        return {
            'type': 'discrete', 
            'confidence': discrete_score,
            'discrete_score': discrete_score,
            'continuous_score': continuous_score
        }
    else:
        return {
            'type': 'continuous', 
            'confidence': continuous_score,
            'discrete_score': discrete_score,
            'continuous_score': continuous_score
        }


def _detect_discrete_subtype(data: np.ndarray, properties: Dict[str, float]) -> Dict[str, Union[str, float]]:
    """Detect discrete subtype."""
    n_unique = properties['n_unique']
    unique_vals = np.unique(data)
    
    if n_unique == 2:
        return {'subtype': 'binary', 'confidence': 1.0}
    
    # Check if count data FIRST: monotonically increasing integers
    if properties['fraction_integers'] > 0.99 and len(data) > 1:
        # Check if data is monotonically non-decreasing (count data always goes up)
        diffs = np.diff(data)
        if np.all(diffs >= 0):  # All differences are non-negative (monotonic)
            return {'subtype': 'count', 'confidence': 0.95}
    
    # Check for timeline data (regularly spaced values with high uniqueness)
    # Timeline data should have n_unique ≈ n_samples (each timestamp mostly unique)
    if properties['uniqueness_ratio'] > 0.8 and n_unique >= 10:
        sorted_vals = np.sort(unique_vals)
        gaps = np.diff(sorted_vals)
        if len(gaps) > 0:
            # Check if gaps are consistent
            gap_cv = np.std(gaps) / (np.mean(gaps) + 1e-10)  # Coefficient of variation
            if gap_cv < 0.05:  # Very regular spacing
                # This is timeline data (e.g., 0, 0.5, 1, 1.5, ... or 0, 10, 20, 30, ...)
                return {'subtype': 'timeline', 'confidence': 0.95}
    
    # Default to categorical
    return {'subtype': 'categorical', 'confidence': 0.8}


def _detect_continuous_subtype(
    data: np.ndarray, 
    properties: Dict[str, float],
    circular_result: Dict[str, Union[bool, float, None]],
    periodicity_result: Dict[str, Union[float, None]]
) -> Dict[str, Union[str, float]]:
    """Detect continuous subtype."""
    # Check if circular (angles, phases, directions)
    if circular_result['is_circular']:
        return {'subtype': 'circular', 'confidence': circular_result['confidence']}
    
    # Default to linear for all other continuous data
    return {'subtype': 'linear', 'confidence': 0.8}


def _print_detection_details(
    data: np.ndarray,
    properties: Dict[str, float],
    primary_result: Dict[str, Union[str, float]],
    subtype_result: Dict[str, Union[str, float]],
    circular_result: Dict[str, Union[bool, float, None]],
    periodicity_result: Dict[str, Union[float, None]]
):
    """Print detailed detection information."""
    print("\n=== Time Series Type Detection ===")
    print(f"Samples: {len(data)}")
    print(f"Unique values: {properties['n_unique']} ({properties['uniqueness_ratio']:.2%})")
    print(f"Range: [{properties['min']:.3f}, {properties['max']:.3f}]")
    print(f"Normalized entropy: {properties['normalized_entropy']:.3f}")
    print(f"Fraction integers: {properties['fraction_integers']:.2%}")
    
    print(f"\nPrimary type: {primary_result['type']} (confidence: {primary_result['confidence']:.2f})")
    print(f"Subtype: {subtype_result['subtype']} (confidence: {subtype_result['confidence']:.2f})")
    
    if circular_result['is_circular']:
        print(f"Circular detected! Period: {circular_result['period']}")
    if periodicity_result.get('period') is not None:
        print(f"Periodicity detected! Period: {periodicity_result['period']:.1f} samples")
    print("=" * 30)


# Simple detection function for backward compatibility
def is_discrete_time_series(
    ts: np.ndarray,
    return_confidence: bool = False
) -> Union[bool, Tuple[bool, float]]:
    """
    Simple function that returns whether time series is discrete (True) or continuous (False).
    
    Parameters
    ----------
    ts : np.ndarray
        Time series data
    return_confidence : bool
        If True, also return confidence score
        
    Returns
    -------
    bool or tuple
        True if discrete, False if continuous.
        If return_confidence=True, returns (is_discrete, confidence)
    """
    result = analyze_time_series_type(ts, confidence_threshold=0.5)
    
    if return_confidence:
        return result.is_discrete, result.confidence
    return result.is_discrete


# Legacy alias for backward compatibility
detect_ts_type = is_discrete_time_series