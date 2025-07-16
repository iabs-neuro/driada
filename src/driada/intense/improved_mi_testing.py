"""
Improved statistical testing approaches for MI distributions in INTENSE.

This module implements better alternatives to the current norm/gamma distribution
fitting approach, addressing the fundamental issue of using poorly-fitting distributions.
"""

import numpy as np
from scipy import stats
from scipy.stats import rankdata
from typing import Tuple, Optional, Dict, Union
import warnings


def empirical_p_value(observed_value: float, 
                     null_distribution: np.ndarray,
                     method: str = 'conservative') -> float:
    """
    Calculate empirical p-value without assuming any distribution.
    
    This is the most robust approach - it makes no distributional assumptions
    and directly uses the empirical distribution of shuffled values.
    
    Parameters
    ----------
    observed_value : float
        The observed MI value to test.
    null_distribution : np.ndarray
        Array of MI values from shuffled data (null hypothesis).
    method : str, optional
        How to handle ties. Options:
        - 'conservative': (rank + 1) / (n + 1) - never gives p=0
        - 'standard': rank / n - can give p=0
        - 'mid-rank': Uses mid-rank for ties
        Default: 'conservative'
        
    Returns
    -------
    p_value : float
        Empirical p-value.
        
    Notes
    -----
    This is the most defensible approach statistically as it requires
    no assumptions about the distribution of MI values.
    """
    n = len(null_distribution)
    
    if method == 'conservative':
        # Add 1 to both numerator and denominator to avoid p=0
        # This is the (r+1)/(n+1) formula recommended by many statisticians
        rank = np.sum(null_distribution >= observed_value)
        p_value = (rank + 1) / (n + 1)
    elif method == 'standard':
        # Simple proportion of values >= observed
        rank = np.sum(null_distribution >= observed_value)
        p_value = rank / n
    elif method == 'mid-rank':
        # Handle ties by using mid-rank
        all_values = np.concatenate([null_distribution, [observed_value]])
        ranks = rankdata(all_values, method='average')
        obs_rank = ranks[-1]
        p_value = 1 - (obs_rank - 1) / n
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return p_value


def adaptive_distribution_test(observed_value: float,
                              null_distribution: np.ndarray,
                              candidate_distributions: Optional[Dict] = None,
                              selection_criterion: str = 'aic') -> Tuple[float, str, Dict]:
    """
    Adaptively select the best-fitting distribution and compute p-value.
    
    This approach fits multiple candidate distributions and selects the best
    one based on goodness-of-fit criteria, then uses it for p-value calculation.
    
    Parameters
    ----------
    observed_value : float
        The observed MI value to test.
    null_distribution : np.ndarray
        Array of MI values from shuffled data.
    candidate_distributions : dict, optional
        Dictionary mapping distribution names to scipy.stats distributions.
        If None, uses a default set of candidates.
    selection_criterion : str, optional
        Criterion for selecting best distribution: 'aic', 'bic', or 'ks'.
        Default: 'aic'
        
    Returns
    -------
    p_value : float
        P-value from the best-fitting distribution.
    best_dist : str
        Name of the selected distribution.
    fit_info : dict
        Information about the fit quality for all distributions.
    """
    if candidate_distributions is None:
        candidate_distributions = {
            'gamma': stats.gamma,
            'lognorm': stats.lognorm,
            'weibull_min': stats.weibull_min,
            'exponweib': stats.exponweib,
            'beta': stats.beta,  # After transformation to [0,1]
            'norm': stats.norm  # Include for comparison
        }
    
    fit_results = {}
    n = len(null_distribution)
    
    for dist_name, dist_class in candidate_distributions.items():
        try:
            # Special handling for beta distribution
            if dist_name == 'beta':
                # Transform to [0, 1] range
                data_min = np.min(null_distribution)
                data_max = np.max(null_distribution)
                if data_min == data_max:
                    continue
                transformed_data = (null_distribution - data_min) / (data_max - data_min)
                params = dist_class.fit(transformed_data)
                
                # Transform observed value
                transformed_obs = (observed_value - data_min) / (data_max - data_min)
                # Ensure within bounds
                transformed_obs = np.clip(transformed_obs, 0, 1)
                
                # Calculate metrics on transformed data
                fitted_dist = dist_class(*params)
                log_likelihood = np.sum(dist_class.logpdf(transformed_data, *params))
                ks_stat, ks_p = stats.kstest(transformed_data, fitted_dist.cdf)
                p_value = fitted_dist.sf(transformed_obs)
                
            else:
                # Standard fitting
                if dist_name in ['gamma', 'lognorm', 'weibull_min']:
                    # These require positive values
                    if np.any(null_distribution <= 0):
                        # Add small constant to make positive
                        shifted_data = null_distribution + 1e-10
                        shifted_obs = observed_value + 1e-10
                        params = dist_class.fit(shifted_data, floc=0)
                        fitted_dist = dist_class(*params)
                        log_likelihood = np.sum(dist_class.logpdf(shifted_data, *params))
                        ks_stat, ks_p = stats.kstest(shifted_data, fitted_dist.cdf)
                        p_value = fitted_dist.sf(shifted_obs)
                    else:
                        params = dist_class.fit(null_distribution, floc=0)
                        fitted_dist = dist_class(*params)
                        log_likelihood = np.sum(dist_class.logpdf(null_distribution, *params))
                        ks_stat, ks_p = stats.kstest(null_distribution, fitted_dist.cdf)
                        p_value = fitted_dist.sf(observed_value)
                else:
                    params = dist_class.fit(null_distribution)
                    fitted_dist = dist_class(*params)
                    log_likelihood = np.sum(dist_class.logpdf(null_distribution, *params))
                    ks_stat, ks_p = stats.kstest(null_distribution, fitted_dist.cdf)
                    p_value = fitted_dist.sf(observed_value)
            
            # Calculate information criteria
            k = len(params)  # Number of parameters
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood
            
            fit_results[dist_name] = {
                'params': params,
                'p_value': p_value,
                'aic': aic,
                'bic': bic,
                'ks_stat': ks_stat,
                'ks_p': ks_p,
                'log_likelihood': log_likelihood
            }
            
        except Exception as e:
            warnings.warn(f"Failed to fit {dist_name}: {e}")
            continue
    
    if not fit_results:
        raise ValueError("All distribution fits failed")
    
    # Select best distribution
    if selection_criterion == 'aic':
        best_dist = min(fit_results.keys(), key=lambda d: fit_results[d]['aic'])
    elif selection_criterion == 'bic':
        best_dist = min(fit_results.keys(), key=lambda d: fit_results[d]['bic'])
    elif selection_criterion == 'ks':
        best_dist = min(fit_results.keys(), key=lambda d: fit_results[d]['ks_stat'])
    else:
        raise ValueError(f"Unknown selection criterion: {selection_criterion}")
    
    return fit_results[best_dist]['p_value'], best_dist, fit_results


def robust_parametric_test(observed_value: float,
                          null_distribution: np.ndarray,
                          min_samples_for_parametric: int = 100) -> Tuple[float, str]:
    """
    Use parametric test only when data strongly supports it, otherwise empirical.
    
    This hybrid approach uses parametric testing when:
    1. We have enough samples
    2. The distribution clearly fits well
    Otherwise, it falls back to empirical p-values.
    
    Parameters
    ----------
    observed_value : float
        The observed MI value to test.
    null_distribution : np.ndarray
        Array of MI values from shuffled data.
    min_samples_for_parametric : int, optional
        Minimum number of samples required to attempt parametric fitting.
        Default: 100
        
    Returns
    -------
    p_value : float
        P-value from the selected method.
    method_used : str
        Description of the method used.
    """
    n = len(null_distribution)
    
    # Check if we have enough samples
    if n < min_samples_for_parametric:
        p_value = empirical_p_value(observed_value, null_distribution, method='conservative')
        return p_value, f"empirical (n={n} too small)"
    
    # Try to fit gamma distribution (most theoretically justified)
    try:
        # Ensure positive values
        if np.any(null_distribution <= 0):
            null_dist_positive = null_distribution + 1e-10
            obs_positive = observed_value + 1e-10
        else:
            null_dist_positive = null_distribution
            obs_positive = observed_value
        
        # Fit gamma
        params = stats.gamma.fit(null_dist_positive, floc=0)
        fitted_gamma = stats.gamma(*params)
        
        # Check goodness of fit
        ks_stat, ks_p = stats.kstest(null_dist_positive, fitted_gamma.cdf)
        
        # Use parametric only if fit is good
        if ks_p > 0.1:  # Reasonable fit
            p_value = fitted_gamma.sf(obs_positive)
            return p_value, f"gamma (KS p={ks_p:.3f})"
        else:
            # Poor fit, use empirical
            p_value = empirical_p_value(observed_value, null_distribution, method='conservative')
            return p_value, f"empirical (gamma KS p={ks_p:.3f} too low)"
            
    except Exception:
        # If fitting fails, use empirical
        p_value = empirical_p_value(observed_value, null_distribution, method='conservative')
        return p_value, "empirical (gamma fit failed)"


def extreme_value_correction(observed_value: float,
                           null_distribution: np.ndarray,
                           threshold_percentile: float = 95) -> Tuple[float, str]:
    """
    Use extreme value theory for very high MI values.
    
    When the observed MI is in the extreme tail, standard methods may be
    unreliable. This uses extreme value theory for better estimates.
    
    Parameters
    ----------
    observed_value : float
        The observed MI value to test.
    null_distribution : np.ndarray
        Array of MI values from shuffled data.
    threshold_percentile : float, optional
        Percentile above which to use extreme value theory.
        Default: 95
        
    Returns
    -------
    p_value : float
        P-value using appropriate method.
    method_used : str
        Description of the method used.
    """
    threshold = np.percentile(null_distribution, threshold_percentile)
    
    if observed_value <= threshold:
        # Use standard empirical method
        p_value = empirical_p_value(observed_value, null_distribution, method='conservative')
        return p_value, "empirical (below extreme threshold)"
    
    # Use extreme value theory
    # Fit Generalized Pareto Distribution to exceedances
    exceedances = null_distribution[null_distribution > threshold] - threshold
    
    if len(exceedances) < 10:
        # Not enough extreme values
        p_value = empirical_p_value(observed_value, null_distribution, method='conservative')
        return p_value, "empirical (too few exceedances)"
    
    try:
        # Fit GPD
        params = stats.genpareto.fit(exceedances)
        gpd = stats.genpareto(*params)
        
        # Probability of exceeding threshold
        p_threshold = len(exceedances) / len(null_distribution)
        
        # Probability of exceeding observed value given exceedance
        if observed_value > threshold:
            p_exceed_given_threshold = gpd.sf(observed_value - threshold)
            p_value = p_threshold * p_exceed_given_threshold
        else:
            p_value = empirical_p_value(observed_value, null_distribution, method='conservative')
            
        return p_value, f"extreme value theory (threshold={threshold:.3f})"
        
    except Exception:
        # If EVT fails, use empirical
        p_value = empirical_p_value(observed_value, null_distribution, method='conservative')
        return p_value, "empirical (EVT fit failed)"


def bootstrap_p_value_confidence(observed_value: float,
                               null_distribution: np.ndarray,
                               n_bootstrap: int = 1000,
                               confidence_level: float = 0.95) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate p-value with bootstrap confidence interval.
    
    This provides uncertainty quantification for the p-value estimate,
    which is especially important for small sample sizes.
    
    Parameters
    ----------
    observed_value : float
        The observed MI value to test.
    null_distribution : np.ndarray
        Array of MI values from shuffled data.
    n_bootstrap : int, optional
        Number of bootstrap samples. Default: 1000
    confidence_level : float, optional
        Confidence level for interval. Default: 0.95
        
    Returns
    -------
    p_value : float
        Point estimate of p-value.
    confidence_interval : tuple
        (lower, upper) bounds of confidence interval.
    """
    n = len(null_distribution)
    bootstrap_p_values = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample from null distribution
        bootstrap_sample = np.random.choice(null_distribution, size=n, replace=True)
        
        # Calculate empirical p-value
        p_boot = empirical_p_value(observed_value, bootstrap_sample, method='conservative')
        bootstrap_p_values.append(p_boot)
    
    # Point estimate (original p-value)
    p_value = empirical_p_value(observed_value, null_distribution, method='conservative')
    
    # Confidence interval
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_p_values, 100 * alpha / 2)
    upper = np.percentile(bootstrap_p_values, 100 * (1 - alpha / 2))
    
    return p_value, (lower, upper)


class ImprovedMITesting:
    """
    Improved MI testing framework that addresses the distribution fitting issues.
    
    This class provides multiple testing strategies and automatically selects
    the most appropriate one based on the data characteristics.
    """
    
    def __init__(self, 
                 method: str = 'auto',
                 min_samples_for_parametric: int = 100,
                 extreme_value_threshold: float = 95):
        """
        Initialize improved MI testing framework.
        
        Parameters
        ----------
        method : str, optional
            Testing method to use. Options:
            - 'auto': Automatically select best method
            - 'empirical': Always use empirical p-values
            - 'adaptive': Use adaptive distribution selection
            - 'robust': Use robust parametric/empirical hybrid
            - 'extreme': Use extreme value theory when appropriate
            Default: 'auto'
        min_samples_for_parametric : int, optional
            Minimum samples required for parametric methods. Default: 100
        extreme_value_threshold : float, optional
            Percentile threshold for extreme value theory. Default: 95
        """
        self.method = method
        self.min_samples_for_parametric = min_samples_for_parametric
        self.extreme_value_threshold = extreme_value_threshold
        
    def compute_p_value(self, 
                       observed_value: float,
                       null_distribution: np.ndarray,
                       return_details: bool = False) -> Union[float, Tuple[float, Dict]]:
        """
        Compute p-value using the configured method.
        
        Parameters
        ----------
        observed_value : float
            The observed MI value.
        null_distribution : np.ndarray
            Shuffled MI values.
        return_details : bool, optional
            Whether to return additional details. Default: False
            
        Returns
        -------
        p_value : float
            The computed p-value.
        details : dict (optional)
            Additional information about the computation.
        """
        details = {}
        
        if self.method == 'empirical':
            p_value = empirical_p_value(observed_value, null_distribution, method='conservative')
            details['method'] = 'empirical'
            
        elif self.method == 'adaptive':
            p_value, best_dist, fit_info = adaptive_distribution_test(observed_value, null_distribution)
            details['method'] = f'adaptive ({best_dist})'
            details['fit_info'] = fit_info
            
        elif self.method == 'robust':
            p_value, method_desc = robust_parametric_test(
                observed_value, null_distribution, self.min_samples_for_parametric
            )
            details['method'] = f'robust ({method_desc})'
            
        elif self.method == 'extreme':
            p_value, method_desc = extreme_value_correction(
                observed_value, null_distribution, self.extreme_value_threshold
            )
            details['method'] = f'extreme ({method_desc})'
            
        elif self.method == 'auto':
            # Automatic method selection based on data characteristics
            n = len(null_distribution)
            
            if n < 50:
                # Too few samples - use empirical
                p_value = empirical_p_value(observed_value, null_distribution, method='conservative')
                details['method'] = f'empirical (n={n} too small)'
                
            elif observed_value > np.percentile(null_distribution, 99):
                # Extreme value - use EVT
                p_value, method_desc = extreme_value_correction(
                    observed_value, null_distribution, self.extreme_value_threshold
                )
                details['method'] = f'extreme/auto ({method_desc})'
                
            else:
                # Try robust parametric approach
                p_value, method_desc = robust_parametric_test(
                    observed_value, null_distribution, self.min_samples_for_parametric
                )
                details['method'] = f'robust/auto ({method_desc})'
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        details['p_value'] = p_value
        details['n_shuffles'] = len(null_distribution)
        details['observed_value'] = observed_value
        
        if return_details:
            return p_value, details
        else:
            return p_value


def compare_testing_methods(observed_value: float,
                          null_distribution: np.ndarray) -> Dict[str, Dict]:
    """
    Compare all available testing methods for diagnostic purposes.
    
    Parameters
    ----------
    observed_value : float
        The observed MI value.
    null_distribution : np.ndarray
        Shuffled MI values.
        
    Returns
    -------
    comparison : dict
        Results from all testing methods.
    """
    results = {}
    
    # Current INTENSE approach
    from .stats import get_mi_distr_pvalue
    results['current_norm'] = {
        'p_value': get_mi_distr_pvalue(null_distribution, observed_value, 'norm'),
        'method': 'normal distribution (current)'
    }
    results['current_gamma'] = {
        'p_value': get_mi_distr_pvalue(null_distribution, observed_value, 'gamma'),
        'method': 'gamma distribution (current)'
    }
    
    # Empirical
    results['empirical'] = {
        'p_value': empirical_p_value(observed_value, null_distribution),
        'method': 'empirical (conservative)'
    }
    
    # Adaptive
    try:
        p_val, best_dist, _ = adaptive_distribution_test(observed_value, null_distribution)
        results['adaptive'] = {
            'p_value': p_val,
            'method': f'adaptive ({best_dist})'
        }
    except:
        results['adaptive'] = {'p_value': np.nan, 'method': 'adaptive (failed)'}
    
    # Robust
    try:
        p_val, method_desc = robust_parametric_test(observed_value, null_distribution)
        results['robust'] = {
            'p_value': p_val,
            'method': f'robust ({method_desc})'
        }
    except:
        results['robust'] = {'p_value': np.nan, 'method': 'robust (failed)'}
    
    # Extreme value
    try:
        p_val, method_desc = extreme_value_correction(observed_value, null_distribution)
        results['extreme'] = {
            'p_value': p_val,
            'method': f'extreme ({method_desc})'
        }
    except:
        results['extreme'] = {'p_value': np.nan, 'method': 'extreme (failed)'}
    
    # Bootstrap CI
    try:
        p_val, (ci_low, ci_high) = bootstrap_p_value_confidence(observed_value, null_distribution, n_bootstrap=200)
        results['bootstrap'] = {
            'p_value': p_val,
            'method': f'empirical with 95% CI: [{ci_low:.4f}, {ci_high:.4f}]'
        }
    except:
        results['bootstrap'] = {'p_value': np.nan, 'method': 'bootstrap (failed)'}
    
    return results