from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics.cluster import mutual_info_score
import scipy

from .ksg import *
from .gcmi import *
from .info_utils import binary_mi_score
from ..utils.data import correlation_matrix
from .entropy import entropy_d, joint_entropy_dd, joint_entropy_cd, joint_entropy_cdd
from ..dim_reduction.data import MVData

import numpy as np
import warnings
from typing import Optional
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import entropy, differential_entropy

from ..utils.data import to_numpy_array


DEFAULT_NN = 5

# TODO: add @property decorators to properly set getter-setter functionality


class TimeSeries():
    @staticmethod
    def define_ts_type(ts):
        if len(ts) < 100:
            warnings.warn('Time series is too short for accurate type (discrete/continuous) determination')

        unique_vals = np.unique(ts)
        sc1 = len(unique_vals) / len(ts)
        hist = np.histogram(ts, bins=len(ts))[0]
        ent = entropy(hist)
        maxent = entropy(np.ones(len(ts)))
        sc2 = ent / maxent

        # TODO: refactor thresholds
        if sc1 > 0.70 and sc2 > 0.70:
            return False  # both scores are high - the variable is most probably continuous
        elif sc1 < 0.25 and sc2 < 0.25:
            return True  # both scores are low - the variable is most probably discrete
        else:
            raise ValueError(f'Unable to determine time series type automatically: score 1 = {sc1}, score 2 = {sc2}')

    # TODO: complete this function
    def _check_input(self):
        pass

    def __init__(self, data, discrete=None, shuffle_mask=None):
        self.data = to_numpy_array(data)

        if discrete is None:
            #warnings.warn('Time series type not specified and will be inferred automatically')
            self.discrete = TimeSeries.define_ts_type(self.data)
        else:
            self.discrete = discrete

        scaler = MinMaxScaler()
        self.scdata = scaler.fit_transform(self.data.reshape(-1, 1)).reshape(1, -1)[0]
        self.data_scale = scaler.scale_
        self.copula_normal_data = None

        if self.discrete:
            self.int_data = np.round(self.data).astype(int)
            if len(set(self.data.astype(int))) == 2:
                self.is_binary = True
                self.bool_data = self.int_data.astype(bool)
            else:
                self.is_binary = False

        else:
            self.copula_normal_data = copnorm(self.data).ravel()

        self.entropy = dict()  # supports various downsampling constants
        self.kdtree = None
        self.kdtree_query = None

        if shuffle_mask is None:
            # which shuffles are valid
            self.shuffle_mask = np.ones(len(self.data)).astype(bool)
        else:
            self.shuffle_mask = shuffle_mask.astype(bool)

    def get_kdtree(self):
        if self.kdtree is None:
            tree = self._compute_kdtree()
            self.kdtree = tree

        return self.kdtree

    def _compute_kdtree(self):
        d = self.data.reshape(self.data.shape[0], -1)
        return build_tree(d)

    def get_kdtree_query(self, k=DEFAULT_NN):
        if self.kdtree_query is None:
            q = self._compute_kdtree_query(k=k)
            self.kdtree_query = q

        return self.kdtree_query

    def _compute_kdtree_query(self, k=DEFAULT_NN):
        tree = self.get_kdtree()
        return tree.query(self.data, k=k + 1)

    def get_entropy(self, ds=1):
        if ds not in self.entropy.keys():
            self._compute_entropy(ds=ds)
        return self.entropy[ds]

    def _compute_entropy(self, ds=1):
        if self.discrete:
            # TODO: rewrite this using int_data and via ent_d from driada.information.entropy
            counts = []
            for val in np.unique(self.data[::ds]):
                counts.append(len(np.where(self.data[::ds] == val)[0]))

            self.entropy[ds] = entropy(counts, base=np.e)

        else:
            self.entropy[ds] = nonparam_entropy_c(self.data) / np.log(2)
            #self.entropy[ds] = get_tdmi(self.scdata[::ds], min_shift=1, max_shift=2)[0]
            #raise AttributeError('Entropy for continuous variables is not yet implemented'
    
    def filter(self, method='gaussian', **kwargs):
        """
        Apply filtering to the time series and return a new filtered TimeSeries.
        
        Parameters
        ----------
        method : str
            Filtering method: 'gaussian', 'savgol', 'wavelet', or 'none'
        **kwargs : dict
            Method-specific parameters:
            - gaussian: sigma (default: 1.0)
            - savgol: window_length (default: 5), polyorder (default: 2)
            - wavelet: wavelet (default: 'db4'), level (default: None)
            
        Returns
        -------
        TimeSeries
            New TimeSeries object with filtered data
        """
        from ..utils.signals import filter_1d_timeseries
        
        if method == 'none':
            return TimeSeries(self.data.copy(), discrete=self.discrete, 
                            shuffle_mask=self.shuffle_mask.copy())
        
        if self.discrete:
            warnings.warn("Filtering discrete time series may produce unexpected results")
        
        # Apply filtering to 1D time series
        filtered_data = filter_1d_timeseries(self.data, method=method, **kwargs)
        
        # Create new TimeSeries with filtered data
        return TimeSeries(filtered_data, discrete=self.discrete, 
                         shuffle_mask=self.shuffle_mask.copy())
    
    def approximate_entropy(self, m: int = 2, r: Optional[float] = None) -> float:
        """
        Calculate approximate entropy (ApEn) of the time series.
        
        Approximate entropy is a regularity statistic that quantifies the 
        unpredictability of fluctuations in a time series. A time series 
        containing many repetitive patterns has a relatively small ApEn; 
        a less predictable process has a higher ApEn.
        
        Parameters
        ----------
        m : int, optional
            Pattern length. Common values are 1 or 2. Default is 2.
        r : float, optional
            Tolerance threshold for pattern matching. If None, defaults to
            0.2 times the standard deviation of the data.
        
        Returns
        -------
        float
            The approximate entropy value. Higher values indicate more 
            randomness/complexity.
        
        Raises
        ------
        ValueError
            If called on a discrete TimeSeries.
            
        Notes
        -----
        This method is only valid for continuous time series. For discrete
        time series, consider using other complexity measures.
        
        Examples
        --------
        >>> ts = TimeSeries(np.random.randn(1000), discrete=False)
        >>> apen = ts.approximate_entropy(m=2)
        >>> print(f"Approximate entropy: {apen:.3f}")
        """
        if self.discrete:
            raise ValueError("approximate_entropy is only valid for continuous time series")
        
        # Use lazy import to avoid circular imports
        from ..utils.signals import approximate_entropy
        
        # Default r to 0.2 * std if not provided
        if r is None:
            r = 0.2 * np.std(self.data)
        
        return approximate_entropy(self.data, m=m, r=r)


class MultiTimeSeries(MVData):
    """
    MultiTimeSeries represents multiple aligned time series.
    Now inherits from MVData to enable direct dimensionality reduction.
    Supports either all-continuous or all-discrete components (no mixing).
    """
    
    def __init__(self, data_or_tslist, labels=None, distmat=None, rescale_rows=False, 
                 data_name=None, downsampling=None, discrete=None, shuffle_mask=None):
        # Handle both numpy array and list of TimeSeries inputs
        if isinstance(data_or_tslist, np.ndarray):
            # Direct numpy array input: each row is a time series
            if data_or_tslist.ndim != 2:
                raise ValueError("When providing numpy array, it must be 2D with shape (n_series, n_timepoints)")
            if discrete is None:
                raise ValueError("When providing numpy array, 'discrete' parameter must be specified")
            
            # Set discrete flag early for numpy array input
            self.discrete = discrete
            
            # Create TimeSeries objects from numpy array rows for processing
            tslist = [TimeSeries(data_or_tslist[i, :], discrete=discrete) for i in range(data_or_tslist.shape[0])]
            data = data_or_tslist
            
            # Store provided shuffle_mask for later use (after combining with TimeSeries masks)
            self._provided_shuffle_mask = shuffle_mask
        else:
            # List of TimeSeries objects
            tslist = data_or_tslist
            self._check_input(tslist)
            # Stack data from all TimeSeries
            data = np.vstack([ts.data for ts in tslist])
            
            # Store provided shuffle_mask for later use
            self._provided_shuffle_mask = shuffle_mask
        
        # Initialize MVData parent class
        super().__init__(data, labels=labels, distmat=distmat, 
                        rescale_rows=rescale_rows, data_name=data_name, 
                        downsampling=downsampling)
        
        # Additional MultiTimeSeries specific attributes
        self.scdata = np.vstack([ts.scdata for ts in tslist])
        
        # Handle copula normal data for continuous components
        if not self.discrete:
            self.copula_normal_data = np.vstack([ts.copula_normal_data for ts in tslist])
        else:
            # For discrete MultiTimeSeries, store integer data
            self.int_data = np.vstack([ts.int_data for ts in tslist])
            self.copula_normal_data = None
        
        # Combine shuffle masks
        if hasattr(self, '_provided_shuffle_mask') and self._provided_shuffle_mask is not None:
            # If shuffle_mask was provided explicitly, use it
            self.shuffle_mask = self._provided_shuffle_mask
            if not np.any(self.shuffle_mask):
                warnings.warn('Provided shuffle_mask has no valid positions for shuffling!')
        else:
            # Otherwise, combine individual TimeSeries masks restrictively
            shuffle_masks = np.vstack([ts.shuffle_mask for ts in tslist])
            # Restrictive combination: ALL masks must allow shuffling at a position
            self.shuffle_mask = np.all(shuffle_masks, axis=0)
            
            # Check if the combined mask is problematic
            valid_positions = np.sum(self.shuffle_mask)
            total_positions = len(self.shuffle_mask)
            
            if valid_positions == 0:
                raise ValueError(f'Combined shuffle_mask has NO valid positions for shuffling! '
                                f'This typically happens when combining many neurons with restrictive individual masks. '
                                f'Consider providing an explicit shuffle_mask parameter to MultiTimeSeries.')
            elif valid_positions < 0.1 * total_positions:
                warnings.warn(f'Combined shuffle_mask is extremely restrictive: only {valid_positions}/{total_positions} '
                             f'({100*valid_positions/total_positions:.1f}%) positions are valid for shuffling. '
                             f'This may cause issues with shuffle-based significance testing.')
        
        self.entropy = dict()  # supports various downsampling constants
    
    @property
    def shape(self):
        """Return shape of the data for compatibility with numpy-like access."""
        return self.data.shape

    def _check_input(self, tslist):
        is_ts = np.array([isinstance(ts, TimeSeries) for ts in tslist])
        if not np.all(is_ts):
            raise ValueError('Input to MultiTimeSeries must be iterable of TimeSeries')
        
        # Check all TimeSeries have same length
        lengths = np.array([len(ts.data) for ts in tslist])
        if not np.all(lengths == lengths[0]):
            raise ValueError('All TimeSeries must have the same length')
        
        # Check all TimeSeries have same discrete/continuous type
        is_discrete = np.array([ts.discrete for ts in tslist])
        if not (np.all(is_discrete) or np.all(~is_discrete)):
            raise ValueError('All components of MultiTimeSeries must be either continuous or discrete (no mixing)')
        
        # Set discrete flag based on components
        self.discrete = is_discrete[0]

    def get_entropy(self, ds=1):
        if ds not in self.entropy.keys():
            self._compute_entropy(ds=ds)
        return self.entropy[ds]

    def _compute_entropy(self, ds=1):
        if self.discrete:
            # All components are discrete - use joint discrete entropy
            self.entropy[ds] = entropy_d(self.int_data[:, ::ds])
        else:
            # All continuous - use existing continuous entropy
            self.entropy[ds] = ent_g(self.data[:, ::ds])
    
    def filter(self, method='gaussian', **kwargs):
        """
        Apply filtering to all time series components and return a new filtered MultiTimeSeries.
        
        Parameters
        ----------
        method : str
            Filtering method: 'gaussian', 'savgol', 'wavelet', or 'none'
        **kwargs : dict
            Method-specific parameters (see TimeSeries.filter for details)
            
        Returns
        -------
        MultiTimeSeries
            New MultiTimeSeries object with all components filtered
        """
        from ..signals.neural_filtering import filter_neural_signals
        
        # Apply filtering to all time series at once
        filtered_data = filter_neural_signals(self.data, method=method, **kwargs)
        
        # Create new MultiTimeSeries from filtered data
        return MultiTimeSeries(filtered_data, labels=self.labels, 
                              rescale_rows=self.rescale_rows, 
                              data_name=self.data_name, discrete=self.discrete)


def get_stats_function(sname):
    try:
        return getattr(scipy.stats, sname)
    except AttributeError:
        raise ValueError(f"Metric '{sname}' not found in scipy.stats")


def calc_signal_ratio(binary_ts, continuous_ts):
    # Calculate average of continuous_ts when binary_ts is 1 or 0
    avg_on = np.mean(continuous_ts[binary_ts == 1])
    avg_off = np.mean(continuous_ts[binary_ts == 0])

    # Calculate ratio (handle division by zero)
    if avg_off == 0:
        return np.inf if avg_on != 0 else np.nan

    return avg_on / avg_off


def get_sim(x, y, metric, shift=0, ds=1, k=5, estimator='gcmi', check_for_coincidence=False):
    """Computes similarity between two (possibly multidimensional) variables efficiently

            Parameters
            ----------
            x: TimeSeries/MultiTimeSeries instance or numpy array

            y: TimeSeries/MultiTimeSeries instance or numpy array

            metric: similarity metric between time series

            shift: int
                y will be roll-moved by the number 'shift' after downsampling by 'ds' factor

            ds: int
                downsampling constant (take every 'ds'-th point)

            Returns
            -------
            me: similarity metric between x and (possibly) shifted y

            """
    def _check_input(ts):
        if not isinstance(ts, TimeSeries) and not isinstance(ts, MultiTimeSeries):
            if np.ndim(ts) == 1:
                ts = TimeSeries(ts)
            else:
                raise Exception('Multidimensional inputs must be provided as MultiTimeSeries')
        return ts

    ts1 = _check_input(x)
    ts2 = _check_input(y)

    if metric == 'mi':
        me = get_mi(ts1, ts2, shift=shift, ds=ds, k=k, estimator=estimator,
                                check_for_coincidence=check_for_coincidence)

    else:
        if isinstance(ts1, TimeSeries) and isinstance(ts2, TimeSeries):
            if not ts1.discrete and not ts2.discrete:
                if metric == 'fast_pearsonr':
                    x = ts1.data[::ds]
                    y = np.roll(ts2.data[::ds], shift)
                    me = correlation_matrix(np.vstack([x, y]))[0, 1]
                else:
                    metric_func = get_stats_function(metric)
                    me = metric_func(ts1.data[::ds], np.roll(ts2.data[::ds], shift))[0]

            if ts1.discrete and not ts2.discrete:
                if metric == 'av':
                    if ts1.is_binary:
                        me = calc_signal_ratio(ts1.data[::ds], np.roll(ts2.data[::ds], shift))
                    else:
                        raise ValueError(f'Discrete ts must be binary for metric={metric}')
                else:
                    raise ValueError("Only 'av' and 'mi' metrics are supported for binary-continuous similarity")

            if ts2.discrete and not ts1.discrete:
                if metric == 'av':
                    if ts2.is_binary:
                        me = calc_signal_ratio(ts2.data[::ds], np.roll(ts1.data[::ds], shift))
                    else:
                        raise ValueError(f'Discrete ts must be binary for metric={metric}')
                else:
                    raise ValueError("Only 'av' and 'mi' metrics are supported for binary-continuous similarity")

            if ts2.discrete and ts1.discrete:
                raise ValueError(f'Metric={metric} is not supported for two discrete ts')

        else:
            raise Exception("Metrics except 'mi' are not supported for multi-dimensional data")

    return me


def get_mi(x, y, shift=0, ds=1, k=5, estimator='gcmi', check_for_coincidence=False):
    """Computes mutual information between two (possibly multidimensional) variables efficiently

    Parameters
    ----------
    x: TimeSeries/MultiTimeSeries instance or numpy array
    y: TimeSeries/MultiTimeSeries instance or numpy array
    shift: int
        y will be roll-moved by the number 'shift' after downsampling by 'ds' factor
    ds: int
        downsampling constant (take every 'ds'-th point)
    k: int
        number of neighbors for ksg estimator
    estimator: str
        Estimation method. Should be 'ksg' (accurate but slow) and 'gcmi' (fast, but estimates the lower bound on MI).
        In most cases 'gcmi' should be preferred.

    Returns
    -------
    mi: mutual information (or its lower bound in case of 'gcmi' estimator) between x and (possibly) shifted y

    """

    def _check_input(ts):
        if not isinstance(ts, TimeSeries) and not isinstance(ts, MultiTimeSeries):
            if np.ndim(ts) == 1:
                ts = TimeSeries(ts)
            else:
                raise Exception('Multidimensional inputs must be provided as MultiTimeSeries')
        return ts

    def multi_single_mi(mts, ts, ds=1, k=5, estimator='gcmi'):
        if estimator == 'ksg':
            raise NotImplementedError('KSG estimator is not supported for dim>1 yet')

        # Safety check: if single TimeSeries data is contained in MultiTimeSeries
        # This should not happen due to aggregate_multiple_ts adding noise, but add as safety net
        if not ts.discrete and shift == 0:
            # Check if any row of the MultiTimeSeries matches the single TimeSeries
            for i in range(mts.data.shape[0]):
                if np.allclose(mts.data[i, ::ds], ts.data[::ds], rtol=1e-10, atol=1e-10):
                    warnings.warn('MI computation between MultiTimeSeries containing identical data detected, returning 0')
                    return 0

        if ts.discrete:
            ny1 = np.roll(mts.copula_normal_data[:, ::ds], shift)
            # Ensure ny1 is contiguous for better performance with Numba
            if not ny1.flags['C_CONTIGUOUS']:
                ny1 = np.ascontiguousarray(ny1)
            ny2 = ts.int_data[::ds]
            mi = mi_model_gd(ny1, ny2, np.max(ny2), biascorrect=True, demeaned=True)

        else:
            ny1 = mts.copula_normal_data[:, ::ds]
            ny2 = np.roll(ts.copula_normal_data[::ds], shift)
            mi = mi_gg(ny1, ny2, True, True)

        return mi

    def multi_multi_mi(mts1, mts2, ds=1, k=5, estimator='gcmi', check_for_coincidence=False):
        if estimator == 'ksg':
            raise NotImplementedError('KSG estimator is not supported for dim>1 yet')

        if check_for_coincidence:
            if np.allclose(ts1.data, ts2.data) and shift == 0:  # and not (ts1.discrete and ts2.discrete):
                warnings.warn('MI computation of a MultiTimeSeries with itself is meaningless, 0 will be returned forcefully')
                # raise ValueError('MI(X,X) computation for continuous variable X should give an infinite result')
                return 0

        if mts1.discrete or mts2.discrete:
            raise NotImplementedError('MI computation between MultiTimeSeries\
             is currently supported for continuous data only')

        else:
            ny1 = mts1.copula_normal_data[:, ::ds]
            ny2 = np.roll(mts2.copula_normal_data[:, ::ds], shift, axis=1)
            mi = mi_gg(ny1, ny2, True, True)

        return mi

    ts1 = _check_input(x)
    ts2 = _check_input(y)

    if isinstance(ts1, TimeSeries) and isinstance(ts2, TimeSeries):
        mi = get_1d_mi(x, y, shift=shift, ds=ds, k=k, estimator=estimator,
                       check_for_coincidence=check_for_coincidence)

    if isinstance(ts1, MultiTimeSeries) and isinstance(ts2, TimeSeries):
        mi = multi_single_mi(ts1, ts2, ds=ds, k=k, estimator=estimator)

    if isinstance(ts2, MultiTimeSeries) and isinstance(ts1, TimeSeries):
        mi = multi_single_mi(ts2, ts1, ds=ds, k=k, estimator=estimator)

    if isinstance(ts1, MultiTimeSeries) and isinstance(ts2, MultiTimeSeries):
        mi = multi_multi_mi(ts1, ts2, ds=ds, k=k, estimator=estimator,
                            check_for_coincidence=check_for_coincidence)
        #raise NotImplementedError('MI computation between two MultiTimeSeries is not supported yet')

    if mi < 0:
        mi = 0

    return mi


def get_1d_mi(ts1, ts2, shift=0, ds=1, k=5, estimator='gcmi', check_for_coincidence=True):
    """Computes mutual information between two 1d variables efficiently

    Parameters
    ----------
    ts1: TimeSeries/MultiTimeSeries instance or numpy array
    ts2: TimeSeries/MultiTimeSeries instance or numpy array
    shift: int
        ts2 will be roll-moved by the number 'shift' after downsampling by 'ds' factor
    ds: int
        downsampling constant (take every 'ds'-th point)
    k: int
        number of neighbors for ksg estimator
    estimator: str
        Estimation method. Should be 'ksg' (accurate but slow) and 'gcmi' (fast, but estimates the lower bound on MI).
        In most cases 'gcmi' should be preferred.
    check_for_coincidence : bool, optional
        If True, raises error when computing MI of a signal with itself at zero shift. Default: True.

    Returns
    -------
    mi: mutual information (or its lower bound in case of 'gcmi' estimator) between ts1 and (possibly) shifted ts2

    """

    def _check_input(ts):
        if not isinstance(ts, TimeSeries) and not isinstance(ts, MultiTimeSeries):
            if np.ndim(ts) == 1:
                ts = TimeSeries(ts)
            else:
                raise Exception('Multidimensional inputs must be provided as MultiTimeSeries')
        return ts

    ts1 = _check_input(ts1)
    ts2 = _check_input(ts2)

    if check_for_coincidence and ts1.data.shape == ts2.data.shape:
        if np.allclose(ts1.data, ts2.data) and shift == 0: #and not (ts1.discrete and ts2.discrete):
            raise ValueError('MI computation of a TimeSeries or MultiTimeSeries with itself is not allowed')
            #raise ValueError('MI(X,X) computation for continuous variable X should give an infinite result')

    if estimator == 'ksg':
        #TODO: add shifts everywhere in this branch
        x = ts1.data[::ds].reshape(-1, 1)
        y = ts2.data[::ds]
        if shift != 0:
            y = np.roll(y, shift)

        if not ts1.discrete and not ts2.discrete:
            mi = nonparam_mi_cc_mod(ts1.data, y, k=k,
                                    precomputed_tree_x=ts1.get_kdtree(),
                                    precomputed_tree_y=ts2.get_kdtree())

        elif ts1.discrete and ts2.discrete:
            mi = mutual_info_classif(ts1.int_data[::ds].reshape(-1, 1),
                                     ts2.int_data[::ds],
                                     discrete_features=True,
                                     n_neighbors=k)[0]

        # TODO: refactor using ksg functions
        elif ts1.discrete and not ts2.discrete:
            mi = mutual_info_regression(ts1.int_data[::ds],
                                        y[::ds],
                                        discrete_features=False,
                                        n_neighbors=k)[0]

        elif not ts1.discrete and ts2.discrete:
            mi = mutual_info_classif(x[::ds],
                                     ts2.int_data[::ds],
                                     discrete_features=True,
                                     n_neighbors=k)[0]

        return mi

    elif estimator == 'gcmi':
        if not ts1.discrete and not ts2.discrete:
            ny1 = ts1.copula_normal_data[::ds]
            ny2 = np.roll(ts2.copula_normal_data[::ds], shift)
            mi = mi_gg(ny1, ny2, True, True)

        elif ts1.discrete and ts2.discrete:
            # if features are binary:
            if ts1.is_binary and ts2.is_binary:
                ny1 = ts1.bool_data[::ds]
                ny2 = np.roll(ts2.bool_data[::ds], shift)

                contingency = np.zeros((2, 2))
                contingency[0, 0] = (ny1 & ny2).sum()
                contingency[0, 1] = (~ny1 & ny2).sum()
                contingency[1, 0] = (ny1 & ~ny2).sum()
                contingency[1, 1] = (~ny1 & ~ny2).sum()

                mi = binary_mi_score(contingency)

            else:
                ny1 = ts1.int_data[::ds]  # .reshape(-1, 1)
                ny2 = np.roll(ts2.int_data[::ds], shift)
                mi = mutual_info_score(ny1, ny2)

        elif ts1.discrete and not ts2.discrete:
            ny1 = ts1.int_data[::ds]
            ny2 = np.roll(ts2.copula_normal_data[::ds], shift)
            # Ensure ny2 is contiguous for better performance with Numba
            if not ny2.flags['C_CONTIGUOUS']:
                ny2 = np.ascontiguousarray(ny2)
            mi = mi_model_gd(ny2, ny1, np.max(ny1), biascorrect=True, demeaned=True)

        elif not ts1.discrete and ts2.discrete:
            ny1 = ts1.copula_normal_data[::ds]
            #TODO: fix zd error
            ny2 = np.roll(ts2.int_data[::ds], shift)
            #ny2 = np.roll(ts2.data[::ds], shift)
            '''
            print(ny2)
            print(sum(ny2))
            print(ny1)
            '''
            # Ensure ny1 is contiguous for better performance with Numba
            if not ny1.flags['C_CONTIGUOUS']:
                ny1 = np.ascontiguousarray(ny1)
            mi = mi_model_gd(ny1, ny2, np.max(ny2), biascorrect=True, demeaned=True)

        if mi < 0:
            mi = 0

        return mi


def get_tdmi(data, min_shift=1, max_shift=100, nn=DEFAULT_NN):
    ts = TimeSeries(data, discrete=False)
    tdmi = [get_1d_mi(ts, ts, shift=shift, k=nn) for shift in range(min_shift, max_shift)]

    return tdmi


def get_multi_mi(tslist, ts2, shift=0, ds=1, k=DEFAULT_NN, estimator='gcmi'):

    #TODO: make shift the same as in get_1d_mi
    if ~np.all([ts.discrete for ts in tslist]) and not ts2.discrete:
        nylist = [ts.copula_normal_data[::ds] for ts in tslist]
        ny1 = np.vstack(nylist)
        ny2 = np.roll(ts2.copula_normal_data, shift)[::ds]
        mi = mi_gg(ny1, ny2, True, True)
    else:
        raise ValueError('Multidimensional MI only implemented for continuous data!')

    if mi < 0:
        mi = 0

    return mi


def aggregate_multiple_ts(*ts_args, noise=1e-5):
    """Aggregate multiple continuous TimeSeries into a single MultiTimeSeries.
    
    Adds small noise to break degeneracy and creates a MultiTimeSeries from
    the input TimeSeries objects.
    
    Parameters
    ----------
    *ts_args : TimeSeries
        Variable number of TimeSeries objects to aggregate.
    noise : float, optional
        Amount of noise to add to break degeneracy. Default: 1e-5.
        
    Returns
    -------
    MultiTimeSeries
        Aggregated multi-dimensional time series.
        
    Raises
    ------
    ValueError
        If any input TimeSeries is discrete.
        
    Examples
    --------
    >>> ts1 = TimeSeries(np.random.randn(100), discrete=False)
    >>> ts2 = TimeSeries(np.random.randn(100), discrete=False)
    >>> mts = aggregate_multiple_ts(ts1, ts2)
    """
    # add small noise to break degeneracy
    mod_tslist = []
    for ts in ts_args:
        if ts.discrete:
            raise ValueError('this is not applicable to discrete TimeSeries')
        mod_ts = TimeSeries(ts.data + np.random.random(size=len(ts.data)) * noise, discrete=False)
        mod_tslist.append(mod_ts)

    mts = MultiTimeSeries(mod_tslist)  # add last two TS into a single 2-d MTS
    return mts


def conditional_mi(ts1, ts2, ts3, ds=1, k=5):
    """Calculate conditional mutual information I(X;Y|Z).
    
    Computes the conditional mutual information between ts1 (X) and ts2 (Y)
    given ts3 (Z) for various combinations of continuous and discrete variables.
    
    Parameters
    ----------
    ts1 : TimeSeries
        First variable (X). Must be continuous.
    ts2 : TimeSeries
        Second variable (Y). Can be continuous or discrete.
    ts3 : TimeSeries
        Conditioning variable (Z). Can be continuous or discrete.
    ds : int, optional
        Downsampling factor. Default: 1.
    k : int, optional
        Number of neighbors for entropy estimation. Default: 5.
        
    Returns
    -------
    float
        Conditional mutual information I(X;Y|Z) in bits.
        
    Raises
    ------
    ValueError
        If ts1 is discrete (only continuous X is currently supported).
        
    Notes
    -----
    Supports four cases:
    - CCC: All continuous - uses Gaussian copula
    - CCD: X,Y continuous, Z discrete - uses Gaussian copula per Z value
    - CDC: X,Z continuous, Y discrete - uses chain rule identity
    - CDD: X continuous, Y,Z discrete - uses entropy decomposition
    
    For the CDD case, GCMI estimator has limitations due to uncontrollable 
    biases (copula transform does not conserve entropy). See 
    https://doi.org/10.1002/hbm.23471 for details.
    """
    if ts1.discrete:
        raise ValueError('conditional MI(X,Y|Z) is currently implemented for continuous X only')

    #print(ts1.discrete, ts2.discrete, ts3.discrete)
    if not ts2.discrete and not ts3.discrete:
        # CCC: All continuous
        g1 = ts1.copula_normal_data[::ds]
        g2 = ts2.copula_normal_data[::ds]
        g3 = ts3.copula_normal_data[::ds]
        cmi = cmi_ggg(g1, g2, g3, biascorrect=True, demeaned=True)

    elif not ts2.discrete and ts3.discrete:
        # CCD: X,Y continuous, Z discrete
        unique_discrete_vals = np.unique(ts3.int_data[::ds])
        cmi = gccmi_ccd(ts1.data[::ds],
                        ts2.data[::ds],
                        ts3.int_data[::ds],
                        len(unique_discrete_vals))

    elif ts2.discrete and not ts3.discrete:
        # CDC: X,Z continuous, Y discrete
        # Use entropy-based identity: I(X;Y|Z) = H(X|Z) - H(X|Y,Z)
        # This avoids mixing different MI estimators that cause bias inconsistency
        
        # H(X|Z) for continuous X,Z using GCMI
        x_data = ts1.data[::ds].reshape(1, -1)
        z_data = ts3.data[::ds].reshape(1, -1)
        
        # Joint data for H(X,Z) and marginal H(Z)
        xz_joint = np.vstack([x_data, z_data])
        H_xz = ent_g(xz_joint, biascorrect=True)
        H_z = ent_g(z_data, biascorrect=True)
        H_x_given_z = H_xz - H_z
        
        # H(X|Y,Z) - conditional entropy of X given both Y (discrete) and Z (continuous)
        unique_y_vals = np.unique(ts2.int_data[::ds])
        H_x_given_yz = 0.0
        
        for y_val in unique_y_vals:
            # Find indices where Y = y_val
            y_mask = (ts2.int_data[::ds] == y_val)
            n_y = np.sum(y_mask)
            
            if n_y > 2:  # Need sufficient samples for entropy estimation
                # Extract X,Z values for this Y group
                x_subset = x_data[:, y_mask]
                z_subset = z_data[:, y_mask]
                
                # Joint entropy H(X,Z|Y=y_val)
                xz_subset = np.vstack([x_subset, z_subset])
                H_xz_given_y = ent_g(xz_subset, biascorrect=True)
                
                # Marginal entropy H(Z|Y=y_val)
                H_z_given_y = ent_g(z_subset, biascorrect=True)
                
                # Conditional entropy H(X|Z,Y=y_val) = H(X,Z|Y=y_val) - H(Z|Y=y_val)
                H_x_given_z_y = H_xz_given_y - H_z_given_y
                
                # Weight by probability P(Y=y_val)
                p_y = n_y / len(ts2.int_data[::ds])
                H_x_given_yz += p_y * H_x_given_z_y
        
        # Final CMI calculation
        cmi = H_x_given_z - H_x_given_yz
        
        # Ensure CMI >= 0 due to information theory constraint
        # Small negative values are due to numerical precision and estimation noise
        if cmi < 0 and abs(cmi) < 0.01:
            cmi = 0.0

    else:
        # CDD: X continuous, Y,Z discrete
        # Here we use the identity I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
        '''
        # TODO: check this
        # Note that GCMI estimator is poorly applicable here because of the uncontrollable biases:
        # GCMI correctly estimates the lower bound on MI, but copula transform does not conserve the entropy
        # See  https://doi.org/10.1002/hbm.23471 for further details
        # Therefore, joint entropy estimation relies on ksg estimator instead
        '''
        # Note: Original code used copula_normal_data, but our entropy functions expect raw data
        # Using data instead of copula_normal_data for consistency with entropy functions
        H_xz = joint_entropy_cd(ts3.int_data[::ds], ts1.data[::ds], k=k)
        H_yz = joint_entropy_dd(ts2.int_data[::ds], ts3.int_data[::ds])
        H_xyz = joint_entropy_cdd(ts2.int_data[::ds], ts3.int_data[::ds], ts1.data[::ds], k=k)
        H_z = entropy_d(ts3.int_data[::ds])
        #print('entropies:', H_xz, H_yz, H_xyz, H_z)
        cmi = H_xz + H_yz - H_xyz - H_z

    return cmi


def interaction_information(ts1, ts2, ts3, ds=1, k=5):
    """Calculate three-way interaction information II(X;Y;Z).
    
    The interaction information quantifies the amount of information
    that is shared among all three variables. It can be positive (synergy)
    or negative (redundancy).
    
    Parameters
    ----------
    ts1 : TimeSeries
        First variable (X). Must be continuous.
    ts2 : TimeSeries
        Second variable (Y). Can be continuous or discrete.
    ts3 : TimeSeries
        Third variable (Z). Can be continuous or discrete.
    ds : int, optional
        Downsampling factor. Default: 1.
    k : int, optional
        Number of neighbors for entropy estimation. Default: 5.
        
    Returns
    -------
    float
        Interaction information II(X;Y;Z) in bits.
        - II < 0: Redundancy (Y and Z provide overlapping information about X)
        - II > 0: Synergy (Y and Z together provide more information than separately)
        
    Notes
    -----
    The interaction information is computed using Williams & Beer convention:
    II(X;Y;Z) = I(X;Y|Z) - I(X;Y) = I(X;Z|Y) - I(X;Z)
    
    This implementation assumes X is the target variable (e.g., neural activity)
    and Y, Z are predictor variables (e.g., behavioral features).
    """
    # Compute pairwise mutual information
    mi_xy = get_mi(ts1, ts2, ds=ds)
    mi_xz = get_mi(ts1, ts3, ds=ds)
    
    # Compute conditional mutual information
    cmi_xy_given_z = conditional_mi(ts1, ts2, ts3, ds=ds, k=k)
    cmi_xz_given_y = conditional_mi(ts1, ts3, ts2, ds=ds, k=k)
    
    # Compute interaction information (should be the same from both formulas)
    # Using Williams & Beer convention: II = I(X;Y|Z) - I(X;Y)
    # This gives negative II for redundancy and positive II for synergy
    ii_1 = cmi_xy_given_z - mi_xy
    ii_2 = cmi_xz_given_y - mi_xz
    
    # Average for numerical stability
    ii = (ii_1 + ii_2) / 2.0
    
    return ii