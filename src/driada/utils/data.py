import hashlib
import h5py
import scipy.sparse as ssp
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import hilbert
import numpy as np
import scipy.stats as st


def create_correlated_gaussian_data(
    n_features=10, n_samples=10000, correlation_pairs=None, seed=42
):
    """Generate multivariate Gaussian data with specified correlations.

    Creates synthetic data from a multivariate normal distribution with
    specified correlations between features. The data has zero mean and
    correlations specified by correlation_pairs.

    Parameters
    ----------
    n_features : int, optional
        Number of features (dimensions). Must be positive. Default is 10.
    n_samples : int, optional
        Number of samples to generate. Must be non-negative. Default is 10000.
    correlation_pairs : list of tuples or None, optional
        List of (i, j, correlation) tuples specifying correlated features.
        Indices i, j should be in range [0, n_features). Correlations must
        be in [-1, 1]. Out-of-bounds indices are silently ignored.
        If None, uses default pattern: [(1, 9, 0.9), (2, 8, 0.8), (3, 7, 0.7)].
    seed : int, optional
        Random seed for reproducibility. Sets global numpy random state.
        Default is 42.

    Returns
    -------
    data : np.ndarray
        Data array of shape (n_features, n_samples) with samples as columns.
    cov_matrix : np.ndarray
        Correlation matrix used to generate the data (n_features, n_features).
        Positive definite, with 1s on diagonal.
        
    Raises
    ------
    ValueError
        If n_features <= 0 or n_samples < 0.
        If any correlation value is outside [-1, 1].
        
    Notes
    -----
    This function modifies the global numpy random state via np.random.seed().
    For thread-safe random generation, consider using numpy.random.Generator.
    
    The correlation matrix is made positive definite if needed by adding a
    small value to the diagonal.
    
    Examples
    --------
    >>> data, corr = create_correlated_gaussian_data(n_features=3, n_samples=100)
    >>> data.shape
    (3, 100)
    >>> np.allclose(corr.diagonal(), 1.0)
    True    """
    # Input validation
    check_positive(n_features=n_features)
    check_nonnegative(n_samples=n_samples)
    
    np.random.seed(seed)
    if correlation_pairs is None:
        correlation_pairs = [(1, 9, 0.9), (2, 8, 0.8), (3, 7, 0.7)]

    # Create correlation matrix
    C = np.eye(n_features)
    for i, j, corr in correlation_pairs:
        # Validate correlation value
        if not -1 <= corr <= 1:
            raise ValueError(f"Correlation must be in [-1, 1], got {corr} for pair ({i}, {j})")
        # Apply correlation only if indices are valid
        if i < n_features and j < n_features:
            C[i, j] = C[j, i] = corr

    # Ensure positive definite
    min_eig = np.min(np.linalg.eigvals(C))
    if min_eig < 0:
        C += (-min_eig + 0.01) * np.eye(n_features)

    # Generate data
    data = np.random.multivariate_normal(np.zeros(n_features), C, size=n_samples).T

    return data, C


def populate_nested_dict(content, outer, inner):
    """Create a nested dictionary with specified structure and content.
    
    Creates a two-level nested dictionary where each outer key maps to
    a dictionary of inner keys, and each inner key maps to a copy of
    the provided content.
    
    Parameters
    ----------
    content : dict or any copyable object
        The content to populate at each leaf of the nested dictionary.
        Must have a .copy() method. Will be copied for each inner key to avoid aliasing.
    outer : list or iterable
        Keys for the outer level of the nested dictionary.
    inner : list or iterable
        Keys for the inner level of the nested dictionary.
        
    Returns
    -------
    dict
        Nested dictionary with structure {outer_key: {inner_key: content_copy}}.

    Raises
    ------
    AttributeError
        If content does not have a .copy() method.

    Notes
    -----
    - Duplicate keys in outer or inner iterables will overwrite previous values.
    - The content must have a .copy() method (e.g., dict, list, numpy array).
    - Primitive types (int, str, float) will raise AttributeError.
        
    Examples
    --------
    >>> content = {'value': 0, 'count': 0}
    >>> outer = ['A', 'B']
    >>> inner = ['x', 'y', 'z']
    >>> nested = populate_nested_dict(content, outer, inner)
    >>> nested['A']['x']
    {'value': 0, 'count': 0}
    >>> # Each entry is a separate copy
    >>> nested['A']['x']['value'] = 5
    >>> nested['B']['x']['value']
    0    """
    if not hasattr(content, 'copy'):
        raise AttributeError(f"Content of type {type(content).__name__} does not have a .copy() method")
    
    nested_dict = {o: {} for o in outer}
    for o in outer:
        nested_dict[o] = {i: content.copy() for i in inner}

    return nested_dict


def nested_dict_to_seq_of_tables(datadict, ordered_names1=None, ordered_names2=None):
    """Convert a nested dictionary to a sequence of 2D tables.
    
    Transforms a nested dictionary with structure {outer: {inner: {key: value}}}
    into a dictionary of 2D numpy arrays where rows correspond to outer keys
    and columns to inner keys.
    
    Parameters
    ----------
    datadict : dict
        Nested dictionary with three levels. Structure should be:
        {outer_key: {inner_key: {data_key: value}}}
    ordered_names1 : list or None, optional
        Ordered list of outer keys to use as row indices.
        If None, uses sorted outer keys. Default is None.
    ordered_names2 : list or None, optional
        Ordered list of inner keys to use as column indices.
        If None, uses sorted inner keys. Default is None.
        
    Returns
    -------
    dict
        Dictionary mapping data keys to 2D numpy arrays where:
        - Rows correspond to ordered_names1 (outer keys)
        - Columns correspond to ordered_names2 (inner keys)
        - Values are from the nested dictionary

    Raises
    ------
    ValueError
        If datadict is empty.
    IndexError
        If datadict structure is inconsistent.

    Notes
    -----
    - Missing values are filled with np.nan, not zeros.
    - Assumes all inner dictionaries have the same data keys.
    - Uses the first entry to determine the data key structure.
        
    Examples
    --------
    >>> data = {
    ...     'A': {'x': {'metric1': 1, 'metric2': 2},
    ...           'y': {'metric1': 3, 'metric2': 4}},
    ...     'B': {'x': {'metric1': 5, 'metric2': 6},
    ...           'y': {'metric1': 7, 'metric2': 8}}
    ... }
    >>> tables = nested_dict_to_seq_of_tables(data)
    >>> tables['metric1']
    array([[1., 3.],
           [5., 7.]])
    >>> # Rows are ['A', 'B'], columns are ['x', 'y']    """
    # Validate input
    if not datadict:
        raise ValueError("Cannot process empty dictionary")
    
    names1 = list(datadict.keys())
    if not names1:
        raise ValueError("Cannot process empty dictionary")
    
    names2 = list(datadict[names1[0]].keys())
    if not names2:
        raise ValueError("First level dictionary is empty")
    
    datakeys = list(datadict[names1[0]][names2[0]].keys())

    if ordered_names1 is None:
        ordered_names1 = sorted(names1)
    if ordered_names2 is None:
        ordered_names2 = sorted(names2)

    table_seq = {dkey: np.zeros((len(names1), len(names2))) for dkey in datakeys}
    for dkey in datakeys:
        for i, n1 in enumerate(ordered_names1):
            for j, n2 in enumerate(ordered_names2):
                try:
                    table_seq[dkey][i, j] = datadict[n1][n2][dkey]
                except KeyError:
                    table_seq[dkey][i, j] = np.nan

    return table_seq


def add_names_to_nested_dict(datadict, names1, names2):
    """Replace numeric keys in a nested dictionary with meaningful names.
    
    Takes a nested dictionary with integer keys at two levels and replaces
    them with provided names. Useful for converting indexed data structures
    to named structures for better readability and access.
    
    Parameters
    ----------
    datadict : dict
        Nested dictionary with integer keys at two levels. Expected structure
        is datadict[i][j] where i and j are integers starting from 0.
    names1 : list-like or None
        Names to use for the first level keys. If None, uses range(n1) where
        n1 is the number of first-level keys. Length must match the number
        of first-level keys in datadict.
    names2 : list-like or None
        Names to use for the second level keys. If None, uses range(n2) where
        n2 is the number of second-level keys. Length must match the number
        of second-level keys in datadict.
        
    Returns
    -------
    dict
        New nested dictionary with the same data but keys replaced by names.
        If both names1 and names2 are None, returns the original dict unchanged.
        Structure: renamed_dict[name1][name2] contains datadict[i][j].

    Raises
    ------
    KeyError
        If integer keys are not consecutive starting from 0.
    ValueError
        If names length doesn't match number of keys.
        
    Examples
    --------
    >>> data = {0: {0: {'value': 1}, 1: {'value': 2}}, 
    ...         1: {0: {'value': 3}, 1: {'value': 4}}}
    >>> names1 = ['row1', 'row2']
    >>> names2 = ['col1', 'col2']
    >>> renamed = add_names_to_nested_dict(data, names1, names2)
    >>> renamed['row1']['col1']
    {'value': 1}
    
    Notes
    -----
    - Requires consecutive integer keys starting from 0.
    - Uses .update() to merge inner dictionaries.
    - Returns original dict if both names are None.
    - Assumes all outer keys contain the same inner keys (e.g., if datadict[0] has keys [0,1,2],
      then datadict[1], datadict[2], etc. must also have keys [0,1,2]).    """
    if names1 is None and names2 is None:
        return datadict
    
    # renaming for convenience
    n1 = len(datadict.keys())
    n2 = len(datadict[list(datadict.keys())[0]])
    
    # Validate consecutive integer keys
    outer_keys = sorted(datadict.keys())
    if outer_keys != list(range(len(outer_keys))):
        raise KeyError(f"Outer keys must be consecutive integers from 0. Found: {outer_keys}")
    
    if names1 is None:
        names1 = range(n1)
    elif len(names1) != n1:
        raise ValueError(f"names1 length ({len(names1)}) must match number of outer keys ({n1})")
    
    if names2 is None:
        names2 = range(n2)
    elif len(names2) != n2:
        raise ValueError(f"names2 length ({len(names2)}) must match number of inner keys ({n2})")

    renamed_dict = populate_nested_dict(dict(), names1, names2)
    for i in range(n1):
        for j in range(n2):
            renamed_dict[names1[i]][names2[j]].update(datadict[i][j])
    return renamed_dict


def retrieve_relevant_from_nested_dict(
    nested_dict, target_key, target_value, operation="=", allow_missing_keys=False
):
    """Find all (outer_key, inner_key) pairs where a condition is met.
    
    Searches through a nested dictionary structure to find all locations where
    a specific key-value condition is satisfied. Useful for filtering nested
    data structures based on criteria.
    
    Parameters
    ----------
    nested_dict : dict
        Nested dictionary with structure {outer_key: {inner_key: data_dict}},
        where data_dict contains the key-value pairs to search.
    target_key : str
        The key to look for in the innermost dictionaries.
    target_value : any
        The value to compare against. Must be comparable with stored values
        when using ">" or "<" operations.
    operation : {"=", ">", "<"}, default="="
        Comparison operation to use:
        
        * "=" : Find entries where target_key equals target_value
        * ">" : Find entries where target_key is greater than target_value  
        * "<" : Find entries where target_key is less than target_value
    allow_missing_keys : bool, default=False
        If True, skip entries where target_key is missing instead of raising
        an error. Missing keys are treated as not matching any criteria.
        
    Returns
    -------
    list of tuples
        List of (outer_key, inner_key) tuples identifying locations where
        the condition is satisfied.
        
    Raises
    ------
    ValueError
        If target_key is not found and allow_missing_keys is False, or if
        an invalid operation is specified.
    TypeError
        If comparison operations ">" or "<" are used with incomparable types
        (will propagate from Python's comparison).
        
    Examples
    --------
    >>> data = {
    ...     'exp1': {'cell1': {'score': 0.8, 'type': 'A'},
    ...              'cell2': {'score': 0.6, 'type': 'B'}},
    ...     'exp2': {'cell1': {'score': 0.9, 'type': 'A'},
    ...              'cell2': {'score': 0.7, 'type': 'B'}}
    ... }
    >>> retrieve_relevant_from_nested_dict(data, 'score', 0.7, '>')
    [('exp1', 'cell1'), ('exp2', 'cell1')]
    >>> retrieve_relevant_from_nested_dict(data, 'type', 'A', '=')
    [('exp1', 'cell1'), ('exp2', 'cell1')]
    
    Notes
    -----
    For ">" and "<" operations:
    - Missing keys (when allow_missing_keys=True) are treated as not matching
    - None values are treated as not matching (since None comparisons would fail)  
    - Incomparable types (e.g., string vs number) will raise TypeError    """
    if operation not in ['=', '>', '<']:
        raise ValueError(f"Operation must be one of '=', '>', '<'. Got: {operation}")
    
    relevant_pairs = []
    for key1 in nested_dict.keys():
        for key2 in nested_dict[key1].keys():
            data = nested_dict[key1][key2]
            if target_key not in data and not allow_missing_keys:
                raise ValueError(f"Target key {target_key} not found in data dict")

            if operation == "=":
                criterion = data.get(target_key) == target_value
            elif operation == ">":
                criterion = (
                    data.get(target_key) > target_value
                    if data.get(target_key) is not None
                    else False
                )
            elif operation == "<":
                criterion = (
                    data.get(target_key) < target_value
                    if data.get(target_key) is not None
                    else False
                )
            else:
                raise ValueError(
                    f'Operation should be one of "=", ">", "<", but {operation} was passed'
                )

            if criterion:
                relevant_pairs.append((key1, key2))

    return relevant_pairs


def rescale(data):
    """Rescale 1D data to the range [0, 1] using min-max normalization.
    
    Applies min-max scaling to transform data linearly so that the minimum
    value becomes 0 and the maximum value becomes 1. Useful for normalizing
    time series or feature vectors to a common scale.
    
    Parameters
    ----------
    data : array-like
        Input data to rescale. Must be 1-dimensional.
        
    Returns
    -------
    ndarray
        Rescaled data with same length as input, values in [0, 1].
        If input min equals max, returns array of 0.5 values.
        
    Raises
    ------
    ValueError
        If input data has more than 1 dimension.
        
    Notes
    -----
    - Uses sklearn's MinMaxScaler internally. The transformation is:
      X_scaled = (X - X.min()) / (X.max() - X.min())
    - Constant arrays (where all values are equal) return 0.5.
    - NaN values are preserved in the output.
    - Single element arrays return 0.0.
    
    Examples
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> rescale(data)
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    
    >>> # Attempting to rescale 2D data raises an error
    >>> data2d = np.array([[1, 5], [2, 4]])
    >>> try:
    ...     rescale(data2d)
    ... except ValueError as e:
    ...     print(f"Error: {e}")
    Error: Input data must be 1-dimensional, got shape (2, 2)    """
    data = np.asarray(data)
    if data.ndim > 1:
        raise ValueError(f"Input data must be 1-dimensional, got shape {data.shape}")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    res = scaler.fit_transform(data.reshape(-1, 1)).ravel()
    return res


def get_hash(data):
    """Create a hash of numpy array or other data.

    Parameters
    ----------
    data : np.ndarray or other
        Data to hash. For arrays, uses the raw bytes.
        For other types, converts to string first.

    Returns
    -------
    str
        Hexadecimal hash string

    Notes
    -----
    - For numpy arrays: includes shape and dtype in hash, so same data with
      different shape produces different hashes.
    - For non-arrays: uses str() representation which may vary across Python
      versions and implementations.
    - Byte order affects the hash for arrays.
    - Uses SHA256 algorithm with UTF-8 encoding.

    Examples
    --------
    >>> arr = np.array([1, 2, 3])
    >>> hash1 = get_hash(arr)
    >>> arr_reshaped = arr.reshape(3, 1)
    >>> hash2 = get_hash(arr_reshaped)
    >>> hash1 == hash2  # Different shapes produce different hashes
    False    """
    if isinstance(data, np.ndarray):
        # For numpy arrays, use the raw bytes for consistent hashing
        # Include shape and dtype in the hash to distinguish reshaped arrays
        hash_id = hashlib.sha256()
        hash_id.update(data.shape.__repr__().encode("utf-8"))
        hash_id.update(data.dtype.str.encode("utf-8"))
        hash_id.update(data.tobytes())
        return hash_id.hexdigest()
    else:
        # For other data types, convert to string
        # This is less ideal but provides a fallback
        hash_id = hashlib.sha256()
        hash_id.update(str(data).encode("utf-8"))
        return hash_id.hexdigest()


def phase_synchrony(vec1, vec2):
    """Calculate instantaneous phase synchrony between two signals.
    
    Computes phase synchrony using the Hilbert transform to extract
    instantaneous phases, then measures phase coupling using a 
    sine-based metric that ranges from 0 (no synchrony) to 1 (perfect synchrony).
    
    Parameters
    ----------
    vec1 : array-like
        First signal, must be 1D array of same length as vec2.
    vec2 : array-like  
        Second signal, must be 1D array of same length as vec1.
        
    Returns
    -------
    ndarray
        Phase synchrony values at each time point, ranging from 0 to 1.
        Same length as input signals.

    Raises
    ------
    ValueError
        If input signals have different lengths.
        
    Notes
    -----
    The algorithm:
    1. Applies Hilbert transform to get analytic signals
    2. Extracts instantaneous phases using np.angle()
    3. Computes phase difference at each time point
    4. Maps to [0,1] using: 1 - sin(|Δφ|/2)
    
    This metric is 1 when phases are aligned (Δφ = 0) and 0 when
    maximally misaligned (Δφ = π).
    
    - Edge effects from Hilbert transform may affect boundary values.
    - Designed for real-valued signals.
    - NaN values propagate through the calculation.
    
    Examples
    --------
    >>> t = np.linspace(0, 1, 1000)
    >>> signal1 = np.sin(2 * np.pi * 10 * t)  # 10 Hz
    >>> signal2 = np.sin(2 * np.pi * 10 * t + np.pi/4)  # 10 Hz, phase shifted
    >>> sync = phase_synchrony(signal1, signal2)
    >>> round(np.mean(sync), 2)  # High synchrony despite phase shift
    0.62
    
    See Also
    --------
    :func:`scipy.signal.hilbert` :
        Hilbert transform used to extract phases.
    """
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    
    if vec1.shape != vec2.shape:
        raise ValueError(f"Input signals must have same shape. Got {vec1.shape} and {vec2.shape}")
    
    al1 = np.angle(hilbert(vec1), deg=False)
    al2 = np.angle(hilbert(vec2), deg=False)
    phase_sync = 1 - np.sin(np.abs(al1 - al2) / 2)
    return phase_sync



def correlation_matrix(A):
    """
    Compute Pearson correlation matrix between variables (rows).

    Parameters
    ----------
    A : numpy array of shape (n_variables, n_observations)
        Data matrix where each row is a variable

    Returns
    -------
    numpy array of shape (n_variables, n_variables)
        Correlation matrix

    Notes
    -----
    - Variables with zero variance (constant values) are handled by setting their
      correlation to 1.0 with themselves and NaN with other variables.
    - Single observation (n=1) returns NaN matrix as correlation is undefined.
    - Uses row-wise computation (variables are rows).

    Examples
    --------
    >>> A = np.array([[1, 2, 3], [4, 5, 6]])
    >>> corr = correlation_matrix(A)
    >>> corr.shape
    (2, 2)    """
    # Center the data
    am = A - np.mean(A, axis=1, keepdims=True)

    # Compute correlation matrix
    n = A.shape[1]
    if n > 1:
        # Compute covariance matrix
        cov = (am @ am.T) / (n - 1)

        # Compute standard deviations
        var_diag = np.diag(cov)
        stds = np.sqrt(var_diag)

        # Normalize to get correlation
        # Handle zero variance case
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = cov / np.outer(stds, stds)
            # Set diagonal to 1 for zero-variance variables
            np.fill_diagonal(corr, 1.0)

        return corr
    else:
        # Single observation - correlation is undefined
        return np.full((A.shape[0], A.shape[0]), np.nan)


def cross_correlation_matrix(A, B):
    """Compute cross-correlation matrix between two sets of variables.
    
    Computes Pearson correlations between variables (rows) in A and variables (rows) in B.

    Parameters
    ----------
    A : numpy array of shape (n_variables1, n_observations)
        First data matrix where each row is a variable
    B : numpy array of shape (n_variables2, n_observations)  
        Second data matrix where each row is a variable

    Returns
    -------
    numpy array of shape (n_variables1, n_variables2)
        Cross-correlation matrix where element [i,j] is the correlation
        between A[i,:] and B[j,:]

    Raises
    ------
    ValueError
        If A and B have different numbers of observations (columns).

    Notes
    -----
    - Uses row-wise computation (variables are rows), consistent with correlation_matrix.
    - Variables with zero variance will result in NaN correlations.
    - Centers data by row means.

    Examples
    --------
    >>> A = np.array([[1, 2, 3], [4, 5, 6]])  # 2 variables, 3 observations
    >>> B = np.array([[7, 8, 9], [10, 11, 12]])  # 2 variables, 3 observations
    >>> cross_corr = cross_correlation_matrix(A, B)
    >>> cross_corr.shape
    (2, 2)    """
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"A and B must have same number of observations. Got {A.shape[1]} and {B.shape[1]}")
    
    # Center the data by row means (like correlation_matrix)
    am = A - np.mean(A, axis=1, keepdims=True)
    bm = B - np.mean(B, axis=1, keepdims=True)
    
    # Compute cross-correlation
    n = A.shape[1]
    if n > 1:
        # Handle zero variance case
        with np.errstate(divide="ignore", invalid="ignore"):
            # Compute standard deviations
            std_a = np.sqrt(np.sum(am**2, axis=1, keepdims=True) / (n - 1))
            std_b = np.sqrt(np.sum(bm**2, axis=1, keepdims=True) / (n - 1))
            
            # Compute correlation
            corr = (am @ bm.T) / (n - 1) / (std_a @ std_b.T)
            
        return corr
    else:
        # Single observation - correlation is undefined
        return np.full((A.shape[0], B.shape[0]), np.nan)


def norm_cross_corr(a, b, mode='full'):
    """Compute normalized cross-correlation between two signals.
    
    This function computes the normalized cross-correlation between two signals.
    Each overlapping window is normalized independently to have zero mean and 
    unit variance, making the correlation insensitive to amplitude scaling and 
    DC offset.
    
    Parameters
    ----------
    a : array_like
        First signal
    b : array_like
        Second signal
    mode : {'full', 'valid', 'same'}, optional
        Mode parameter controlling output size. Default is 'full'.
        - 'full': returns correlation at all lags (length: len(a) + len(b) - 1)
        - 'valid': returns only correlations without zero-padding (length: max(len(a) - len(b) + 1, 1))
        - 'same': returns correlation of same length as first input (length: len(a))
        
    Returns
    -------
    np.ndarray
        Normalized cross-correlation values. Values are in the range [-1, 1].
        The lag index can be computed as: lag = index - (len(a) - 1) for 'full' mode.

    Raises
    ------
    ValueError
        If mode is not one of 'full', 'valid', or 'same'.
        If either input signal is empty.
        
    Notes
    -----
    The normalization is performed on each overlapping window separately, ensuring 
    that correlation values are in the range [-1, 1], where:
    - 1 indicates perfect positive correlation
    - -1 indicates perfect negative correlation (anti-correlation)
    - 0 indicates no correlation
    
    For constant signals (zero variance), the function returns zeros.
    
    Examples
    --------
    >>> # Detect time shift between signals
    >>> signal = np.sin(np.linspace(0, 4*np.pi, 100))
    >>> shifted = np.roll(signal, 10)  # Shift by 10 samples
    >>> corr = norm_cross_corr(signal, shifted, mode='full')
    >>> lag = np.argmax(corr) - (len(signal) - 1)  # Should be close to -10    """
    if mode not in ['full', 'valid', 'same']:
        raise ValueError(f"mode must be one of 'full', 'valid', or 'same'. Got: {mode}")
    
    # Convert to numpy arrays
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    
    # Check for empty signals
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Input signals cannot be empty")
    
    # Handle constant signals (zero variance)
    a_std = np.std(a)
    b_std = np.std(b)
    
    if a_std < 1e-10 or b_std < 1e-10:
        # Return zeros for constant signals
        if mode == 'full':
            return np.zeros(len(a) + len(b) - 1)
        elif mode == 'valid':
            return np.zeros(max(len(a) - len(b) + 1, 1))
        else:  # 'same'
            return np.zeros(max(len(a), len(b)))
    
    # For normalized cross-correlation, we need to normalize each window
    len_a = len(a)
    len_b = len(b)
    
    if mode == 'full':
        # For full mode, slide one signal across the other
        n = len_a + len_b - 1
        result = np.zeros(n)
        
        # Pad signal a for easier indexing
        a_padded = np.concatenate([np.zeros(len_b - 1), a, np.zeros(len_b - 1)])
        
        for i in range(n):
            # Extract overlapping portions
            start = i
            end = min(i + len_b, len(a_padded))
            
            if end > start:
                a_window = a_padded[start:end]
                b_window = b[:end-start]
                
                # Only compute if we have a valid window
                if len(a_window) > 0 and len(b_window) > 0:
                    # Normalize each window
                    a_mean = np.mean(a_window)
                    b_mean = np.mean(b_window)
                    a_centered = a_window - a_mean
                    b_centered = b_window - b_mean
                    
                    # Compute normalized correlation
                    numerator = np.sum(a_centered * b_centered)
                    denominator = np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2))
                    
                    if denominator > 1e-10:
                        result[i] = numerator / denominator
    
    elif mode == 'valid':
        # For valid mode, all windows have the same size
        if len_a >= len_b:
            result = np.zeros(len_a - len_b + 1)
            for i in range(len_a - len_b + 1):
                a_window = a[i:i + len_b]
                
                # Normalize each window
                a_mean = np.mean(a_window)
                b_mean = np.mean(b)
                a_centered = a_window - a_mean
                b_centered = b - b_mean
                
                numerator = np.sum(a_centered * b_centered)
                denominator = np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2))
                
                if denominator > 1e-10:
                    result[i] = numerator / denominator
        else:
            result = np.zeros(1)
    
    else:  # mode == 'same'
        # For same mode, return central part of full correlation
        full_result = norm_cross_corr(a, b, mode='full')
        # Extract the central part
        if len_a >= len_b:
            start_idx = (len_b - 1) // 2
            result = full_result[start_idx:start_idx + len_a]
        else:
            start_idx = (len_a - 1) // 2
            result = full_result[start_idx:start_idx + len_b]
    
    return result


def to_numpy_array(data):
    """Convert various data types to numpy array.
    
    Handles numpy arrays, sparse matrices, and other array-like objects.
    Warning: Converting large sparse matrices can cause memory issues.
    
    Parameters
    ----------
    data : array_like, sparse matrix, or any object
        Input data to convert to numpy array. Can be:
        - numpy array (returned as-is)
        - scipy sparse matrix (converted to dense)
        - list, tuple, or other array-like object
    
    Returns
    -------
    numpy.ndarray
        Dense numpy array representation of the input data.
        
    Warnings
    --------
    Converting large sparse matrices to dense arrays can cause memory issues.
    Consider the memory implications before converting sparse data.
    
    Examples
    --------
    >>> import scipy.sparse as sp
    >>> sparse_data = sp.csr_matrix([[1, 0, 0], [0, 2, 0]])
    >>> dense = to_numpy_array(sparse_data)
    >>> dense.toarray() if hasattr(dense, 'toarray') else dense
    array([[1, 0, 0],
           [0, 2, 0]])    """
    if ssp.issparse(data):
        return data.toarray()
    else:
        return np.asarray(data)


def remove_outliers(data, method='zscore', threshold=3.0, quantile_range=(0.05, 0.95)):
    """Remove outliers from data using various detection strategies.
    
    Parameters
    ----------
    data : np.ndarray
        1D array of data points.
    method : str, default='zscore'
        Outlier detection method:
        - 'zscore': Remove points beyond threshold standard deviations from mean
        - 'iqr': Interquartile range method (1.5*IQR beyond Q1/Q3)
        - 'mad': Median absolute deviation method
        - 'quantile': Remove points outside specified quantile range
        - 'isolation': Local outlier factor based on density
    threshold : float, default=3.0
        Detection threshold:
        - For 'zscore': number of standard deviations
        - For 'iqr': multiplier for IQR (typically 1.5)
        - For 'mad': number of median absolute deviations
        - For 'isolation': contamination fraction (0-0.5)
    quantile_range : tuple, default=(0.05, 0.95)
        For 'quantile' method: (lower_quantile, upper_quantile)
        
    Returns
    -------
    inlier_indices : np.ndarray
        Indices of non-outlier points.
    clean_data : np.ndarray
        Data with outliers removed.

    Raises
    ------
    ValueError
        If method is not recognized.
    ImportError
        If 'isolation' method is used but scikit-learn is not installed.
        
    Notes
    -----
    - Input data is flattened with ravel().
    - For constant data (zero variance), all points are considered inliers.
    - MAD method uses scaling factor 1.4826 for consistency with normal distribution.
    - Isolation method uses random_state=42 for reproducibility.
        
    Examples
    --------
    >>> data = np.array([1, 2, 3, 100, 4, 5])  # 100 is outlier
    >>> # Z-score method
    >>> indices, cleaned = remove_outliers(data, method='zscore', threshold=2)
    >>> cleaned
    array([1, 2, 3, 4, 5])
    
    >>> # IQR method
    >>> indices, cleaned = remove_outliers(data, method='iqr', threshold=1.5)
    
    >>> # Quantile method
    >>> indices, cleaned = remove_outliers(data, method='quantile', quantile_range=(0.1, 0.9))    """
    data = np.asarray(data).ravel()
    n = len(data)
    
    if method == 'zscore':
        # Classical z-score method
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            # All values are identical
            inlier_mask = np.ones(n, dtype=bool)
        else:
            z_scores = np.abs((data - mean) / std)
            inlier_mask = z_scores < threshold
        
    elif method == 'iqr':
        # Interquartile range method
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        inlier_mask = (data >= lower_bound) & (data <= upper_bound)
        
    elif method == 'mad':
        # Median Absolute Deviation method (robust to outliers)
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        # Scale MAD to be consistent with standard deviation
        mad_scaled = 1.4826 * mad
        if mad_scaled == 0:
            # All values are identical
            inlier_mask = np.ones(n, dtype=bool)
        else:
            modified_z_scores = np.abs((data - median) / mad_scaled)
            inlier_mask = modified_z_scores < threshold
            
    elif method == 'quantile':
        # Simple quantile-based method
        lower_bound = np.percentile(data, quantile_range[0] * 100)
        upper_bound = np.percentile(data, quantile_range[1] * 100)
        inlier_mask = (data >= lower_bound) & (data <= upper_bound)
        
    elif method == 'isolation':
        # Isolation Forest for anomaly detection
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError:
            raise ImportError(
                "IsolationForest requires scikit-learn. "
                "Install it with: pip install scikit-learn"
            )
            
        # Reshape for sklearn
        data_reshaped = data.reshape(-1, 1)
        iso_forest = IsolationForest(
            contamination=min(threshold, 0.5),  # threshold is contamination rate
            random_state=42
        )
        predictions = iso_forest.fit_predict(data_reshaped)
        inlier_mask = predictions == 1
        
    else:
        raise ValueError(
            f"Unknown outlier detection method: {method}. "
            f"Choose from: 'zscore', 'iqr', 'mad', 'quantile', 'isolation'"
        )
    
    inlier_indices = np.where(inlier_mask)[0]
    clean_data = data[inlier_mask]
    
    return inlier_indices, clean_data


def write_dict_to_hdf5(data, hdf5_file, group_name=""):
    """
    Recursively writes a dictionary to an HDF5 file.

    Parameters
    ----------
    data : dict
        The dictionary to write. Can contain nested dictionaries, lists,
        numpy arrays, or scalar values.
    hdf5_file : str
        The path to the HDF5 file.
    group_name : str, default=""
        The name of the current group in the HDF5 file.

    Notes
    -----
    - Opens file in append mode ('a'), which can overwrite existing data.
    - Converts all numeric data to float64 (may cause precision loss/gain).
    - Lists are converted to numpy arrays before storage.
    - Non-numeric/non-array values are stored as attributes.
    - No protection against circular references (infinite recursion).

    Warnings
    --------
    - All numeric data is cast to float64, which may not preserve original dtype.
    - Opening in append mode can corrupt existing file structure.

    Side Effects
    ------------
    Writes to the file system.

    Examples
    --------
    >>> # Example usage:
    >>> # data = {'group1': {'array': np.array([1, 2, 3]), 'value': 42}}
    >>> # write_dict_to_hdf5(data, 'output.h5')    """
    with h5py.File(hdf5_file, "a") as f:
        # Create a new group or get existing one
        group = f.create_group(group_name) if group_name else f

        for key, value in data.items():
            if isinstance(value, dict):
                # If the value is a dictionary, recurse into it
                write_dict_to_hdf5(value, hdf5_file, f"{group_name}/{key}")
            elif isinstance(value, list):
                # If the value is a list, convert it to a numpy array and store it
                group.create_dataset(key, data=np.array(value).astype(np.float64))
            elif isinstance(value, np.ndarray):
                # If the value is already a numpy array, store it directly
                group.create_dataset(key, data=value.astype(np.float64))
            else:
                # Otherwise, store it as an attribute (string or number)
                group.attrs[key] = value


def read_hdf5_to_dict(hdf5_file):
    """
    Reads an HDF5 file and converts it into a nested dictionary.

    Parameters
    ----------
    hdf5_file : str
        The path to the HDF5 file.

    Returns
    -------
    dict
        A nested dictionary representing the contents of the HDF5 file.

    Notes
    -----
    - Uses [()] syntax to read full datasets into memory.
    - Group attributes can overwrite dataset/group keys if names conflict.
    - Large datasets are loaded entirely into memory.

    Examples
    --------
    >>> # Example usage (requires existing HDF5 file):
    >>> # data_dict = read_hdf5_to_dict('input.h5')
    >>> # data_dict['group1']['array']
    >>> # array([1, 2, 3])    """

    def _read_group(group):
        """
        Recursively reads an HDF5 group and converts it to a dictionary.

        Parameters
        ----------
        group : h5py.Group
            The HDF5 group to read.

        Returns
        -------
        dict
            A dictionary representation of the group.
        """
        data = {}

        # Iterate over all items in the group
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                # If the item is a group, recurse into it
                data[key] = _read_group(item)
            elif isinstance(item, h5py.Dataset):
                # If the item is a dataset, convert it to a numpy array or list
                data[key] = item[()]
            else:
                # Handle attributes
                data[key] = item.attrs

        # Add attributes of the group itself
        for attr_key in group.attrs:
            data[attr_key] = group.attrs[attr_key]

        return data

    with h5py.File(hdf5_file, "r") as f:
        return _read_group(f)


def check_nonnegative(**kwargs):
    """Check that all provided parameters are non-negative.
    
    Validates that numeric parameters are >= 0. Useful for input validation
    in functions that require non-negative values (counts, rates, probabilities).
    
    Parameters
    ----------
    **kwargs
        Parameter name to value mappings. All values should be numeric.
        
    Raises
    ------
    ValueError
        If any parameter value is negative, NaN, or infinite.
        Error message includes parameter name and value.
        
    Examples
    --------
    >>> check_nonnegative(n_neurons=10, rate=0.5)  # No error
    
    >>> check_nonnegative(n_neurons=10, rate=-0.5)
    Traceback (most recent call last):
        ...
    ValueError: rate must be non-negative, got -0.5
    
    >>> import numpy as np
    >>> check_nonnegative(count=5, prob=np.nan)
    Traceback (most recent call last):
        ...
    ValueError: prob cannot be NaN    """
    for name, value in kwargs.items():
        if value is None:
            continue  # Skip None values
        try:
            val = float(value)
            if np.isnan(val):
                raise ValueError(f"{name} cannot be NaN")
            if np.isinf(val):
                raise ValueError(f"{name} cannot be infinite")
            if val < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        except (TypeError, ValueError) as e:
            if "cannot be" in str(e) or "must be" in str(e):
                raise  # Re-raise our validation errors
            raise TypeError(f"{name} must be numeric, got {type(value).__name__}")


def check_unit(left_open=False, right_open=False, **kwargs):
    """Check that all provided parameters are in the unit interval.
    
    Validates that numeric parameters are within [0, 1] with configurable
    bound inclusion. Useful for probabilities, fractions, and normalized values.
    
    Parameters
    ----------
    left_open : bool, optional
        If True, left bound is open (0, 1]. If False, closed [0, 1]. Default: False.
    right_open : bool, optional
        If True, right bound is open [0, 1). If False, closed [0, 1]. Default: False.
    **kwargs
        Parameter name to value mappings. All values should be numeric in [0, 1].
        
    Raises
    ------
    ValueError
        If any parameter value is outside the unit interval, NaN, or infinite.
        Error message includes parameter name, value, and expected range.
        
    Examples
    --------
    >>> check_unit(probability=0.5, fraction=0.8)  # No error
    
    >>> check_unit(probability=1.5)
    Traceback (most recent call last):
        ...
    ValueError: probability must be in [0, 1], got 1.5
    
    >>> check_unit(left_open=True, rate=0.0)
    Traceback (most recent call last):
        ...
    ValueError: rate must be in (0, 1], got 0.0
    
    >>> check_unit(left_open=True, right_open=True, value=0.5)  # No error    """
    # Determine bounds description
    left_bracket = "(" if left_open else "["
    right_bracket = ")" if right_open else "]"
    bounds_desc = f"{left_bracket}0, 1{right_bracket}"
    
    for name, value in kwargs.items():
        if value is None:
            continue  # Skip None values
        try:
            val = float(value)
            if np.isnan(val):
                raise ValueError(f"{name} cannot be NaN")
            if np.isinf(val):
                raise ValueError(f"{name} cannot be infinite")
            
            # Check bounds
            if left_open and val <= 0:
                raise ValueError(f"{name} must be in {bounds_desc}, got {value}")
            elif not left_open and val < 0:
                raise ValueError(f"{name} must be in {bounds_desc}, got {value}")
                
            if right_open and val >= 1:
                raise ValueError(f"{name} must be in {bounds_desc}, got {value}")
            elif not right_open and val > 1:
                raise ValueError(f"{name} must be in {bounds_desc}, got {value}")
                
        except (TypeError, ValueError) as e:
            if "cannot be" in str(e) or "must be" in str(e):
                raise  # Re-raise our validation errors
            raise TypeError(f"{name} must be numeric, got {type(value).__name__}")


def check_positive(**kwargs):
    """Check that all provided parameters are positive (> 0).
    
    Validates that numeric parameters are strictly positive. Useful for input 
    validation in functions that require positive values (dimensions, sizes).
    
    Parameters
    ----------
    **kwargs
        Parameter name to value mappings. All values should be numeric.
        
    Raises
    ------
    ValueError
        If any parameter value is not positive, NaN, or infinite.
        Error message includes parameter name and value.
        
    Examples
    --------
    >>> check_positive(n_neurons=10, dim=5)  # No error
    
    >>> check_positive(n_neurons=0)
    Traceback (most recent call last):
        ...
    ValueError: n_neurons must be positive, got 0
    
    >>> check_positive(dim=-5)
    Traceback (most recent call last):
        ...
    ValueError: dim must be positive, got -5    """
    for name, value in kwargs.items():
        if value is None:
            continue  # Skip None values
        try:
            val = float(value)
            if np.isnan(val):
                raise ValueError(f"{name} cannot be NaN")
            if np.isinf(val):
                raise ValueError(f"{name} cannot be infinite")
            if val <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        except (TypeError, ValueError) as e:
            if "cannot be" in str(e) or "must be" in str(e):
                raise  # Re-raise our validation errors
            raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
