"""I/O utilities for INTENSE results."""
import json
import sys

import numpy as np


def _numpy_to_python(obj):
    """Convert numpy types to Python native types for JSON serialization.

    Recursively converts numpy scalars, arrays, and nested containers to
    JSON-serializable Python types.

    Parameters
    ----------
    obj : any
        Object to convert. Can be numpy scalar, array, dict, list, or native Python type.

    Returns
    -------
    any
        JSON-serializable Python object.
    """
    if obj is None:
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_numpy_to_python(v) for v in obj]
    return obj


class IntenseResults:
    """
    Container for INTENSE computation results.

    Attributes
    ----------
    info : dict
        Metadata about the computation (optimal delays, thresholds, etc.).
    intense_params : dict
        Parameters used for the INTENSE computation.
    stats : dict
        Statistical results (p-values, metric values, etc.).
    significance : dict
        Significance test results for each neuron-feature pair.

    Methods
    -------
    update(property_name, data)
        Add or update a property with data.
    update_multiple(datadict)
        Update multiple properties from a dictionary.

    Examples
    --------
    >>> # Create results container and add analysis outputs
    >>> results = IntenseResults()
    >>> # Add statistical results
    >>> results.update('stats', {'neuron1': {'feature1': {'me': 0.5, 'pval': 0.01}}})
    >>> # Add computation metadata
    >>> results.update('info', {'optimal_delays': [[0, 5], [10, 0]],
    ...                        'n_shuffles': 1000})
    >>> # Access stored data
    >>> results.stats['neuron1']['feature1']['me']
    0.5
    """

    def __init__(self):
        """
        Initialize an empty IntenseResults container.

        Creates an IntenseResults object with no initial data. Properties are added
        dynamically using the update() or update_multiple() methods.

        Notes
        -----
        The IntenseResults class serves as a flexible container for storing INTENSE
        computation outputs. It allows dynamic addition of properties to accommodate
        different analysis configurations and results.

        Common properties added during INTENSE analysis:
        - 'stats': Statistical test results (p-values, metric values)
        - 'significance': Binary significance indicators
        - 'info': Computation metadata (delays, parameters used)
        - 'intense_params': Parameters used for the computation

        See Also
        --------
        ~driada.intense.pipelines.compute_cell_feat_significance : Main function that returns IntenseResults
        ~driada.intense.io.save_results : Method to persist results to disk
        """
        pass

    def update(self, property_name, data):
        """Add or update a property with data.

        Stores analysis results as attributes of the IntenseResults object,
        allowing flexible storage of various data types and structures.

        Parameters
        ----------
        property_name : str
            Name of the property to store. Will become an attribute of the
            object accessible via dot notation.
        data : any
            Data to store. Can be any Python object: arrays, dictionaries,
            dataframes, custom objects, etc.

        Examples
        --------
        >>> # Store different types of analysis results
        >>> import numpy as np
        >>> results = IntenseResults()
        >>> # Add mutual information matrix
        >>> results.update('mi_matrix', np.array([[0, 0.5], [0.5, 0]]))
        >>> # Add list of significant neuron-feature pairs
        >>> results.update('significant_pairs', [(0, 1), (2, 3)])
        >>> # Access via attribute notation
        >>> results.mi_matrix
        array([[0. , 0.5],
               [0.5, 0. ]])
        >>> results.significant_pairs
        [(0, 1), (2, 3)]

        Notes
        -----
        Property names should be valid Python identifiers. Existing properties
        will be overwritten without warning.
        """
        setattr(self, property_name, data)

    def update_multiple(self, datadict):
        """Update multiple properties from a dictionary.

        Batch update of multiple properties at once, useful for storing
        related analysis results together.

        Parameters
        ----------
        datadict : dict
            Dictionary mapping property names to data values. Each key-value
            pair will be stored as an attribute.

        Examples
        --------
        >>> # Batch update multiple analysis results at once
        >>> import numpy as np
        >>> results = IntenseResults()
        >>> # Add multiple related results together
        >>> results.update_multiple({
        ...     'mi_values': np.array([0.1, 0.5, 0.3]),
        ...     'p_values': np.array([0.05, 0.001, 0.02]),
        ...     'significant': np.array([False, True, True]),
        ...     'parameters': {'metric': 'mi', 'correction': 'fdr'}
        ... })
        >>> # All properties are now accessible
        >>> results.mi_values
        array([0.1, 0.5, 0.3])
        >>> results.significant
        array([False,  True,  True])

        See Also
        --------
        ~driada.intense.io.IntenseResults.update : Add single property
        """
        for dname, data in datadict.items():
            setattr(self, dname, data)

    def validate_against_ground_truth(self, ground_truth, verbose=True):
        """
        Compare INTENSE detections against known ground truth.

        Validates the analysis results against a ground truth dictionary,
        typically generated by generate_tuned_selectivity_exp(). Computes
        sensitivity, precision, F1 score, and per-type detection rates.

        Parameters
        ----------
        ground_truth : dict
            Ground truth from generate_tuned_selectivity_exp(). Must contain:
            - "expected_pairs" : list of (neuron_idx, feature_name) tuples
            - "neuron_types" : dict mapping neuron_idx to group name (optional)
        verbose : bool, optional
            Print detailed results. Default: True.

        Returns
        -------
        metrics : dict
            Validation metrics containing:
            - "true_positives" : int - Number of correctly detected pairs
            - "false_positives" : int - Number of spurious detections
            - "false_negatives" : int - Number of missed pairs
            - "sensitivity" : float - TP / (TP + FN)
            - "precision" : float - TP / (TP + FP)
            - "f1" : float - Harmonic mean of sensitivity and precision
            - "type_stats" : dict - Per-neuron-type statistics
            - "tp_pairs" : set - True positive (neuron, feature) pairs
            - "fp_pairs" : set - False positive pairs
            - "fn_pairs" : set - False negative pairs

        Notes
        -----
        This method requires that the IntenseResults object has a 'significance'
        attribute populated with neuron-feature significance results.

        Examples
        --------
        >>> # After running INTENSE analysis
        >>> results = IntenseResults()
        >>> # ... populate with analysis results ...
        >>> ground_truth = {"expected_pairs": [(0, "hd"), (1, "x")],
        ...                 "neuron_types": {0: "hd_cell", 1: "place_cell"}}
        >>> # metrics = results.validate_against_ground_truth(ground_truth)

        See Also
        --------
        generate_tuned_selectivity_exp : Generates experiments with ground truth
        """
        if not hasattr(self, "significance"):
            raise ValueError(
                "IntenseResults has no 'significance' attribute. "
                "Run INTENSE analysis first."
            )

        # Extract detected pairs from significance results
        # Check for both 'stage2' (two-stage mode) and 'criterion' (single-stage)
        detected_pairs = set()
        for neuron_id in self.significance:
            for feat_name in self.significance[neuron_id]:
                sig_entry = self.significance[neuron_id][feat_name]
                # Two-stage mode uses 'stage2', single-stage uses 'criterion'
                is_significant = sig_entry.get("stage2") or sig_entry.get("criterion")
                if is_significant:
                    detected_pairs.add((neuron_id, feat_name))

        expected_pairs = set(ground_truth["expected_pairs"])

        # Calculate metrics
        true_positives = detected_pairs & expected_pairs
        false_positives = detected_pairs - expected_pairs
        false_negatives = expected_pairs - detected_pairs

        n_tp = len(true_positives)
        n_fp = len(false_positives)
        n_fn = len(false_negatives)

        sensitivity = n_tp / max(n_tp + n_fn, 1)
        precision = n_tp / max(n_tp + n_fp, 1)
        f1 = 2 * sensitivity * precision / max(sensitivity + precision, 1e-10)

        # Per-neuron-type breakdown
        type_stats = {}
        neuron_types = ground_truth.get("neuron_types", {})
        if neuron_types:
            # Get unique neuron types (excluding nonselective)
            unique_types = set(neuron_types.values())
            for neuron_type in unique_types:
                if "nonselective" in neuron_type.lower():
                    continue
                type_expected = [
                    p for p in expected_pairs
                    if neuron_types.get(p[0]) == neuron_type
                ]
                type_detected = [
                    p for p in true_positives
                    if neuron_types.get(p[0]) == neuron_type
                ]
                type_stats[neuron_type] = {
                    "expected": len(type_expected),
                    "detected": len(type_detected),
                    "sensitivity": len(type_detected) / max(len(type_expected), 1),
                }

        metrics = {
            "true_positives": n_tp,
            "false_positives": n_fp,
            "false_negatives": n_fn,
            "sensitivity": sensitivity,
            "precision": precision,
            "f1": f1,
            "type_stats": type_stats,
            "tp_pairs": true_positives,
            "fp_pairs": false_positives,
            "fn_pairs": false_negatives,
        }

        if verbose:
            print("\n" + "=" * 60)
            print("GROUND TRUTH VALIDATION")
            print("=" * 60)
            print(f"\nOverall Performance:")
            print(f"  True Positives:  {n_tp}")
            print(f"  False Positives: {n_fp}")
            print(f"  False Negatives: {n_fn}")
            print(f"  Sensitivity:     {sensitivity:.1%}")
            print(f"  Precision:       {precision:.1%}")
            print(f"  F1 Score:        {f1:.1%}")

            if type_stats:
                print(f"\nPer-Type Detection Rates:")
                for neuron_type, stats in sorted(type_stats.items()):
                    print(
                        f"  {neuron_type:15s}: {stats['detected']}/{stats['expected']} "
                        f"({stats['sensitivity']:.0%})"
                    )

            if false_positives:
                print(f"\nFalse Positives (unexpected detections):")
                for neuron_id, feat in sorted(false_positives)[:5]:
                    ntype = neuron_types.get(neuron_id, "unknown")
                    print(f"  Neuron {neuron_id} ({ntype}) -> {feat}")
                if len(false_positives) > 5:
                    print(f"  ... and {len(false_positives) - 5} more")

            if false_negatives:
                print(f"\nFalse Negatives (missed detections):")
                for neuron_id, feat in sorted(false_negatives)[:5]:
                    ntype = neuron_types.get(neuron_id, "unknown")
                    print(f"  Neuron {neuron_id} ({ntype}) -> {feat}")
                if len(false_negatives) > 5:
                    print(f"  ... and {len(false_negatives) - 5} more")

        return metrics

    def memory_usage(self):
        """
        Return memory usage breakdown in bytes.

        Analyzes the memory consumption of all stored data in the IntenseResults
        object, providing a detailed breakdown by attribute. Useful for diagnosing
        memory issues and verifying that memory optimizations (like store_random_shifts=False)
        are working as expected.

        Returns
        -------
        usage : dict
            Dictionary mapping attribute names to their memory usage in bytes.
            Keys include:
            - "info.{key}": Memory for each numpy array in the info dict
            - "info.{key}": Memory for DataFrames in info (sum of all columns)
            - "stats": Approximate memory for stats dict
            - "significance": Approximate memory for significance dict

        Notes
        -----
        - For numpy arrays, uses the .nbytes attribute for accurate measurement
        - For pandas DataFrames, uses memory_usage(deep=True) for accurate measurement
        - For other objects, uses sys.getsizeof() which may underestimate nested structures
        - The "random_shifts1" and "random_shifts2" arrays are the largest consumers
          when store_random_shifts=True

        Examples
        --------
        >>> from driada.intense import IntenseResults
        >>> import numpy as np
        >>> results = IntenseResults()
        >>> results.update('info', {
        ...     'me_total1': np.zeros((10, 5, 101)),
        ...     'me_total2': np.zeros((10, 5, 10001))
        ... })
        >>> usage = results.memory_usage()
        >>> 'info.me_total1' in usage
        True
        >>> usage['info.me_total1']
        40400
        """
        usage = {}

        if hasattr(self, 'info') and isinstance(self.info, dict):
            for key, val in self.info.items():
                if isinstance(val, np.ndarray):
                    usage[f"info.{key}"] = val.nbytes
                elif isinstance(val, list) and val:
                    # Check if list of DataFrames
                    if hasattr(val[0], 'memory_usage'):
                        total = sum(df.memory_usage(deep=True).sum() for df in val)
                        usage[f"info.{key}"] = int(total)
                    else:
                        usage[f"info.{key}"] = sys.getsizeof(val)
                else:
                    usage[f"info.{key}"] = sys.getsizeof(val)

        if hasattr(self, 'stats'):
            usage["stats"] = sys.getsizeof(self.stats)

        if hasattr(self, 'significance'):
            usage["significance"] = sys.getsizeof(self.significance)

        if hasattr(self, 'intense_params'):
            usage["intense_params"] = sys.getsizeof(self.intense_params)

        return usage


def save_results(results, fname, compressed=False):
    """Save IntenseResults to NPZ file.

    Arrays stored as NPZ arrays, dicts embedded as JSON strings.
    Uncompressed is fastest; compressed saves ~10x disk space.

    Parameters
    ----------
    results : IntenseResults
        Results object from INTENSE analysis.
    fname : str or Path
        Output file path (should end with .npz).
    compressed : bool, default False
        If True, use zlib compression (slower but 10x smaller files).

    Examples
    --------
    >>> from driada.intense.io import IntenseResults, save_results, load_results
    >>> results = IntenseResults()
    >>> results.update('stats', {'cell1': {'feat1': {'me': 0.5}}})
    >>> save_results(results, 'test.npz')
    >>> loaded = load_results('test.npz')
    >>> loaded.stats['cell1']['feat1']['me']
    0.5
    """
    arrays = {}

    # 1. Extract arrays from info
    info_scalars = {}
    if hasattr(results, 'info') and results.info:
        for key, val in results.info.items():
            if isinstance(val, np.ndarray):
                # Use sparse float16 for large MI arrays (me_total1, me_total2)
                if key.startswith('me_total') and val.ndim == 3:
                    # Store only non-NaN slices (neuron, feature pairs)
                    val_f16 = val.astype(np.float16)
                    # Check which (neuron, feature) slices have any non-NaN data
                    valid_mask = ~np.all(np.isnan(val_f16), axis=2)
                    valid_indices = np.argwhere(valid_mask).astype(np.int16)
                    valid_data = val_f16[valid_mask]
                    arrays[f'info_{key}_indices'] = valid_indices
                    arrays[f'info_{key}_data'] = valid_data
                    arrays[f'info_{key}_shape'] = np.array(val.shape, dtype=np.int32)
                else:
                    arrays[f'info_{key}'] = val
            elif isinstance(val, list) and len(val) > 0 and hasattr(val[0], 'values'):
                # List of DataFrames - store values only (stage stats)
                for i, df in enumerate(val):
                    arrays[f'info_{key}_{i}'] = df.values
            else:
                info_scalars[key] = val

    # 2. Extract disentanglement arrays
    disent_json = {}
    if hasattr(results, 'disentanglement') and results.disentanglement:
        disent = results.disentanglement
        for key in ['feat_feat_significance', 'disent_matrix', 'count_matrix']:
            if key in disent and disent[key] is not None:
                arrays[f'disent_{key}'] = disent[key]
        disent_json = {
            'feature_names': disent.get('feature_names', []),
            'summary': _numpy_to_python(disent.get('summary', {})),
        }

    # 3. Embed dicts as JSON strings in NPZ
    if hasattr(results, 'stats') and results.stats:
        arrays['_stats_json'] = np.array([json.dumps(_numpy_to_python(results.stats))])

    if hasattr(results, 'significance') and results.significance:
        arrays['_significance_json'] = np.array([json.dumps(_numpy_to_python(results.significance))])

    params_data = {
        'intense_params': _numpy_to_python(getattr(results, 'intense_params', {})),
        'info_scalars': _numpy_to_python(info_scalars),
    }
    arrays['_params_json'] = np.array([json.dumps(params_data)])

    if disent_json:
        arrays['_disentanglement_json'] = np.array([json.dumps(disent_json)])

    # 4. Save (compressed or uncompressed)
    if compressed:
        np.savez_compressed(fname, **arrays)
    else:
        np.savez(fname, **arrays)


def load_results(fname):
    """Load IntenseResults from NPZ file.

    Parameters
    ----------
    fname : str or Path
        Path to the NPZ file.

    Returns
    -------
    IntenseResults
        Reconstructed results object.

    Examples
    --------
    >>> from driada.intense.io import IntenseResults, save_results, load_results
    >>> results = IntenseResults()
    >>> results.update('stats', {'cell1': {'feat1': {'me': 0.5}}})
    >>> save_results(results, 'test.npz')
    >>> loaded = load_results('test.npz')
    >>> loaded.stats['cell1']['feat1']['me']
    0.5
    """
    data = np.load(fname, allow_pickle=True)
    results = IntenseResults()

    # 1. Load JSON-embedded dicts
    if '_stats_json' in data:
        results.update('stats', json.loads(str(data['_stats_json'][0])))

    if '_significance_json' in data:
        results.update('significance', json.loads(str(data['_significance_json'][0])))

    if '_params_json' in data:
        params = json.loads(str(data['_params_json'][0]))
        results.update('intense_params', params.get('intense_params', {}))
        info_scalars = params.get('info_scalars', {})
    else:
        info_scalars = {}

    # 2. Reconstruct info dict with arrays
    info = dict(info_scalars)

    # First pass: identify sparse arrays (me_total with _indices, _data, _shape)
    sparse_keys = set()
    for key in data.files:
        if key.startswith('info_') and key.endswith('_shape'):
            base_key = key[5:-6]  # Remove 'info_' prefix and '_shape' suffix
            sparse_keys.add(base_key)

    # Reconstruct sparse arrays
    for base_key in sparse_keys:
        indices = data[f'info_{base_key}_indices']
        sparse_data = data[f'info_{base_key}_data']
        shape = tuple(data[f'info_{base_key}_shape'])

        # Reconstruct dense array filled with NaN
        arr = np.full(shape, np.nan, dtype=sparse_data.dtype)
        for i, (ni, fi) in enumerate(indices):
            arr[ni, fi] = sparse_data[i]
        info[base_key] = arr

    # Second pass: load regular arrays
    for key in data.files:
        if key.startswith('info_') and not key.startswith('info_stage'):
            info_key = key[5:]  # Remove 'info_' prefix
            # Skip sparse array components
            if any(info_key.startswith(sk) or info_key.endswith('_indices') or
                   info_key.endswith('_data') or info_key.endswith('_shape') for sk in sparse_keys):
                continue
            arr = data[key]
            info[info_key] = arr

    if info:
        results.update('info', info)

    # 3. Reconstruct disentanglement
    if '_disentanglement_json' in data:
        disent = json.loads(str(data['_disentanglement_json'][0]))
        for key in ['feat_feat_significance', 'disent_matrix', 'count_matrix']:
            npz_key = f'disent_{key}'
            if npz_key in data:
                disent[key] = data[npz_key]
        results.update('disentanglement', disent)

    return results
