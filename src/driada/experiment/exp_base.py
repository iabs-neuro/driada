import numpy as np
import warnings
import tqdm
import pickle

from ..information.info_base import TimeSeries
from ..information.info_base import MultiTimeSeries
from .neuron import (
    DEFAULT_MIN_BEHAVIOUR_TIME,
    DEFAULT_T_OFF,
    DEFAULT_FPS,
    MIN_CA_SHIFT,
    Neuron,
)
from ..utils.data import get_hash, populate_nested_dict
from ..information.info_base import get_1d_mi
from ..intense.intense_base import get_multicomp_correction_thr

STATS_VARS = [
    "data_hash",
    "opt_delay",
    "pre_pval",
    "pre_rval",
    "pval",
    "rval",
    "me",
    "rel_me_beh",
    "rel_me_ca",
]
SIGNIFICANCE_VARS = [
    "stage1",
    "shuffles1",
    "stage2",
    "shuffles2",
    "final_p_thr",
    "multicomp_corr",
    "pairwise_pval_thr",
]
DEFAULT_STATS = dict(zip(STATS_VARS, [None for _ in STATS_VARS]))
DEFAULT_SIGNIFICANCE = dict(zip(SIGNIFICANCE_VARS, [None for _ in SIGNIFICANCE_VARS]))


def check_dynamic_features(dynamic_features):
    """Validate that all dynamic features have the same length.

    Parameters
    ----------
    dynamic_features : dict
        Dictionary of feature_name: feature_data pairs

    Raises
    ------
    ValueError
        If features have different lengths
    TypeError
        If feature type is not supported
    """
    if not dynamic_features:
        return  # Handle empty features gracefully

    dfeat_lengths = {}
    for feat_id, current_ts in dynamic_features.items():
        # Only accept specific types - reject everything else
        if isinstance(current_ts, TimeSeries):
            len_ts = len(current_ts.data)
        elif isinstance(current_ts, MultiTimeSeries):
            # MultiTimeSeries inherits from MVData which has n_points attribute
            len_ts = current_ts.n_points
        elif isinstance(current_ts, np.ndarray):
            # Handle raw numpy arrays - last dimension is time
            if current_ts.ndim == 0:
                raise TypeError(
                    f"Feature '{feat_id}' is a scalar numpy array. "
                    f"Expected TimeSeries, MultiTimeSeries, or numpy array with at least 1 dimension."
                )
            len_ts = current_ts.shape[-1]
        else:
            # Reject all other types
            raise TypeError(
                f"Feature '{feat_id}' has unsupported type: {type(current_ts).__name__}. "
                f"Expected TimeSeries, MultiTimeSeries, or numpy array only."
            )

        dfeat_lengths[feat_id] = len_ts

    # Check all features have same length
    unique_lengths = set(dfeat_lengths.values())
    if len(unique_lengths) != 1:
        # Create informative error message
        length_info = [
            f"  {feat}: {length} timepoints" for feat, length in dfeat_lengths.items()
        ]
        raise ValueError(
            "Dynamic features have different lengths:\n" + "\n".join(length_info)
        )


class Experiment:
    """
    Class for all Ca2+ experiment types

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(
        self,
        signature,
        calcium,
        spikes,
        exp_identificators,
        static_features,
        dynamic_features,
        **kwargs,
    ):

        fit_individual_t_off = kwargs.get("fit_individual_t_off", False)
        reconstruct_spikes = kwargs.get("reconstruct_spikes", "wavelet")
        bad_frames_mask = kwargs.get("bad_frames_mask", None)
        spike_kwargs = kwargs.get("spike_kwargs", None)
        self.verbose = kwargs.get("verbose", True)

        check_dynamic_features(dynamic_features)
        self.exp_identificators = exp_identificators
        self.signature = signature

        for idx in exp_identificators:
            setattr(self, idx, exp_identificators[idx])

        if calcium is None:
            raise ValueError(
                "Calcium data is required. Please provide a numpy array with shape (n_neurons, n_timepoints)."
            )

        if reconstruct_spikes is None:
            if spikes is None:
                warnings.warn(
                    "No spike data provided, spikes reconstruction from Ca2+ data disabled"
                )
        else:
            if spikes is not None:
                warnings.warn(
                    f"Spike data will be overridden by reconstructed spikes from Ca2+ data with method={reconstruct_spikes}"
                )

            # Store the reconstruction method for potential future use
            self.spike_reconstruction_method = reconstruct_spikes

            # Reconstruct spikes
            spikes = self._reconstruct_spikes(
                calcium, reconstruct_spikes, static_features.get("fps"), spike_kwargs
            )

        self.filtered_flag = False
        if bad_frames_mask is not None:
            calcium, spikes, dynamic_features = self._trim_data(
                calcium, spikes, dynamic_features, bad_frames_mask
            )
        else:
            for feat_id in dynamic_features.copy():
                feat_data = dynamic_features[feat_id]
                # Skip if already a TimeSeries or MultiTimeSeries
                if not isinstance(feat_data, (TimeSeries, MultiTimeSeries)):
                    # Convert numpy arrays based on dimensionality
                    if isinstance(feat_data, np.ndarray):
                        if feat_data.ndim == 1:
                            # 1D array -> TimeSeries
                            dynamic_features[feat_id] = TimeSeries(feat_data)
                        elif feat_data.ndim == 2:
                            # 2D array -> MultiTimeSeries (each row is a component)
                            ts_list = [
                                TimeSeries(feat_data[i, :], discrete=False)
                                for i in range(feat_data.shape[0])
                            ]
                            dynamic_features[feat_id] = MultiTimeSeries(ts_list)
                        else:
                            raise ValueError(
                                f"Feature {feat_id} has unsupported dimensionality: {feat_data.ndim}D"
                            )
                    else:
                        # Assume it's 1D data if not numpy array
                        dynamic_features[feat_id] = TimeSeries(feat_data)

        self.n_cells = calcium.shape[0]
        self.n_frames = calcium.shape[1]

        self.neurons = []

        if self.verbose:
            print("Building neurons...")
        for i in tqdm.tqdm(np.arange(self.n_cells), position=0, leave=True, disable=not self.verbose):
            cell = Neuron(
                str(i),
                calcium[i, :],
                spikes[i, :] if spikes is not None else None,
                default_t_rise=static_features.get("t_rise_sec"),
                default_t_off=static_features.get("t_off_sec"),
                fps=static_features.get("fps"),
                fit_individual_t_off=fit_individual_t_off,
            )

            self.neurons.append(cell)

        # Now create MultiTimeSeries from neurons to preserve their shuffle masks
        calcium_ts_list = [neuron.ca for neuron in self.neurons]
        spikes_ts_list = [
            (
                neuron.sp
                if neuron.sp is not None
                else TimeSeries(np.zeros(self.n_frames), discrete=True)
            )
            for neuron in self.neurons
        ]

        # Create MultiTimeSeries from the TimeSeries objects in neurons
        # This preserves the individual shuffle masks created by each Neuron
        self.calcium = MultiTimeSeries(calcium_ts_list)
        # Allow zero columns for spikes since many neurons might not spike
        self.spikes = MultiTimeSeries(spikes_ts_list, allow_zero_columns=True)

        self.dynamic_features = dynamic_features

        # Protected attributes that should not be overwritten
        protected_attrs = {
            "neurons",
            "calcium",
            "spikes",
            "n_cells",
            "n_frames",
            "stats_tables",
            "significance_tables",
            "embeddings",
            "_rdm_cache",
            "_data_hashes",
            "signature",
            "exp_identificators",
        }

        # Check for protected attributes in dynamic features and remove them
        conflicting_features = []
        for feat_id in dynamic_features:
            if isinstance(feat_id, str) and feat_id in protected_attrs:
                conflicting_features.append(feat_id)

        if conflicting_features:
            raise ValueError(
                f"Dynamic feature names conflict with protected attributes: {conflicting_features}. "
                f"Protected attributes are: {sorted(protected_attrs)}"
            )

        # Set dynamic features as attributes
        for feat_id in dynamic_features:
            if isinstance(feat_id, str):
                if hasattr(self, feat_id):
                    warnings.warn(
                        f"Feature name '{feat_id}' overwrites existing attribute."
                    )
                setattr(self, feat_id, dynamic_features[feat_id])
            # Skip tuples (multifeatures) as they can't be attribute names

        for sfeat_name in static_features:
            if sfeat_name in protected_attrs:
                warnings.warn(
                    f"Static feature name '{sfeat_name}' conflicts with protected attribute. "
                    f"Access via attribute name with underscore: _{sfeat_name}"
                )
                setattr(self, f"_{sfeat_name}", static_features[sfeat_name])
            else:
                setattr(self, sfeat_name, static_features[sfeat_name])

        # for selectivity data from INTENSE
        self.stats_tables = {}
        self.significance_tables = {}
        self.selectivity_tables_initialized = False

        # for dimensionality reduction embeddings
        self.embeddings = {"calcium": {}, "spikes": {}}

        # Cache for RDM computations
        self._rdm_cache = {}

        if self.verbose:
            print("Building data hashes...")
        self._build_data_hashes(mode="calcium")
        if reconstruct_spikes is not None or spikes is not None:
            self._build_data_hashes(mode="spikes")

        if self.verbose:
            print("Final checkpoint...")
        self._checkpoint()
        # self._load_precomputed_data(**kwargs)

        if self.verbose:
            print(
                f'Experiment "{self.signature}" constructed successfully with {self.n_cells} neurons and {len(self.dynamic_features)} features'
            )

    def check_ds(self, ds):
        if not hasattr(self, "fps"):
            raise ValueError(f"fps not set for {self.signature}")

        time_step = 1.0 / self.fps
        if time_step * ds > DEFAULT_MIN_BEHAVIOUR_TIME:
            if self.verbose:
                print(
                    "Downsampling constant is too high: some behaviour acts may be skipped. "
                    f"Current minimal behaviour time interval is set to {DEFAULT_MIN_BEHAVIOUR_TIME} sec, "
                    f"downsampling {ds} will create time gaps of {time_step*ds} sec"
                )

    def _set_selectivity_tables(self, mode, fbunch=None, cbunch=None):
        # neuron-feature pair statistics
        stats_table = self._populate_cell_feat_dict(
            DEFAULT_STATS, fbunch=fbunch, cbunch=cbunch
        )

        # neuron-feature pair significance-related data
        significance_table = self._populate_cell_feat_dict(
            DEFAULT_SIGNIFICANCE, fbunch=fbunch, cbunch=cbunch
        )
        self.stats_tables[mode] = stats_table
        self.significance_tables[mode] = significance_table
        self.selectivity_tables_initialized = True

    def _build_pair_hash(self, cell_id, feat_id, mode="calcium"):
        """
        Builds a unique hash-based representation of activity-feature pair data.
        feat_id should be a string or an iterable of strings (in case of joint MI calculation).
        """
        if mode == "calcium":
            act = self.neurons[cell_id].ca.data
        elif mode == "spikes":
            act = self.neurons[cell_id].sp.data
        else:
            raise ValueError('"mode" can be either "calcium" or "spikes"')

        act_hash = get_hash(act)

        if (not isinstance(feat_id, str)) and len(feat_id) == 1:
            feat_id = feat_id[0]

        if isinstance(feat_id, str):
            dyn_data = self.dynamic_features[feat_id].data
            dyn_data_hash = get_hash(dyn_data)
            pair_hash = (act_hash, dyn_data_hash)

        else:
            ordered_fnames = tuple(sorted(list(feat_id)))
            list_of_hashes = [act_hash]
            for fname in ordered_fnames:
                dyn_data = self.dynamic_features[fname].data
                dyn_data_hash = get_hash(dyn_data)
                list_of_hashes.append(dyn_data_hash)

            pair_hash = tuple(list_of_hashes)

        return pair_hash

    def _build_data_hashes(self, mode="calcium"):
        """
        Builds a unique hash-based representation of calcium-feature pair data for all cell-feature pairs..
        """
        # Create default hashes structure for this mode only
        if not hasattr(self, "_data_hashes"):
            self._data_hashes = {}

        # Create independent dictionary for each mode to avoid aliasing
        mode_hashes = {
            dfeat: dict(zip(range(self.n_cells), [None for _ in range(self.n_cells)]))
            for dfeat in self.dynamic_features.keys()
        }
        self._data_hashes[mode] = mode_hashes

        # Populate hashes for this mode
        for feat_id in self.dynamic_features:
            for cell_id in range(self.n_cells):
                self._data_hashes[mode][feat_id][cell_id] = self._build_pair_hash(
                    cell_id, feat_id, mode=mode
                )

    def _trim_data(
        self, calcium, spikes, dynamic_features, bad_frames_mask, force_filter=False
    ):

        if not force_filter and self.filtered_flag:
            raise AttributeError(
                'Data is already filtered, if you want to force filtering it again, set "force_filter = True"'
            )

        f_calcium = calcium[:, ~bad_frames_mask]
        if spikes is not None:
            f_spikes = spikes[:, ~bad_frames_mask]
        else:
            f_spikes = None

        f_dynamic_features = {}
        for feat_id in dynamic_features:
            current_ts = dynamic_features[feat_id]
            if isinstance(current_ts, TimeSeries):
                f_ts = TimeSeries(
                    current_ts.data[~bad_frames_mask], discrete=current_ts.discrete
                )
            elif isinstance(current_ts, MultiTimeSeries):
                # Handle MultiTimeSeries by trimming each component
                filtered_components = []
                for i in range(current_ts.n_dim):
                    component_data = current_ts.data[i, ~bad_frames_mask]
                    filtered_components.append(
                        TimeSeries(component_data, discrete=current_ts.discrete)
                    )
                f_ts = MultiTimeSeries(filtered_components)
            elif isinstance(current_ts, np.ndarray):
                # Handle raw arrays
                if current_ts.ndim == 1:
                    f_ts = TimeSeries(current_ts[~bad_frames_mask])
                else:
                    # Multi-dimensional array
                    f_ts = current_ts[:, ~bad_frames_mask]
            else:
                # Fallback for other types
                f_ts = TimeSeries(current_ts[~bad_frames_mask])

            f_dynamic_features[feat_id] = f_ts

        self.filtered_flag = True
        self.bad_frames_mask = bad_frames_mask

        return f_calcium, f_spikes, f_dynamic_features

    def _checkpoint(self):
        """
        Check build for common errors
        """
        # Check minimal length for proper shuffle mask creation
        t_off_sec = getattr(self, "t_off_sec", DEFAULT_T_OFF)
        fps = getattr(self, "fps", DEFAULT_FPS)
        t_off_frames = t_off_sec * fps
        min_required_frames = (
            int(t_off_frames * MIN_CA_SHIFT * 2) + 10
        )  # Need space for both ends + some valid positions

        if self.n_frames < min_required_frames:
            raise ValueError(
                f"Signal too short: {self.n_frames} frames. "
                f"Minimum {min_required_frames} frames required for proper shuffle mask creation "
                f"(based on t_off={t_off_sec}s, fps={fps}Hz)."
            )

        if self.n_cells > self.n_frames:
            raise ValueError(
                f"Number of cells ({self.n_cells}) > number of time frames ({self.n_frames}). "
                f"Data appears to be transposed. Expected shape: (n_neurons, n_timepoints)."
            )

        for dfeat in ["calcium", "spikes"]:
            if self.n_frames not in getattr(self, dfeat).shape:
                raise ValueError(
                    f'"{dfeat}" feature has inappropriate shape: {getattr(self, dfeat).data.shape}'
                    f"inconsistent with data length {self.n_frames}"
                )

        for dfeat in self.dynamic_features.keys():
            if isinstance(dfeat, str):
                if self.n_frames not in getattr(self, dfeat).data.shape:
                    raise ValueError(
                        f'"{dfeat}" feature has inappropriate shape: {getattr(self, dfeat).data.shape}'
                        f"inconsistent with data length {self.n_frames}"
                    )
            else:
                # For tuple features (multifeatures), check the underlying data
                feat_data = self.dynamic_features[dfeat]
                if (
                    hasattr(feat_data, "data")
                    and self.n_frames not in feat_data.data.shape
                ):
                    raise ValueError(
                        f'"{dfeat}" feature has inappropriate shape: {feat_data.data.shape}'
                        f"inconsistent with data length {self.n_frames}"
                    )

    def _populate_cell_feat_dict(self, content, fbunch=None, cbunch=None):
        """
        Helper function. Creates a nested dictionary of feature-cell pairs and populates every cell with 'content' variable.
        Outer dict: dynamic features, inner dict: cells
        """
        cell_ids = self._process_cbunch(cbunch)
        feat_ids = self._process_fbunch(fbunch, allow_multifeatures=True)
        nested_dict = populate_nested_dict(content, feat_ids, cell_ids)
        return nested_dict

    def _process_cbunch(self, cbunch):
        """
        Helper function. Turns cell indices (int, iterable or None) into a list of cell numbers.
        """
        if isinstance(cbunch, int):
            cell_ids = [cbunch]
        elif cbunch is None:
            cell_ids = list(np.arange(self.n_cells))
        else:
            cell_ids = list(cbunch)

        return cell_ids

    def _process_fbunch(self, fbunch, allow_multifeatures=False, mode="calcium"):
        """
        Helper function. Turns feature names (str, iterable or None) into a list of feature names
        """
        if isinstance(fbunch, str):
            feat_ids = [fbunch]

        elif fbunch is None:  # default set of features
            if allow_multifeatures:
                try:
                    # stats table contains up-to-date set of features, including multifeatures
                    feat_ids = list(self.stats_tables[mode].keys())
                except KeyError:
                    # if stats is not available, take pre-defined full set of features
                    feat_ids = list(self.dynamic_features.keys())
            else:
                feat_ids = list(self.dynamic_features.keys())

        else:
            feat_ids = []

            # check for multifeatures
            for fname in fbunch:
                if isinstance(fname, str):
                    if fname in self.dynamic_features:
                        feat_ids.append(fname)
                else:
                    if allow_multifeatures:
                        feat_ids.append(tuple(sorted(list(fname))))
                    else:
                        raise ValueError(
                            'Multifeature detected in "allow_multifeatures=False" mode'
                        )

        return feat_ids

    def _process_sbunch(self, sbunch, significance_mode=False):
        """
        Helper function. Turns stats type names (str, iterable or None) into a list of stats types
        """
        if significance_mode:
            default_list = SIGNIFICANCE_VARS
        else:
            default_list = STATS_VARS

        if isinstance(sbunch, str):
            return [sbunch]

        elif sbunch is None:
            return default_list

        else:
            return [st for st in sbunch if st in default_list]

    def _add_multifeature_to_data_hashes(self, feat_id, mode="calcium"):
        """
        Add previously unseen multifeature (e.g. ['x','y']) to table with data hashes.
        This function ignores multifeatures that already exist in the table.
        """
        if (not isinstance(feat_id, str)) and len(feat_id) == 1:
            feat_id = feat_id[0]

        if not isinstance(feat_id, str):
            ordered_fnames = tuple(sorted(list(feat_id)))
            if ordered_fnames not in self._data_hashes[mode]:
                all_hashes = [
                    self._build_pair_hash(cell_id, ordered_fnames)
                    for cell_id in range(self.n_cells)
                ]
                new_dict = {ordered_fnames: dict(zip(range(self.n_cells), all_hashes))}
                self._data_hashes[mode].update(new_dict)

        else:
            raise ValueError("This method is for multifeature update only")

    def _add_multifeature_to_stats(self, feat_id, mode="calcium"):
        """
        Add previously unseen multifeature (e.g. ['x','y']) to statistics and significance tables.
        This function ignores multifeatures that already exist in the table.
        """
        if (not isinstance(feat_id, str)) and len(feat_id) == 1:
            feat_id = feat_id[0]

        if not isinstance(feat_id, str):
            ordered_fnames = tuple(sorted(list(feat_id)))
            if ordered_fnames not in self.stats_tables[mode]:
                if self.verbose:
                    print(f"Multifeature {feat_id} is new, it will be added to stats table")
                self.stats_tables[mode][ordered_fnames] = {
                    cell_id: DEFAULT_STATS.copy() for cell_id in range(self.n_cells)
                }

                self.significance_tables[mode][ordered_fnames] = {
                    cell_id: DEFAULT_SIGNIFICANCE.copy()
                    for cell_id in range(self.n_cells)
                }

        else:
            raise ValueError("This method is for multifeature update only")

    def _check_stats_relevance(self, cell_id, feat_id, mode="calcium"):
        """
        A guardian function that prevents access to non-existing and irrelevant data.

        This function checks whether the calcium-feature pair statistics has already been calculated.
        It ensures the data (both calcium and dynamic feature) has not changed since the last
        calculation by checking hash values of both data arrays.

        This function always refers to stats table but works equally well with significance table
        since they are always updated simultaneously
        """

        if not isinstance(feat_id, str):
            feat_id = tuple(sorted(list(feat_id)))

        if feat_id not in self.stats_tables[mode]:
            raise ValueError(
                f"Feature {feat_id} is not present in stats. \n If this is a single feature, "
                "check the input data, since all single features are processed automatically."
                'If this is a multifeature (e.g. ["x", "y"]), compute MI significance to create stats'
            )

        pair_hash = self._data_hashes[mode][feat_id][cell_id]
        existing_hash = self.stats_tables[mode][feat_id][cell_id]["data_hash"]

        # if (stats does not exist yet) or (stats exists and data is the same):
        if (existing_hash is None) or (pair_hash == existing_hash):
            return True

        else:
            if self.verbose:
                print(
                    f"Looks like the data for the pair (cell {cell_id}, feature {feat_id}) "
                    "has been changed since the last calculation)"
                )

            return False

    def _update_stats_and_significance(
        self, stats, mode, cell_id, feat_id, stage2_only
    ):
        """
        Updates stats table and linked significance table to erase irrelevant data properly
        """
        # update statistics
        self.stats_tables[mode][feat_id][cell_id].update(stats)
        if not stage2_only:
            # erase significance data completely since stats for stage 1 has been modified
            self.significance_tables[mode][feat_id][cell_id].update(
                DEFAULT_SIGNIFICANCE.copy()
            )
        else:
            # erase significance data for stage 2 since stats for stage 2 has been modified
            self.significance_tables[mode][feat_id][cell_id].update(
                {"stage2": None, "shuffles2": None}
            )

    def update_neuron_feature_pair_stats(
        self,
        stats,
        cell_id,
        feat_id,
        mode="calcium",
        force_update=False,
        stage2_only=False,
    ):
        """
        Updates calcium-feature pair statistics.
        feat_id should be a string or an iterable of strings (in case of joint MI calculation).
        This function allows multifeatures.
        """

        if not isinstance(feat_id, str):
            self._add_multifeature_to_data_hashes(feat_id, mode=mode)
            self._add_multifeature_to_stats(feat_id, mode=mode)

        if self._check_stats_relevance(cell_id, feat_id, mode=mode):
            self._update_stats_and_significance(
                stats, mode, cell_id, feat_id, stage2_only=stage2_only
            )

        else:
            if not force_update:
                if self.verbose:
                    print('To forcefully update the stats, set "force_update=True"')
            else:
                self._update_stats_and_significance(
                    stats, mode, cell_id, feat_id, stage2_only=stage2_only
                )

    def update_neuron_feature_pair_significance(
        self, sig, cell_id, feat_id, mode="calcium"
    ):
        """
        Updates calcium-feature pair significance data.
        feat_id should be a string or an iterable of strings (in case of joint MI calculation).
        This function allows multifeatures.
        """
        if not isinstance(feat_id, str):
            self._add_multifeature_to_data_hashes(feat_id, mode=mode)
            self._add_multifeature_to_stats(feat_id, mode=mode)

        if self._check_stats_relevance(cell_id, feat_id, mode=mode):
            self.significance_tables[mode][feat_id][cell_id].update(sig)

        else:
            raise ValueError(
                "Can not update significance table until the collision between actual data hashes and "
                "saved stats data hashes is resolved. Use update_neuron_feature_pair_stats"
                'with "force_update=True" to forcefully rewrite statistics'
            )

    def get_neuron_feature_pair_stats(self, cell_id, feat_id, mode="calcium"):
        """
        Returns calcium-feature pair statistics.
        This function allows multifeatures.
        """
        stats = None
        if self._check_stats_relevance(cell_id, feat_id):
            stats = self.stats_tables[mode][feat_id][cell_id]
        else:
            if self.verbose:
                print("Consider recalculating stats")

        return stats

    def get_neuron_feature_pair_significance(self, cell_id, feat_id, mode="calcium"):
        """
        Returns calcium-feature pair significance data.
        This function allows multifeatures.
        """
        sig = None
        if self._check_stats_relevance(cell_id, feat_id):
            sig = self.significance_tables[mode][feat_id][cell_id]
        else:
            if self.verbose:
                print("Consider recalculating stats")

        return sig

    def get_multicell_shuffled_calcium(
        self, cbunch=None, method="roll_based", return_array=True, **kwargs
    ):
        """
        Get shuffled calcium data for multiple cells.

        Parameters
        ----------
        cbunch : int, list, or None
            Cell indices. If None, all cells are used.
        method : {'roll_based', 'waveform_based', 'chunks_based'}, default='roll_based'
            Shuffling method to use
        return_array : bool, default=True
            If True, return numpy array. If False, return MultiTimeSeries object.
        **kwargs
            Additional parameters passed to the shuffling method

        Returns
        -------
        np.ndarray or MultiTimeSeries
            If return_array=True: Shuffled calcium data with shape (n_cells, n_frames)
            If return_array=False: MultiTimeSeries object containing shuffled data

        """
        # Validate method
        valid_methods = ["roll_based", "waveform_based", "chunks_based"]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid shuffling method '{method}'. Must be one of: {valid_methods}"
            )

        cell_list = self._process_cbunch(cbunch)

        # Validate cell indices
        if any(idx < 0 or idx >= self.n_cells for idx in cell_list):
            raise ValueError(
                f"Invalid cell indices. Must be between 0 and {self.n_cells-1}"
            )

        if return_array:
            agg_sh_data = np.zeros((len(cell_list), self.n_frames))
            for i, cell_idx in enumerate(cell_list):
                cell = self.neurons[cell_idx]
                sh_data = cell.get_shuffled_calcium(method=method, return_array=True, **kwargs)
                agg_sh_data[i, :] = sh_data
            return agg_sh_data
        else:
            # Return MultiTimeSeries object
            ts_list = []
            for cell_idx in cell_list:
                cell = self.neurons[cell_idx]
                sh_ts = cell.get_shuffled_calcium(method=method, return_array=False, **kwargs)
                ts_list.append(sh_ts)
            
            # Create MultiTimeSeries from list of TimeSeries
            from ..information.info_base import MultiTimeSeries
            return MultiTimeSeries(ts_list)

    def get_multicell_shuffled_spikes(
        self, cbunch=None, method="isi_based", return_array=True, **kwargs
    ):
        """
        Get shuffled spike data for multiple cells.

        Parameters
        ----------
        cbunch : int, list, or None
            Cell indices. If None, all cells are used.
        method : {'isi_based'}, default='isi_based'
            Shuffling method. Currently only 'isi_based' is supported for spikes.
        return_array : bool, default=True
            If True, return numpy array. If False, return MultiTimeSeries object.
        **kwargs
            Additional parameters passed to the shuffling method

        Returns
        -------
        np.ndarray or MultiTimeSeries
            If return_array=True: Shuffled spike data with shape (n_cells, n_frames)
            If return_array=False: MultiTimeSeries object containing shuffled spike data

        """
        # Check if spikes data is meaningful (not all zeros)
        if not np.any(self.spikes.data):
            raise AttributeError(
                "Unable to shuffle spikes without meaningful spikes data"
            )

        # Validate method
        valid_methods = ["isi_based"]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid spike shuffling method '{method}'. Must be one of: {valid_methods}"
            )

        cell_list = self._process_cbunch(cbunch)

        # Validate cell indices
        if any(idx < 0 or idx >= self.n_cells for idx in cell_list):
            raise ValueError(
                f"Invalid cell indices. Must be between 0 and {self.n_cells-1}"
            )

        if return_array:
            agg_sh_data = np.zeros((len(cell_list), self.n_frames))
            for i, cell_idx in enumerate(cell_list):
                cell = self.neurons[cell_idx]
                sh_data = cell.get_shuffled_spikes(method=method, return_array=True, **kwargs)
                agg_sh_data[i, :] = sh_data
            return agg_sh_data
        else:
            # Return MultiTimeSeries object
            ts_list = []
            for cell_idx in cell_list:
                cell = self.neurons[cell_idx]
                sh_ts = cell.get_shuffled_spikes(method=method, return_array=False, **kwargs)
                ts_list.append(sh_ts)
            
            # Create MultiTimeSeries from list of TimeSeries
            from ..information.info_base import MultiTimeSeries
            return MultiTimeSeries(ts_list)

    def get_stats_slice(
        self,
        table_to_scan=None,
        cbunch=None,
        fbunch=None,
        sbunch=None,
        significance_mode=False,
        mode="calcium",
    ):
        """
        returns slice of accumulated statistics data (or significance data if "significance_mode=True")
        """
        cell_ids = self._process_cbunch(cbunch)
        feat_ids = self._process_fbunch(fbunch, allow_multifeatures=True, mode=mode)
        slist = self._process_sbunch(sbunch, significance_mode=significance_mode)

        if table_to_scan is None:
            if significance_mode:
                full_table = self.significance_tables[mode]
            else:
                full_table = self.stats_tables[mode]
        else:
            full_table = table_to_scan

        out_table = self._populate_cell_feat_dict(dict(), fbunch=fbunch, cbunch=cbunch)
        for feat_id in feat_ids:
            for cell_id in cell_ids:
                out_table[feat_id][cell_id] = {
                    s: full_table[feat_id][cell_id][s] for s in slist
                }

        return out_table

    def get_significance_slice(
        self, cbunch=None, fbunch=None, sbunch=None, mode="calcium"
    ):
        return self.get_stats_slice(
            cbunch=cbunch,
            fbunch=fbunch,
            sbunch=sbunch,
            significance_mode=True,
            mode=mode,
        )

    def get_feature_entropy(self, feat_id, ds=1):
        """
        Calculates entropy of a single dynamic feature or joint entropy of multiple features.

        Parameters
        ----------
        feat_id : str or tuple
            - str: Name of a single feature (TimeSeries or MultiTimeSeries)
            - tuple: Names of exactly 2 features for joint entropy calculation
        ds : int
            Downsampling factor

        Returns
        -------
        float
            Entropy value in bits (or nats for continuous variables)

        Notes
        -----
        - Single features use their native get_entropy() method
        - Tuples calculate joint entropy for exactly 2 variables
        - Joint entropy of 3+ variables is not supported (use MultiTimeSeries instead)
        - Continuous variables may return negative entropy values
        """
        if isinstance(feat_id, str):
            # Single feature - use its get_entropy method
            fts = self.dynamic_features[feat_id]

            # Check for continuous components and warn
            if isinstance(fts, TimeSeries) and not fts.discrete:
                warnings.warn(
                    f"Feature '{feat_id}' is continuous. Entropy may be negative. "
                    "Consider using mutual information or other measures for continuous variables."
                )
            elif isinstance(fts, MultiTimeSeries):
                # MultiTimeSeries with continuous components
                if not fts.discrete:
                    warnings.warn(
                        f"Feature '{feat_id}' contains continuous components. "
                        "Differential entropy may be negative and depends on the scale/units."
                    )

            return fts.get_entropy(ds=ds)

        elif isinstance(feat_id, (tuple, list)):
            # Joint entropy of multiple features
            if len(feat_id) != 2:
                raise ValueError(
                    f"Joint entropy is only supported for exactly 2 variables, got {len(feat_id)}. "
                    f"For {len(feat_id)} variables, create a MultiTimeSeries instead."
                )

            # Get the two features
            feat1_name, feat2_name = feat_id
            feat1 = self.dynamic_features[feat1_name]
            feat2 = self.dynamic_features[feat2_name]

            # Check for continuous components
            has_continuous = False
            if isinstance(feat1, TimeSeries) and not feat1.discrete:
                has_continuous = True
            elif isinstance(feat1, MultiTimeSeries) and not feat1.discrete:
                has_continuous = True
            if isinstance(feat2, TimeSeries) and not feat2.discrete:
                has_continuous = True
            elif isinstance(feat2, MultiTimeSeries) and not feat2.discrete:
                has_continuous = True

            if has_continuous:
                warnings.warn(
                    "One or both features contain continuous components. "
                    "Joint differential entropy may be negative and is scale-dependent."
                )

            # For joint entropy of 2 variables, use H(X,Y) = H(X) + H(Y) - MI(X,Y)
            h_x = feat1.get_entropy(ds=ds)
            h_y = feat2.get_entropy(ds=ds)
            mi_xy = get_1d_mi(feat1, feat2, ds=ds)

            return h_x + h_y - mi_xy

        else:
            raise TypeError(
                f"feat_id must be str or tuple of 2 feature names, got {type(feat_id)}"
            )

    def _reconstruct_spikes(self, calcium, method, fps, spike_kwargs=None):
        """
        Reconstruct spikes from calcium signals using specified method.

        Parameters
        ----------
        calcium : np.ndarray
            Calcium traces, shape (n_neurons, n_timepoints)
        method : str or callable
            Reconstruction method: 'wavelet' or a callable function
        fps : float
            Sampling rate in frames per second
        spike_kwargs : dict, optional
            Method-specific parameters

        Returns
        -------
        spikes : np.ndarray
            Reconstructed spike trains
        """
        from .spike_reconstruction import reconstruct_spikes

        # Convert calcium to MultiTimeSeries if needed
        if isinstance(calcium, np.ndarray):
            # Create temporary MultiTimeSeries from numpy array
            from ..information.info_base import TimeSeries, MultiTimeSeries

            # Calcium data is always continuous, so explicitly set discrete=False
            ts_list = [
                TimeSeries(calcium[i, :], discrete=False)
                for i in range(calcium.shape[0])
            ]
            calcium_mts = MultiTimeSeries(ts_list)
        else:
            calcium_mts = calcium

        # Call the unified reconstruction function
        spikes_mts, metadata = reconstruct_spikes(
            calcium_mts, method=method, fps=fps, params=spike_kwargs
        )

        # Store metadata
        self._reconstruction_metadata = metadata

        # Return numpy array for backward compatibility
        return spikes_mts.data

    def get_significant_neurons(
        self, min_nspec=1, cbunch=None, fbunch=None, mode="calcium",
        override_intense_significance=False, pval_thr=0.05, 
        multicomp_correction=None, significance_update=False
    ):
        """
        Returns a dict with neuron ids as keys and their significantly correlated features as values
        Only neurons with "min_nspec" or more significantly correlated features will be returned

        Parameters
        ----------
        min_nspec : int
            Minimum number of significantly correlated features required
        cbunch : int, list or None
            Cell indices to analyze. By default (None), all neurons will be checked
        fbunch : str, list or None
            Feature names to check. By default (None), all features will be checked
        mode : str
            Data type: 'calcium' or 'spikes'
        override_intense_significance : bool, optional
            If True, recompute significance using pval_thr and multicomp_correction
            instead of using pre-computed INTENSE significance. Default is False.
        pval_thr : float, optional
            P-value threshold for significance testing. Default is 0.05.
            Only used if override_intense_significance=True.
        multicomp_correction : str or None, optional
            Multiple comparison correction method. Default is None (no correction).
            Options: None, 'bonferroni', 'holm', 'fdr_bh'
            Only used if override_intense_significance=True.
        significance_update : bool, optional
            If True, update the significance tables with new thresholds.
            If False (default), only use new thresholds for current query.
            Only used if override_intense_significance=True.

        Returns
        -------
        dict
            Dictionary with neuron IDs as keys and lists of significant features as values
        """
        cell_ids = self._process_cbunch(cbunch)
        feat_ids = self._process_fbunch(fbunch, allow_multifeatures=True, mode=mode)

        # Check relevance only for requested cells and features
        relevance = [
            self._check_stats_relevance(cell_id, feat_id, mode=mode)
            for cell_id in cell_ids
            for feat_id in feat_ids
        ]
        if not np.all(np.array(relevance)):
            raise ValueError("Stats relevance error")

        cell_feat_dict = {cell_id: [] for cell_id in cell_ids}
        
        if override_intense_significance:
            # Collect all p-values for multiple comparison correction
            all_pvals = []
            cell_feat_pvals = {}
            
            for cell_id in cell_ids:
                cell_feat_pvals[cell_id] = {}
                for feat_id in feat_ids:
                    pval = self.stats_tables[mode][feat_id][cell_id].get("pval", 1.0)
                    cell_feat_pvals[cell_id][feat_id] = pval
                    all_pvals.append(pval)
            
            # Calculate corrected threshold
            if multicomp_correction is None:
                corrected_threshold = pval_thr
            elif multicomp_correction == "bonferroni":
                corrected_threshold = get_multicomp_correction_thr(
                    pval_thr, mode="bonferroni", nhyp=len(all_pvals)
                )
            elif multicomp_correction in ["holm", "fdr_bh"]:
                corrected_threshold = get_multicomp_correction_thr(
                    pval_thr, mode=multicomp_correction, all_pvals=all_pvals
                )
            else:
                raise ValueError(
                    f"Unknown multicomp_correction method: {multicomp_correction}. "
                    "Options: None, 'bonferroni', 'holm', 'fdr_bh'"
                )
            
            # Determine significance based on new threshold
            for cell_id in cell_ids:
                for feat_id in feat_ids:
                    pval = cell_feat_pvals[cell_id][feat_id]
                    
                    # Check if significant according to new threshold
                    is_significant = pval < corrected_threshold
                    
                    if is_significant:
                        cell_feat_dict[cell_id].append(feat_id)
                    
                    # Update significance tables if requested
                    if significance_update:
                        self.significance_tables[mode][feat_id][cell_id]["stage2"] = is_significant
                        self.significance_tables[mode][feat_id][cell_id]["pval_thr"] = pval_thr
                        self.significance_tables[mode][feat_id][cell_id]["multicomp_correction"] = multicomp_correction
                        self.significance_tables[mode][feat_id][cell_id]["corrected_pval_thr"] = corrected_threshold
        else:
            # Use pre-computed INTENSE significance
            for cell_id in cell_ids:
                for feat_id in feat_ids:
                    if self.significance_tables[mode][feat_id][cell_id]["stage2"]:
                        cell_feat_dict[cell_id].append(feat_id)

        # filter out cells without enough specializations
        final_cell_feat_dict = {
            cell_id: cell_feat_dict[cell_id]
            for cell_id in cell_ids
            if len(cell_feat_dict[cell_id]) >= min_nspec
        }

        return final_cell_feat_dict

    # ===================================================================================
    # not active

    def save_mi_significance_to_file(self, fname):
        with open(fname, "wb") as f:
            pickle.dump(self.mi_significance_table, f)

    def clear_cells_mi_significance_data(self, cbunch, path_to_save=None):
        for cell_id in cbunch:
            for feat in self.dynamic_features:
                self.mi_significance_table[feat][cell_id] = {}

        if path_to_save is not None:
            self.save_mi_significance_to_file(path_to_save)

    def clear_features_mi_significance_data(self, feat_list, save_to_file=False):
        pass

    def clear_cell_feat_mi_significance_data(self, cell, feat, save_to_file=False):
        pass

    def _load_precomputed_data(self, **kwargs):
        if "mi_significance" in kwargs:
            self.mi_significance_table = {
                **self.mi_significance_table,
                **kwargs["mi_significance"],
            }

    def store_embedding(
        self, embedding, method_name, data_type="calcium", metadata=None
    ):
        """
        Store dimensionality reduction embedding in the experiment.

        Parameters
        ----------
        embedding : np.ndarray
            The embedding array, shape (n_timepoints, n_components)
        method_name : str
            Name of the DR method (e.g., 'pca', 'umap', 'isomap')
        data_type : str
            Type of data used ('calcium' or 'spikes')
        metadata : dict, optional
            Additional metadata about the embedding (e.g., parameters, quality metrics)
        """
        if data_type not in ["calcium", "spikes"]:
            raise ValueError("data_type must be 'calcium' or 'spikes'")

        # Check if embedding matches expected timepoints (accounting for downsampling)
        ds = metadata.get("ds", 1) if metadata else 1
        expected_frames = self.n_frames // ds
        if embedding.shape[0] != expected_frames:
            raise ValueError(
                f"Embedding timepoints ({embedding.shape[0]}) must match expected frames "
                f"({expected_frames} = {self.n_frames} / ds={ds})"
            )

        self.embeddings[data_type][method_name] = {
            "data": embedding,
            "metadata": metadata or {},
            "timestamp": np.datetime64("now"),
            "shape": embedding.shape,
        }

    def get_embedding(self, method_name, data_type="calcium"):
        """
        Retrieve stored embedding.

        Parameters
        ----------
        method_name : str
            Name of the DR method
        data_type : str
            Type of data used ('calcium' or 'spikes')

        Returns
        -------
        dict
            Dictionary containing 'data' and 'metadata'
        """
        if data_type not in ["calcium", "spikes"]:
            raise ValueError("data_type must be 'calcium' or 'spikes'")

        if method_name not in self.embeddings[data_type]:
            raise KeyError(
                f"No embedding found for method '{method_name}' with data_type '{data_type}'"
            )

        return self.embeddings[data_type][method_name]

    def compute_rdm(
        self,
        items,
        data_type="calcium",
        metric="correlation",
        average_method="mean",
        use_cache=True,
    ):
        """
        Compute RDM with caching support.

        Parameters
        ----------
        items : str
            Name of dynamic feature to use as condition labels
        data_type : str, default 'calcium'
            Type of data to use ('calcium' or 'spikes')
        metric : str, default 'correlation'
            Distance metric for RDM computation
        average_method : str, default 'mean'
            How to average within conditions ('mean' or 'median')
        use_cache : bool, default True
            Whether to use cached results

        Returns
        -------
        rdm : np.ndarray
            Representational dissimilarity matrix
        labels : np.ndarray
            The unique labels/conditions
        """
        # Generate cache key
        cache_key = (items, data_type, metric, average_method)

        # Check cache
        if use_cache and cache_key in self._rdm_cache:
            return self._rdm_cache[cache_key]

        # Import here to avoid circular dependency
        from ..rsa.integration import compute_experiment_rdm

        # Compute RDM
        result = compute_experiment_rdm(
            self,
            items,
            data_type=data_type,
            metric=metric,
            average_method=average_method,
        )

        # Cache result
        if use_cache:
            self._rdm_cache[cache_key] = result

        return result

    def clear_rdm_cache(self):
        """Clear the RDM cache."""
        self._rdm_cache = {}
