import numpy as np
import warnings
import tqdm
from itertools import combinations
import pickle

from ..signals.sig_base import TimeSeries
from .neuron import DEFAULT_MIN_BEHAVIOUR_TIME, Neuron
from .wavelet_event_detection import WVT_EVENT_DETECTION_PARAMS, extract_wvt_events, events_to_ts_array
from ..utils.data import get_hash, populate_nested_dict
from ..information.info_base import get_1d_mi

STATS_VARS = ['data_hash', 'pre_pval', 'pre_rval', 'pval', 'rval', 'mi', 'rel_mi_beh', 'rel_mi_ca']
SIGNIFICANCE_VARS = ['stage1', 'shuffles1', 'stage2', 'shuffles2', 'final_p_thr', 'multicomp_corr', 'pairwise_pval_thr']
DEFAULT_STATS = dict(zip(STATS_VARS, [None for _ in STATS_VARS]))
DEFAULT_SIGNIFICANCE = dict(zip(SIGNIFICANCE_VARS, [None for _ in SIGNIFICANCE_VARS]))


def check_dynamic_features(dynamic_features):
    dfeat_lengths = {}
    for feat_id in dynamic_features:
        current_ts = dynamic_features[feat_id]
        if isinstance(current_ts, TimeSeries):
            len_ts = len(current_ts.data)
        else:
            len_ts = len(current_ts)

        dfeat_lengths[feat_id] = len_ts

    if len(set(dfeat_lengths.values())) != 1:
        print(dfeat_lengths)
        raise ValueError('Dynamic features have different lengths!')


class Experiment():
    '''
    Class for all Ca2+ experiment types

    Attributes
    ----------

    Methods
    -------

    '''

    def __init__(self, signature, calcium, spikes,
                 exp_identificators, static_features, dynamic_features,
                 reconstruct_spikes='wavelet', spike_kwargs=None,
                 bad_frames_mask=None, **kwargs):

        check_dynamic_features(dynamic_features)
        self.exp_identificators = exp_identificators
        self.signature = signature

        for idx in exp_identificators:
            setattr(self, idx, exp_identificators[idx])

        if calcium is None:
            raise AttributeError('No calcium data provided')

        if reconstruct_spikes is None:
            if spikes is None:
                warnings.warn('No spike data provided, spikes reconstruction from Ca2+ data disabled')
        else:
            if spikes is not None:
                warnings.warn(f'Spike data will be overridden by reconstructed spikes from Ca2+ data with method={reconstruct_spikes}')

            if reconstruct_spikes == 'wavelet':
                print('Reconstructing events with wavelet method...')
                wvt_kwargs = WVT_EVENT_DETECTION_PARAMS.copy()
                wvt_kwargs['fps'] = static_features.get('fps')
                if spike_kwargs is not None:
                    for k, v in spike_kwargs.items():
                        wvt_kwargs[k] = v
                st_ev_inds, end_ev_inds, all_ridges = extract_wvt_events(calcium, wvt_kwargs)
                spikes = events_to_ts_array(calcium.shape[1], st_ev_inds, end_ev_inds)

            else:
                raise ValueError(f'"{reconstruct_spikes} method for event reconstruction is not available"')

        self.filtered_flag = False
        if bad_frames_mask is not None:
            calcium, spikes, dynamic_features = self._trim_data(calcium,
                                                                spikes,
                                                                dynamic_features,
                                                                bad_frames_mask)
        else:
            for feat_id in dynamic_features.copy():
                if not isinstance(dynamic_features[feat_id], TimeSeries):
                    dynamic_features[feat_id] = TimeSeries(dynamic_features[feat_id])

        self.n_cells = calcium.shape[0]
        self.n_frames = calcium.shape[1]

        self.neurons = []
        self.calcium = np.zeros(calcium.shape)
        if spikes is None:
            self.spikes = np.zeros(calcium.shape)
        else:
            self.spikes = spikes

        print('Building neurons...')
        for i in tqdm.tqdm(np.arange(self.n_cells), position=0, leave=True):
            cell = Neuron(str(i),
                          calcium[i,:],
                          spikes[i,:] if reconstruct_spikes is not None else None,
                          default_t_rise=static_features.get('t_rise_sec'),
                          default_t_off=static_features.get('t_off_sec'),
                          fps=static_features.get('fps'))

                #cell.reconstruct_spikes(**kwargs)

            self.calcium[i, :] = cell.ca.data
            self.spikes[i, :] = cell.sp.data
            self.neurons.append(cell)

        self.dynamic_features = dynamic_features
        for feat_id in dynamic_features:
            setattr(self, feat_id, dynamic_features[feat_id])

        for sfeat_name in static_features:
            setattr(self, sfeat_name, static_features[sfeat_name])

        # attribute for a threshold pvalue for each pair p-value. Computed from desired FWER and
        # multiple hypothesis correction method
        self._pairwise_pval_thr = None

        #neuron-feature pair statistics
        self.stats_table = self._populate_cell_feat_dict(DEFAULT_STATS, fbunch=None, cbunch=None)

        #neuron-feature pair significance-related data
        self.significance_table = self._populate_cell_feat_dict(DEFAULT_SIGNIFICANCE, fbunch=None, cbunch=None)

        print('Building data hashes...')
        self._build_data_hashes(mode='calcium')
        if reconstruct_spikes is not None or spikes is not None:
            self._build_data_hashes(mode='spikes')

        print('Final checkpoint...')
        self._checkpoint()
        #self._load_precomputed_data(**kwargs)

        print(f'Experiment "{self.signature}" constructed successfully with {self.n_cells} neurons and {len(self.dynamic_features)} features')


    def check_ds(self, ds):
        if not hasattr(self, 'fps'):
            raise ValueError(f'fps not set for {self.signature}')

        time_step = 1.0/self.fps
        if time_step*ds>DEFAULT_MIN_BEHAVIOUR_TIME:
            print('Downsampling constant is too high: some behaviour acts may be skipped. '
                  f'Current minimal behaviour time interval is set to {DEFAULT_MIN_BEHAVIOUR_TIME} sec, '
                  f'downsampling {ds} will create time gaps of {time_step*ds} sec')

    def _build_pair_hash(self, cell_id, feat_id, mode='calcium'):
        '''
        Builds a unique hash-based representation of activity-feature pair data.
        feat_id should be a string or an iterable of strings (in case of joint MI calculation).
        '''
        if mode == 'calcium':
            act = self.neurons[cell_id].ca.data
        elif mode == 'spikes':
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

    def _build_data_hashes(self, mode='calcium'):
        '''
        Builds a unique hash-based representation of calcium-feature pair data for all cell-feature pairs..
        '''
        default_data_hashes = {dfeat: dict(zip(range(self.n_cells), [None for _ in range(self.n_cells)])) for dfeat in self.dynamic_features.keys()}
        self._data_hashes = {'calcium': default_data_hashes, 'spikes': default_data_hashes}
        for feat_id in self.dynamic_features:
            for cell_id in range(self.n_cells):
                self._data_hashes[mode][feat_id][cell_id] = self._build_pair_hash(cell_id, feat_id, mode=mode)

    def _trim_data(self, calcium, spikes, dynamic_features, bad_frames_mask, force_filter=False):

        if not force_filter and self.filtered_flag:
            raise AttributeError('Data is already filtered, if you want to force filtering it again, set "force_filter = True"')

        f_calcium = calcium[:, ~bad_frames_mask]
        if spikes is not None:
            f_spikes = spikes[:, ~bad_frames_mask]
        else:
            f_spikes = None

        f_dynamic_features = {}
        for feat_id in dynamic_features:
            current_ts = dynamic_features[feat_id]
            if isinstance(current_ts, TimeSeries):
                f_ts = TimeSeries(current_ts.data[~bad_frames_mask], discrete=current_ts.discrete)
            else:
                f_ts = TimeSeries(current_ts[~bad_frames_mask])

            f_dynamic_features[feat_id] = f_ts

        self.filtered_flag = True
        self.bad_frames_mask = bad_frames_mask

        return f_calcium, f_spikes, f_dynamic_features

    def _checkpoint(self):
        '''
        Check build for common errors
        '''
        if self.n_cells > self.n_frames:
            raise UserWarning('Number of cells > number of time frames, looks like the data is transposed')

        for dfeat in ['calcium', 'spikes']:
            if self.n_frames not in getattr(self, dfeat).shape:
                raise ValueError(f'"{dfeat}" feature has inappropriate shape: {getattr(self, dfeat).shape}'
                f'inconsistent with data length {self.n_frames}')

        for dfeat in self.dynamic_features.keys():
            if self.n_frames not in getattr(self, dfeat).data.shape:
                raise ValueError(f'"{dfeat}" feature has inappropriate shape: {getattr(self, dfeat).shape}'
                 f'inconsistent with data length {self.n_frames}')

    def _populate_cell_feat_dict(self, content, fbunch=None, cbunch=None):
        '''
        Helper function. Creates a nested dictionary of feature-cell pairs and populates every cell with 'content' variable.
        Outer dict: dynamic features, inner dict: cells
        '''
        cell_ids = self._process_cbunch(cbunch)
        feat_ids = self._process_fbunch(fbunch, allow_multifeatures=True)
        nested_dict = populate_nested_dict(content, feat_ids, cell_ids)
        return nested_dict

    def _process_cbunch(self, cbunch):
        '''
        Helper function. Turns cell indices (int, iterable or None) into a list of cell numbers.
        '''
        if isinstance(cbunch, int):
            cell_ids = [cbunch]
        elif cbunch is None:
            cell_ids = list(np.arange(self.n_cells))
        else:
            cell_ids = list(cbunch)

        return cell_ids

    def _process_fbunch(self, fbunch, allow_multifeatures=False):
        '''
        Helper function. Turns feature names (str, iterable or None) into a list of feature names
        '''
        if isinstance(fbunch, str):
            feat_ids = [fbunch]

        elif fbunch is None:  # default set of features
            if allow_multifeatures:
                try:
                    # stats table contains up-to-date set of features, including multifeatures
                    feat_ids = list(self.stats_table.keys())
                except:
                    # if stats is not available, take full set of single features
                    feat_ids = list(self.dynamic_features.keys())
            else:
                feat_ids = list(self.dynamic_features.keys())

        else:
            feat_ids = []

            # check for multifeatures
            for fname in fbunch:
                if isinstance(fname, str):
                    feat_ids.append(fname)
                else:
                    if allow_multifeatures:
                        feat_ids.append(tuple(sorted(list(fname))))
                    else:
                        raise ValueError('Multifeature detected in "allow_multifeatures=False" mode')

        return feat_ids

    def _process_sbunch(self, sbunch, significance_mode=False):
        '''
        Helper function. Turns stats type names (str, iterable or None) into a list of stats types
        '''
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

    def _add_multifeature_to_data_hashes(self, feat_id, mode='calcium'):
        '''
        Add previously unseen multifeature (e.g. ['x','y']) to table with data hashes.
        This function ignores multifeatures that already exist in the table.
        '''
        if (not isinstance(feat_id, str)) and len(feat_id) == 1:
            feat_id = feat_id[0]

        if not isinstance(feat_id, str):
            ordered_fnames = tuple(sorted(list(feat_id)))
            if ordered_fnames not in self._data_hashes[mode]:
                all_hashes = [self._build_pair_hash(cell_id, ordered_fnames) for cell_id in range(self.n_cells)]
                new_dict = {ordered_fnames: dict(zip(range(self.n_cells), all_hashes))}
                self._data_hashes[mode].update(new_dict)

        else:
            raise ValueError('This method is for multifeature update only')

    def _add_multifeature_to_stats(self, feat_id):
        '''
        Add previously unseen multifeature (e.g. ['x','y']) to statistics and significance tables.
        This function ignores multifeatures that already exist in the table.
        '''
        if (not isinstance(feat_id, str)) and len(feat_id) == 1:
            feat_id = feat_id[0]

        if not isinstance(feat_id, str):
            ordered_fnames = tuple(sorted(list(feat_id)))
            if ordered_fnames not in self.stats_table:
                print(f'Multifeature {feat_id} is new, it will be added to stats table')
                self.stats_table[ordered_fnames] = {cell_id: DEFAULT_STATS.copy() for cell_id in range(self.n_cells)}
                self.significance_table[ordered_fnames] = {cell_id: DEFAULT_SIGNIFICANCE.copy() for cell_id in range(self.n_cells)}

        else:
            raise ValueError('This method is for multifeature update only')

    def _check_stats_relevance(self, cell_id, feat_id, mode='calcium'):
        '''
        A guardian function that prevents access to non-existing and irrelevant data.

        This function checks whether the calcium-feature pair statistics has already been calculated.
        It ensures the data (both calcium and dynamic feature) has not changed since the last
        calculation by checking hash values of both data arrays.

        This function always refers to stats table but works equally well with significance table
        since they are always updated simultaneously
        '''

        if not isinstance(feat_id, str):
            feat_id = tuple(sorted(list(feat_id)))

        if feat_id not in self.stats_table:
            raise ValueError(f'Feature {feat_id} is not present in stats. \n If this is a single feature, '
                             'check the input data, since all single features are processed automatically.'
                             'If this is a multifeature (e.g. ["x", "y"]), compute MI significance to create stats')

        pair_hash = self._data_hashes[mode][feat_id][cell_id]
        existing_hash = self.stats_table[feat_id][cell_id]['data_hash']

        # if (stats does not exist yet) or (stats exists and data is the same):
        if (existing_hash is None) or (pair_hash == existing_hash):
            return True

        else:
            print(f'Looks like the data for the pair (cell {cell_id}, feature {feat_id}) '
                  'has been changed since the last calculation)')

            return False

    def _update_stats_and_significance(self, stats, cell_id, feat_id, stage2_only):
        '''
        Updates stats table and linked significance table to erase irrelevant data properly
        '''
        # update statistics
        self.stats_table[feat_id][cell_id].update(stats)
        if not stage2_only:
            # erase significance data completely since stats for stage 1 has been modified
            self.significance_table[feat_id][cell_id].update(DEFAULT_SIGNIFICANCE.copy())
        else:
            # erase significance data for stage 2 since stats for stage 2 has been modified
            self.significance_table[feat_id][cell_id].update({'stage2': None, 'shuffles2': None})

    def update_neuron_feature_pair_stats(self, stats, cell_id, feat_id, force_update=False, stage2_only=False):
        '''
        Updates calcium-feature pair statistics.
        feat_id should be a string or an iterable of strings (in case of joint MI calculation).
        This function allows multifeatures.
        '''

        if not isinstance(feat_id, str):
            self._add_multifeature_to_data_hashes(feat_id)
            self._add_multifeature_to_stats(feat_id)

        if self._check_stats_relevance(cell_id, feat_id):
            self._update_stats_and_significance(stats, cell_id, feat_id, stage2_only=stage2_only)

        else:
            if not force_update:
                print(f'To forcefully update the stats, set "force_update=True"')
            else:
                self._update_stats_and_significance(stats, cell_id, feat_id, stage2_only=stage2_only)

    def update_neuron_feature_pair_significance(self, sig, cell_id, feat_id, mode='calcium'):
        '''
        Updates calcium-feature pair significance data.
        feat_id should be a string or an iterable of strings (in case of joint MI calculation).
        This function allows multifeatures.
        '''
        if not isinstance(feat_id, str):
            self._add_multifeature_to_data_hashes(feat_id, mode=mode)
            self._add_multifeature_to_stats(feat_id)

        if self._check_stats_relevance(cell_id, feat_id):
            self.significance_table[feat_id][cell_id].update(sig)

        else:
            raise ValueError('Can not update significance table until the collision between actual data hashes and '
                             'saved stats data hashes is resolved. Use update_neuron_feature_pair_stats' 
                             'with "force_update=True" to forcefully rewrite statistics')

    def get_neuron_feature_pair_stats(self, cell_id, feat_id):
        '''
        Returns calcium-feature pair statistics.
        This function allows multifeatures.
        '''
        stats = None
        if self._check_stats_relevance(cell_id, feat_id):
            stats = self.stats_table[feat_id][cell_id]
        else:
            print(f'Consider recalculating stats')

        return stats

    def get_neuron_feature_pair_significance(self, cell_id, feat_id):
        '''
        Returns calcium-feature pair significance data.
        This function allows multifeatures.
        '''
        sig = None
        if self._check_stats_relevance(cell_id, feat_id):
            sig = self.significance_table[feat_id][cell_id]
        else:
            print(f'Consider recalculating stats')

        return sig

    def get_multicell_shuffled_calcium(self, cbunch=None, method='roll_based', no_ts=True, **kwargs):
        '''

        Args:
            cbunch:
            method:
            **kwargs:
            no_ts: if True, intermediate TimeSeries objects are nor created, which speeds up shuffling

        Returns:

        '''
        cell_list = self._process_cbunch(cbunch)
        agg_sh_data = np.zeros((len(cell_list), self.n_frames))
        for i, cell in enumerate(cell_list):
            sh_data = cell.get_shuffled_calcium(method=method, **kwargs, no_ts=no_ts)
            if no_ts:
                agg_sh_data[i, :] = sh_data[:]
            else:
                agg_sh_data[i, :] = sh_data.data[:]

        return agg_sh_data

    def get_multicell_shuffled_spikes(self, cbunch = None, method = 'isi_based', no_ts=True, **kwargs):
        '''

        Args:
            cbunch:
            method:
            **kwargs:
            no_ts: if True, intermediate TimeSeries objects are nor created, which speeds up shuffling

        Returns:

        '''
        if self.spikes is None:
            raise AttributeError('Unable to shuffle spikes without spikes data')

        cell_list = self._process_cbunch(cbunch)

        agg_sh_data = np.zeros((len(cell_list), self.n_frames))
        for i in cell_list:
            cell = self.neurons[i]
            sh_data = cell.get_shuffled_spikes(method=method, **kwargs)
            if no_ts:
                agg_sh_data[i, :] = sh_data[:]
            else:
                agg_sh_data[i, :] = sh_data.data[:]

        return agg_sh_data

    def get_stats_slice(self, cbunch=None, fbunch=None, sbunch=None, significance_mode=False):
        '''
        returns slice of accumulated statistics data (or significance data if "significance_mode=True")
        '''
        cell_ids = self._process_cbunch(cbunch)
        feat_ids = self._process_fbunch(fbunch, allow_multifeatures=True)
        slist = self._process_sbunch(sbunch, significance_mode=significance_mode)

        if significance_mode:
            full_table = self.significance_table
        else:
            full_table = self.stats_table

        out_table = self._populate_cell_feat_dict(dict(), fbunch=fbunch, cbunch=cbunch)
        for feat_id in feat_ids:
            for cell_id in cell_ids:
                out_table[feat_id][cell_id] = {s: full_table[feat_id][cell_id][s] for s in slist}

        return out_table

    def get_significance_slice(self, cbunch=None, fbunch=None, sbunch=None):
        return self.get_stats_slice(cbunch=cbunch, fbunch=fbunch, sbunch=sbunch, significance_mode=True)

    def get_feature_entropy(self, feat_id, ds=1):
        '''
        Calculates entropy of a single dynamic feature or a multifeature (e.g. ['x','y']).
        Currently only 2-combinations of features are correctly supported,
        for 3 and more variables calculations will be distorted (correct estimation of multivariate
        entropy for non-gaussian variables is non-trivial).
        '''
        if isinstance(feat_id, str):
            fts = self.dynamic_features[feat_id]
            return fts.get_entropy(ds=ds)

        else:
            ordered_fnames = tuple(sorted(list(feat_id)))
            tslist = [self.dynamic_features[dfeat] for dfeat in ordered_fnames]
            single_entropies = [fts.get_entropy(ds=ds) for fts in tslist]
            fpairs = list(combinations(tslist, 2))
            MIs = [get_1d_mi(ts1, ts2, ds=ds) for (ts1,ts2) in fpairs]
            return sum(single_entropies) - sum(MIs)

    def get_significant_neurons(self, min_nspec=1, fbunch=None):
        '''
        Returns a dict with neuron ids as keys and their significantly correlated features as values
        Only neurons with "min_nspec" or more significantly correlated features will be returned
        '''
        feat_ids = self._process_fbunch(fbunch, allow_multifeatures=True)
        relevance = [self._check_stats_relevance(cell_id, feat_id) for cell_id in range(self.n_cells) for feat_id in feat_ids]
        if not np.all(np.array(relevance)):
            raise ValueError('Stats relevance error')

        cell_ids = np.arange(self.n_cells)

        # TODO: add significance update and pval_thr argument
        cell_feat_dict = {cell_id: [] for cell_id in range(self.n_cells)}
        for i, cell_id in enumerate(cell_ids):
            for j, feat_id in enumerate(feat_ids):
                if self.significance_table[feat_id][cell_id]['stage2']:
                    cell_feat_dict[cell_id].append(feat_id)

        # filter out cells without specializations
        final_cell_feat_dict = {cell_id: cell_feat_dict[cell_id] for cell_id in range(self.n_cells) if len(cell_feat_dict[cell_id])>=min_nspec}

        return final_cell_feat_dict


    #===================================================================================
    # not active

    def save_mi_significance_to_file(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self.mi_significance_table, f)


    def clear_cells_mi_significance_data(self, cbunch, path_to_save = None):
        for cell_id in cbunch:
            for feat in self.dynamic_features:
                self.mi_significance_table[feat][cell_id] = {}

        if path_to_save is not None:
            self.save_mi_significance_to_file(path_to_save)

    def clear_features_mi_significance_data(self, feat_list, save_to_file = False):
        pass

    def clear_cell_feat_mi_significance_data(self, cell, feat, save_to_file = False):
        pass

    def _load_precomputed_data(self, **kwargs):
        if 'mi_significance' in kwargs:
            self.mi_significance_table = {**self.mi_significance_table, **kwargs['mi_significance']}
