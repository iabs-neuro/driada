import hashlib
import h5py
import scipy.sparse as ssp
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import hilbert
import numpy as np
import scipy.stats as st
from numba import njit


def populate_nested_dict(content, outer, inner):
    nested_dict = {o: {} for o in outer}
    for o in outer:
        nested_dict[o] = {i: content.copy() for i in inner}

    return nested_dict


def nested_dict_to_seq_of_tables(datadict, ordered_names1=None, ordered_names2=None):
    names1 = list(datadict.keys())
    names2 = list(datadict[names1[0]].keys())
    datakeys = list(datadict[names1[0]][names2[0]].keys())

    #print(names1)
    #print(names2)
    #print(datakeys)
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
    # renaming for convenience
    n1 = len(datadict.keys())
    n2 = len(datadict[list(datadict.keys())[0]])

    if not (names1 is None and names2 is None):
        if names1 is None:
            names1 = range(n1)
        if names2 is None:
            names2 = range(n2)

        renamed_dict = populate_nested_dict(dict(), names1, names2)
        for i in range(n1):
            for j in range(n2):
                renamed_dict[names1[i]][names2[j]].update(datadict[i][j])
        return renamed_dict

    else:
        return datadict


def retrieve_relevant_from_nested_dict(nested_dict,
                                       target_key,
                                       target_value,
                                       operation='=',
                                       allow_missing_keys=False):
    relevant_pairs = []
    for key1 in nested_dict.keys():
        for key2 in nested_dict[key1].keys():
            data = nested_dict[key1][key2]
            if target_key not in data and not allow_missing_keys:
                raise ValueError(f'Target key {target_key} not found in data dict')

            if operation == '=':
                criterion = data.get(target_key) == target_value
            elif operation == '>':
                criterion = data.get(target_key) > target_value if data.get(target_key) is not None else False
            elif operation == '<':
                criterion = data.get(target_key) < target_value if data.get(target_key) is not None else False
            else:
                raise ValueError(f'Operation should be one of "=", ">", "<", but {operation} was passed')

            if criterion:
                relevant_pairs.append((key1, key2))

    return relevant_pairs


def rescale(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    res = scaler.fit_transform(data.reshape(-1, 1)).ravel()
    return res


def get_hash(data):
    # Prepare the object hash
    hash_id = hashlib.md5()
    hash_id.update(repr(data).encode('utf-8'))
    return hash_id.hexdigest()


def phase_synchrony(vec1, vec2):
    al1 = np.angle(hilbert(vec1), deg=False)
    al2 = np.angle(hilbert(vec2), deg=False)
    phase_sync = 1-np.sin(np.abs(al1-al2)/2)
    return phase_sync


def correlation_matrix_old(a, b):
    if np.allclose(a, b):
        return np.corrcoef(a,a)
    else:
        n1 = a.shape[0]
        n2 = b.shape[0]
        corrmat = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                corrmat[i,j] = st.pearsonr(a[i,:], b[j,:])[0]

        return corrmat


def correlation_matrix(A):
    '''
    # fast implementation.
    A: numpy array of shape (ndims, nvars)

    returns: numpy array of shape (nvars, nvars)
    '''

    am = A - np.mean(A, axis=1, keepdims=True)
    return am @ am.T / np.sum(am**2, axis=1, keepdims=True).T


def cross_correlation_matrix(A, B):
    '''
    # fast implementation.

    A: numpy array of shape (ndims, nvars1)
    B: numpy array of shape (ndims, nvars2)

    returns: numpy array of shape (nvars1, nvars2)
    '''
    am = A - np.mean(A, axis=0, keepdims=True)
    bm = B - np.mean(B, axis=0, keepdims=True)
    return am.T @ bm / (np.sqrt(np.sum(am**2, axis=0, keepdims=True)).T * np.sqrt(np.sum(bm**2, axis=0, keepdims=True)))


# TODO: review this function
def norm_cross_corr(a, b):
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
    c = np.correlate(a, b, 'full')
    return c


def to_numpy_array(data):
    if isinstance(data, np.ndarray):
        return data

    if ssp.issparse(data):
        return data.A
    else:
        return np.array(data)


def write_dict_to_hdf5(data, hdf5_file, group_name=''):
    """
    Recursively writes a dictionary to an HDF5 file.

    Parameters:
        data (dict): The dictionary to write.
        hdf5_file (str): The path to the HDF5 file.
        group_name (str): The name of the current group in the HDF5 file.
    """
    with h5py.File(hdf5_file, 'a') as f:
        # Create a new group or get existing one
        group = f.create_group(group_name) if group_name else f

        for key, value in data.items():
            print(key)
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

    Parameters:
        hdf5_file (str): The path to the HDF5 file.

    Returns:
        dict: A nested dictionary representing the contents of the HDF5 file.
    """

    def _read_group(group):
        """
        Recursively reads an HDF5 group and converts it to a dictionary.

        Parameters:
            group (h5py.Group): The HDF5 group to read.

        Returns:
            dict: A dictionary representation of the group.
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

    with h5py.File(hdf5_file, 'r') as f:
        return _read_group(f)
