import hashlib
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import hilbert
import numpy as np

def rescale(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    res = scaler.fit_transform(data.reshape(-1,1)).ravel()
    return res


def get_hash(data):
    # Prepare the object hash
    hash_id = hashlib.md5()
    hash_id.update(repr(data).encode('utf-8'))
    return hash_id.hexdigest()


def phase_synchrony(vec1, vec2):
    al1 = np.angle(hilbert(vec1), deg=False)
    al2 = np.angle(hilbert(vec2), deg=False)
    phase_synchrony = 1-np.sin(np.abs(al1-al2)/2)
    return phase_synchrony