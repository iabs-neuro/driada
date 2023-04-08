from sklearn.preprocessing import MinMaxScaler
import hashlib

def rescale(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    res = scaler.fit_transform(data.reshape(-1,1)).ravel()
    return res


def get_hash(data):
    # Prepare the object hash
    hash_id = hashlib.md5()
    hash_id.update(repr(data).encode('utf-8'))
    return hash_id.hexdigest()
