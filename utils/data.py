from sklearn.preprocessing import MinMaxScaler

def rescale(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    res = scaler.fit_transform(data.reshape(-1,1)).ravel()
    return res