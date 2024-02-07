import numpy as np
from sklearn.datasets import make_swiss_roll
from ..dim_reduction.data import *

n_gaussian = 10
n_swiss_roll = 1000

def create_default_data(n=n_gaussian, T=10000):
    C = np.zeros((n,n))
    C[1, 9] = 0.9
    C[2, 8] = 0.8
    C = (C + C.T)
    np.fill_diagonal(C, 1)
    signals = np.random.multivariate_normal(np.zeros(n),
                                            C,
                                            size=T,
                                            check_valid='raise').T
    return signals


def create_swiss_roll_data():
    random_state = 42
    data, _ = make_swiss_roll(n_samples=n_swiss_roll,
                              noise=0.0,
                              random_state=random_state,
                              hole=False)
    return data.T


def test_corrmat():
    data = create_default_data()
    D = MVData(data)
    cm = D.corr_mat()
    assert np.allclose(np.diag(cm), np.ones(n_gaussian))


def test_pca():
    data = create_swiss_roll_data()
    D = MVData(data)

    embedding_params = {
        'e_method_name': 'pca',
        'dim': 2
    }

    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]
    emb = D.get_embedding(embedding_params)
    print(emb.coords.shape)
