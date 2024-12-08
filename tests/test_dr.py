from sklearn.datasets import make_swiss_roll
import numpy as np
from src.driada.dim_reduction.data import *

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


def test_proximity_graph():
    data = create_swiss_roll_data()

    metric_params = {
        'metric_name': 'l2',
        'sigma': 1,
        'p': 2
    }

    graph_params = {
        'g_method_name': 'knn',
        'weighted': 0,
        'nn': 10,
        'max_deleted_nodes': 0.2,
        'dist_to_aff': 'hk'
    }

    D = MVData(data)
    G = D.get_proximity_graph(metric_params, graph_params)


def test_pca():
    data = create_swiss_roll_data()
    D = MVData(data)

    embedding_params = {
        'e_method_name': 'pca',
        'dim': 2
    }

    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]
    emb = D.get_embedding(embedding_params)
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0,0]) in [np.float64, np.float32]


def test_le():
    data = create_swiss_roll_data()
    D = MVData(data)

    metric_params = {
        'metric_name': 'l2',
        'sigma': 1,
        'p': 2
    }

    graph_params = {
        'g_method_name': 'knn',
        'weighted': 0,
        'nn': 10,
        'max_deleted_nodes': 0.2,
        'dist_to_aff': 'hk'
    }

    embedding_params = {
        'e_method_name': 'le',
        'dim': 2
    }

    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]
    emb = D.get_embedding(embedding_params, g_params=graph_params, m_params=metric_params)
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_auto_le():
    data = create_swiss_roll_data()
    D = MVData(data)

    metric_params = {
        'metric_name': 'l2',
        'sigma': 1,
        'p': 2
    }

    graph_params = {
        'g_method_name': 'knn',
        'weighted': 0,
        'nn': 10,
        'max_deleted_nodes': 0.2,
        'dist_to_aff': 'hk'
    }

    embedding_params = {
        'e_method_name': 'auto_le',
        'dim': 2
    }

    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]
    emb = D.get_embedding(embedding_params, g_params=graph_params, m_params=metric_params)
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_umap():
    data = create_swiss_roll_data()
    D = MVData(data)

    metric_params = {
        'metric_name': 'l2',
        'sigma': 1,
        'p': 2
    }

    graph_params = {
        'g_method_name': 'knn',
        'weighted': 0,
        'nn': 10,
        'max_deleted_nodes': 0.2,
        'dist_to_aff': 'hk'
    }

    embedding_params = {
        'e_method_name': 'umap',
        'dim': 2,
        'min_dist': 0.1
    }

    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]
    emb = D.get_embedding(embedding_params, g_params=graph_params, m_params=metric_params)
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_isomap():
    data = create_swiss_roll_data()
    D = MVData(data)

    metric_params = {
        'metric_name': 'l2',
        'sigma': 1,
        'p': 2
    }

    graph_params = {
        'g_method_name': 'knn',
        'weighted': 0,
        'nn': 10,
        'max_deleted_nodes': 0.2,
        'dist_to_aff': 'hk'
    }

    embedding_params = {
        'e_method_name': 'isomap',
        'dim': 2,
        'min_dist': 0.1
    }

    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]
    emb = D.get_embedding(embedding_params, g_params=graph_params, m_params=metric_params)
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_tsne():
    data = create_swiss_roll_data()
    D = MVData(data)

    embedding_params = {
        'e_method_name': 'tsne',
        'dim': 2,
    }

    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]
    emb = D.get_embedding(embedding_params)
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_auto_dmaps():
    data = create_swiss_roll_data()
    D = MVData(data)

    metric_params = {
        'metric_name': 'l2',
        'sigma': 1,
        'p': 2
    }

    graph_params = {
        'g_method_name': 'knn',
        'weighted': 0,
        'nn': 10,
        'max_deleted_nodes': 0.2,
        'dist_to_aff': 'hk'
    }

    embedding_params = {
        'e_method_name': 'auto_dmaps',
        'dim': 2,
        'dm_alpha': 1
    }

    embedding_params['e_method'] = METHODS_DICT[embedding_params['e_method_name']]
    emb = D.get_embedding(embedding_params, g_params=graph_params, m_params=metric_params)
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]
