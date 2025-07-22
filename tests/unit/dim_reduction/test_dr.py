from sklearn.datasets import make_swiss_roll, make_s_curve, make_circles
import numpy as np
import pytest
import time
from driada.dim_reduction.data import *
from driada.experiment import Experiment
from driada.experiment.synthetic import (
    generate_synthetic_exp, 
    generate_circular_manifold_exp,
    generate_2d_manifold_exp
)
from driada.utils.data import create_correlated_gaussian_data

n_gaussian = 10
n_swiss_roll = 1000

def create_default_data(n=n_gaussian, T=10000):
    # Use the utility function with the same correlation pattern
    correlation_pairs = [(1, 9, 0.9), (2, 8, 0.8)]
    data, _ = create_correlated_gaussian_data(
        n_features=n,
        n_samples=T,
        correlation_pairs=correlation_pairs,
        seed=None  # Don't set seed to maintain original behavior
    )
    return data


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

    emb = D.get_embedding(method='pca', dim=2)
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0,0]) in [np.float64, np.float32]


def test_le():
    data = create_swiss_roll_data()
    D = MVData(data)

    emb = D.get_embedding(method='le', dim=2, nn=10, metric='l2')
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_auto_le():
    data = create_swiss_roll_data()
    D = MVData(data)

    emb = D.get_embedding(method='auto_le', dim=2, nn=10, metric='l2')
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_umap():
    data = create_swiss_roll_data()
    D = MVData(data)

    emb = D.get_embedding(method='umap', dim=2, min_dist=0.1, nn=10, metric='l2')
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_isomap():
    data = create_swiss_roll_data()
    D = MVData(data)

    emb = D.get_embedding(method='isomap', dim=2, nn=10, metric='l2')
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_tsne():
    data = create_swiss_roll_data()
    D = MVData(data)

    emb = D.get_embedding(method='tsne', dim=2)
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_auto_dmaps():
    data = create_swiss_roll_data()
    D = MVData(data)

    emb = D.get_embedding(method='auto_dmaps', dim=2, dm_alpha=1, nn=10, metric='l2')
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_ae_simple():
    data = create_swiss_roll_data()
    D = MVData(data)

    emb = D.get_embedding(
        method='ae',
        dim=2,
        epochs=200,
        lr=5 * 1e-3,
        seed=42,
        batch_size=512,
        feature_dropout=0.5,
        enc_kwargs={'dropout': 0.2},
        dec_kwargs={'dropout': 0.2}
    )

    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_ae_corr():
    data = create_swiss_roll_data()
    D = MVData(data)

    emb = D.get_embedding(
        method='ae',
        dim=2,
        epochs=200,
        lr=5 * 1e-3,
        seed=42,
        batch_size=512,
        feature_dropout=0.5,
        add_corr_loss=True,
        corr_hyperweight=1,
        enc_kwargs={'dropout': 0.2},
        dec_kwargs={'dropout': 0.2}
    )

    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]