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

# Test data sizes - much smaller for faster execution
n_gaussian = 10
n_swiss_roll = 100  # Reduced from 1000 for faster tests
n_time_samples = 500  # Reduced from 10000


@pytest.fixture
def small_swiss_roll_mvdata():
    """Small swiss roll MVData for fast tests."""
    data, color = make_swiss_roll(
        n_samples=n_swiss_roll, 
        noise=0.0, 
        random_state=42,
        hole=False
    )
    return MVData(data.T)


@pytest.fixture
def small_gaussian_mvdata():
    """Small gaussian MVData for fast tests."""
    correlation_pairs = [(1, 9, 0.9), (2, 8, 0.8)]
    data, _ = create_correlated_gaussian_data(
        n_features=n_gaussian,
        n_samples=n_time_samples,
        correlation_pairs=correlation_pairs,
        seed=42
    )
    return MVData(data)


def test_corrmat(small_gaussian_mvdata):
    cm = small_gaussian_mvdata.corr_mat()
    assert np.allclose(np.diag(cm), np.ones(n_gaussian))


def test_proximity_graph(small_swiss_roll_mvdata):

    metric_params = {
        'metric_name': 'l2',
        'sigma': 1,
        'p': 2
    }

    graph_params = {
        'g_method_name': 'knn',
        'weighted': 0,
        'nn': 5,  # Reduced for faster tests
        'max_deleted_nodes': 0.2,
        'dist_to_aff': 'hk'
    }

    G = small_swiss_roll_mvdata.get_proximity_graph(metric_params, graph_params)


def test_pca(small_swiss_roll_mvdata):

    emb = small_swiss_roll_mvdata.get_embedding(method='pca', dim=2)
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0,0]) in [np.float64, np.float32]


def test_le(small_swiss_roll_mvdata):

    emb = small_swiss_roll_mvdata.get_embedding(method='le', dim=2, nn=5, metric='l2')
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_auto_le(small_swiss_roll_mvdata):

    emb = small_swiss_roll_mvdata.get_embedding(method='auto_le', dim=2, nn=5, metric='l2')
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_umap(small_swiss_roll_mvdata):

    emb = small_swiss_roll_mvdata.get_embedding(method='umap', dim=2, min_dist=0.1, nn=5, metric='l2')
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_isomap(small_swiss_roll_mvdata):

    emb = small_swiss_roll_mvdata.get_embedding(method='isomap', dim=2, nn=5, metric='l2')
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_tsne(small_swiss_roll_mvdata):

    emb = small_swiss_roll_mvdata.get_embedding(method='tsne', dim=2, n_iter=250)  # Reduced iterations
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_auto_dmaps(small_swiss_roll_mvdata):

    emb = small_swiss_roll_mvdata.get_embedding(method='auto_dmaps', dim=2, dm_alpha=1, nn=5, metric='l2')
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_ae_simple(small_swiss_roll_mvdata):
    emb = small_swiss_roll_mvdata.get_embedding(
        method='ae',
        dim=2,
        epochs=50,  # Reduced from 200
        lr=5 * 1e-3,
        seed=42,
        batch_size=32,  # Reduced from 512
        feature_dropout=0.5,
        enc_kwargs={'dropout': 0.2},
        dec_kwargs={'dropout': 0.2}
    )

    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_ae_corr(small_swiss_roll_mvdata):
    emb = small_swiss_roll_mvdata.get_embedding(
        method='ae',
        dim=2,
        epochs=50,  # Reduced from 200
        lr=5 * 1e-3,
        seed=42,
        batch_size=32,  # Reduced from 512
        feature_dropout=0.5,
        add_corr_loss=True,
        corr_hyperweight=1,
        enc_kwargs={'dropout': 0.2},
        dec_kwargs={'dropout': 0.2}
    )

    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]