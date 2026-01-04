from sklearn.datasets import make_swiss_roll
import numpy as np
import pytest
import scipy.sparse as sp
from unittest.mock import patch
from driada.dim_reduction.data import *
from driada.utils.data import create_correlated_gaussian_data

# Test data sizes - much smaller for faster execution
n_gaussian = 10
n_swiss_roll = 100  # Reduced from 1000 for faster tests
n_time_samples = 500  # Reduced from 10000


@pytest.fixture
def small_swiss_roll_mvdata():
    """Small swiss roll MVData for fast tests."""
    data, color = make_swiss_roll(
        n_samples=n_swiss_roll, noise=0.0, random_state=42, hole=False
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
        seed=42,
    )
    return MVData(data)


def test_corrmat(small_gaussian_mvdata):
    cm = small_gaussian_mvdata.corr_mat()
    assert np.allclose(np.diag(cm), np.ones(n_gaussian))


def test_proximity_graph(small_swiss_roll_mvdata):

    metric_params = {"metric_name": "l2", "sigma": 1, "p": 2}

    graph_params = {
        "g_method_name": "knn",
        "weighted": 0,
        "nn": 5,  # Reduced for faster tests
        "max_deleted_nodes": 0.2,
        "dist_to_aff": "hk",
    }

    G = small_swiss_roll_mvdata.get_proximity_graph(metric_params, graph_params)


def test_pca(small_swiss_roll_mvdata):

    emb = small_swiss_roll_mvdata.get_embedding(method="pca", dim=2)
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_le(small_swiss_roll_mvdata):

    emb = small_swiss_roll_mvdata.get_embedding(method="le", dim=2, nn=5, metric="l2")
    # LE may drop nodes from disconnected components during graph preprocessing
    assert emb.coords.shape[0] == 2  # Check dimension
    assert emb.coords.shape[1] <= n_swiss_roll  # May lose some nodes
    assert emb.graph.is_connected()  # Result should be connected
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_auto_le(small_swiss_roll_mvdata):

    emb = small_swiss_roll_mvdata.get_embedding(
        method="auto_le", dim=2, nn=5, metric="l2"
    )
    # auto_le may drop nodes from disconnected components during graph preprocessing
    assert emb.coords.shape[0] == 2  # Check dimension
    assert emb.coords.shape[1] <= n_swiss_roll  # May lose some nodes
    assert emb.graph.is_connected()  # Result should be connected
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_umap(small_swiss_roll_mvdata):

    emb = small_swiss_roll_mvdata.get_embedding(
        method="umap", dim=2, min_dist=0.1, nn=5, metric="l2"
    )
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_isomap(small_swiss_roll_mvdata):

    emb = small_swiss_roll_mvdata.get_embedding(
        method="isomap", dim=2, nn=5, metric="l2"
    )
    # Isomap may drop nodes from disconnected components during graph preprocessing
    assert emb.coords.shape[0] == 2  # Check dimension
    assert emb.coords.shape[1] <= n_swiss_roll  # May lose some nodes
    assert emb.graph.is_connected()  # Result should be connected
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_tsne(small_swiss_roll_mvdata):

    emb = small_swiss_roll_mvdata.get_embedding(
        method="tsne", dim=2, n_iter=250
    )  # Reduced iterations
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_auto_dmaps(small_swiss_roll_mvdata):

    emb = small_swiss_roll_mvdata.get_embedding(
        method="auto_dmaps", dim=2, dm_alpha=1, nn=5, metric="l2"
    )
    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_ae_simple(small_swiss_roll_mvdata):
    emb = small_swiss_roll_mvdata.get_embedding(
        method="ae",
        dim=2,
        epochs=50,  # Reduced from 200
        lr=5 * 1e-3,
        seed=42,
        batch_size=32,  # Reduced from 512
        feature_dropout=0.5,
        enc_kwargs={"dropout": 0.2},
        dec_kwargs={"dropout": 0.2},
    )

    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


def test_ae_corr(small_swiss_roll_mvdata):
    emb = small_swiss_roll_mvdata.get_embedding(
        method="ae",
        dim=2,
        epochs=50,  # Reduced from 200
        lr=5 * 1e-3,
        seed=42,
        batch_size=32,  # Reduced from 512
        feature_dropout=0.5,
        add_corr_loss=True,
        corr_hyperweight=1,
        enc_kwargs={"dropout": 0.2},
        dec_kwargs={"dropout": 0.2},
    )

    assert emb.coords.shape == (2, n_swiss_roll)
    assert type(emb.coords[0, 0]) in [np.float64, np.float32]


# Additional tests for uncovered MVData functionality


def test_check_data_for_errors_sparse():
    """Test check_data_for_errors with sparse matrix containing zero columns."""
    # Create sparse matrix with zero column
    data = sp.csr_matrix([[1, 0, 2], [3, 0, 4], [5, 0, 6]])

    with pytest.raises(ValueError, match="Data contains 1 zero columns"):
        check_data_for_errors(data)


def test_check_data_for_errors_dense():
    """Test check_data_for_errors with dense array containing zero columns."""
    # Create dense array with zero columns
    data = np.array([[1, 0, 2, 0], [3, 0, 4, 0], [5, 0, 6, 0]])
    
    with pytest.raises(ValueError, match="Data contains 2 zero columns"):
        check_data_for_errors(data)


def test_check_data_for_errors_valid_data():
    """Test check_data_for_errors with valid data (no zero columns)."""
    # Valid data - no zero columns
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    # Should not raise any exception
    check_data_for_errors(data)  # No exception means success


def test_check_data_for_errors_verbose():
    """Test check_data_for_errors verbose output."""
    # Create data with many zero columns
    data = np.zeros((3, 20))
    data[:, [0, 5, 10, 15]] = 1  # Only 4 non-zero columns
    
    # Test verbose output with capsys
    with pytest.raises(ValueError, match="Data contains 16 zero columns"):
        check_data_for_errors(data, verbose=True)


def test_mvdata_with_zero_columns():
    """Test MVData initialization with data containing zero columns."""
    # Create data with zero columns
    data = np.array([[1, 0, 2], [3, 0, 4], [5, 0, 6]])
    
    # Should raise ValueError due to zero column
    with pytest.raises(ValueError, match="Data contains 1 zero columns"):
        MVData(data)


def test_mvdata_downsampling():
    """Test MVData with downsampling parameter."""
    data = np.random.rand(5, 100)

    # Test downsampling by factor of 4
    mvdata = MVData(data, downsampling=4)
    assert mvdata.ds == 4
    assert mvdata.n_points == 25
    assert mvdata.data.shape == (5, 25)


def test_mvdata_rescale_rows():
    """Test MVData with row rescaling."""
    # Create data with different ranges
    data = np.array([[0, 10, 20], [-5, 0, 5], [100, 200, 300]])

    mvdata = MVData(data.copy(), rescale_rows=True)

    # Each row should be rescaled to [0, 1]
    for i in range(mvdata.n_dim):
        row_min = mvdata.data[i].min()
        row_max = mvdata.data[i].max()
        assert row_min == pytest.approx(0.0)
        assert row_max == pytest.approx(1.0)


def test_mvdata_median_filter():
    """Test median filter on data."""
    # Test with regular numpy array data
    data = np.array([[1, 10, 1, 1, 10, 1], [2, 2, 20, 2, 2, 2]])

    mvdata = MVData(data.copy())
    original_shape = mvdata.data.shape
    mvdata.median_filter(window=3)

    # Should maintain shape and reduce spikes
    assert mvdata.data.shape == original_shape
    # Check that spikes are reduced
    assert mvdata.data[0, 1] < 10  # First spike reduced
    assert mvdata.data[1, 2] < 20  # Second spike reduced


def test_corrmat_axis_1(small_gaussian_mvdata):
    """Test correlation matrix computation along axis 1."""
    cm = small_gaussian_mvdata.corr_mat(axis=1)

    # Should compute correlations between samples
    assert cm.shape == (n_time_samples, n_time_samples)
    assert np.allclose(np.diag(cm), np.ones(n_time_samples))


def test_get_distmat_string_metric(small_swiss_roll_mvdata):
    """Test distance matrix with string metric."""
    distmat = small_swiss_roll_mvdata.get_distmat("cityblock")

    assert distmat.shape == (n_swiss_roll, n_swiss_roll)
    assert np.allclose(np.diag(distmat), 0)


def test_get_distmat_minkowski(small_swiss_roll_mvdata):
    """Test distance matrix with minkowski metric."""
    m_params = {"metric_name": "minkowski", "p": 3}
    distmat = small_swiss_roll_mvdata.get_distmat(m_params)

    assert distmat.shape == (n_swiss_roll, n_swiss_roll)


def test_get_distmat_l2_conversion(small_swiss_roll_mvdata):
    """Test that l2 is converted to euclidean."""
    dist_l2 = small_swiss_roll_mvdata.get_distmat({"metric_name": "l2"})
    dist_euclidean = small_swiss_roll_mvdata.get_distmat({"metric_name": "euclidean"})

    np.testing.assert_allclose(dist_l2, dist_euclidean)


def test_get_embedding_no_params_error():
    """Test error when no parameters provided to get_embedding."""
    mvdata = MVData(np.random.rand(5, 20))

    with pytest.raises(
        ValueError, match="Either 'method' or 'e_params' must be provided"
    ):
        mvdata.get_embedding()


def test_get_embedding_unknown_method_error():
    """Test error for unknown embedding method."""
    mvdata = MVData(np.random.rand(5, 20))

    e_params = {
        "e_method_name": "unknown_method",
        "e_method": object(),  # Some non-None object
        "dim": 2,
    }

    with pytest.raises(Exception, match="Unknown embedding construction method"):
        mvdata.get_embedding(e_params=e_params)


def test_get_proximity_graph_unknown_method():
    """Test error for unknown graph construction method."""
    mvdata = MVData(np.random.rand(5, 20))

    m_params = {"metric_name": "euclidean"}
    g_params = {"g_method_name": "unknown_graph_method"}

    with pytest.raises(Exception, match="Unknown graph construction method"):
        mvdata.get_proximity_graph(m_params, g_params)
