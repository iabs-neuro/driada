"""Isomap must embed the geodesics of the proximity graph exactly as built."""

import numpy as np
from scipy.sparse.csgraph import shortest_path
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap

from driada.dim_reduction import MVData


def _same_up_to_sign(a, b, tol):
    return all(
        min(np.abs(a[i] - b[i]).max(), np.abs(a[i] + b[i]).max()) < tol
        for i in range(a.shape[0])
    )


def _swiss(n=400):
    return make_swiss_roll(n, noise=0.2, random_state=0)[0].T


def test_isomap_is_mds_of_graph_geodesics():
    """Coordinates are classical MDS of the shortest paths on emb.graph."""
    emb = MVData(_swiss()).get_embedding(method="isomap", dim=2, n_neighbors=10)
    geo = shortest_path(emb.graph.adj, method="D", directed=False)
    ref = KernelPCA(n_components=2, kernel="precomputed").fit_transform(-0.5 * geo**2).T
    assert _same_up_to_sign(emb.coords, ref, 1e-8)


def test_isomap_matches_sklearn_when_degrees_do_not_exceed_k():
    """On a mutual k-NN graph (degree <= k) the result equals sklearn's Isomap
    applied to the same geodesics, so only union graphs change."""
    emb = MVData(_swiss()).get_embedding(
        method="isomap", dim=2, g_params={"nn": 10, "symmetrization": "intersection"}
    )
    geo = shortest_path(emb.graph.adj, method="D", directed=False)
    ref = Isomap(n_components=2, n_neighbors=10, metric="precomputed").fit_transform(geo).T
    scale = np.abs(ref).max()
    assert _same_up_to_sign(emb.coords, ref, 1e-6 * scale)


def test_isomap_on_degenerate_geodesic_spectrum():
    """Two tight clusters joined by an unweighted graph give integer hop-count
    geodesics, whose kernel has a repeated eigenvalue right at the cut between
    kept and dropped components. The embedding must still come out, with the
    variance of each axis equal to the corresponding kernel eigenvalue."""
    rng = np.random.RandomState(42)
    data = np.hstack([rng.randn(10, 5) * 0.1, rng.randn(10, 5) * 0.1 + 100])
    emb = MVData(data).get_embedding(
        method="isomap", n_neighbors=5, dim=2, max_deleted_nodes=0.6
    )

    geo = shortest_path(emb.graph.adj, method="D", directed=False)
    n = geo.shape[0]
    centering = np.eye(n) - 1.0 / n
    eigvals = np.linalg.eigvalsh(centering @ (-0.5 * geo**2) @ centering)[::-1]
    assert np.isclose(eigvals[1], eigvals[2])  # the cut falls inside a repeated eigenvalue

    assert emb.coords.shape == (2, n)
    assert np.allclose((emb.coords**2).sum(axis=1), eigvals[:2])
