"""K-nearest neighbor estimators for mutual information and entropy.

This module implements the Kraskov-Stögbauer-Grassberger (KSG) estimator
and related k-NN based information theoretic measures.

Credits:
    Original implementation by Greg Ver Steeg
    http://www.isi.edu/~gregv/npeet.html
    
References:
    Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
    Estimating mutual information. Physical Review E, 69(6), 066138.
"""

import numpy as np
import numpy.linalg as la
from numpy import log
from sklearn.neighbors import BallTree, KDTree

from .info_utils import py_fast_digamma

DEFAULT_NN = 5
# UTILITY FUNCTIONS

# TODO: add automatic alpha selection for LNC correction from https://github.com/BiuBiuBiLL/NPEET_LNC


def add_noise(x, ampl=1e-10):
    # small noise to break degeneracy, see doc.
    return x + ampl * np.random.random_sample(x.shape)


def query_neighbors(tree, x, k):
    # return tree.query(x, k=k+1, breadth_first = False)[0][:, k]
    return tree.query(x, k=k + 1)[0][:, k]


def _count_neighbors_single(tree, x, radii, ind):
    dists, indices = tree.query(
        x[ind : ind + 1], k=DEFAULT_NN, distance_upper_bound=radii[ind]
    )
    return len(np.unique(indices[0])) - 2


def count_neighbors(tree, x, radii):
    return tree.query_radius(x, radii, count_only=True)
    # dists, indices = tree.query(x, k=DEFAULT_NN, distance_upper_bound=r)
    # out = tree.query(x, k=DEFAULT_NN, distance_upper_bound=r)
    # return np.array([_count_neighbors_single(tree, x, radii, ind) for ind in range(len(x))])
    # return np.array([len(nn)-1 for nn in tree.query_ball_point(x, radii)])


def build_tree(points, lf=5):
    if points.shape[1] >= 20:
        return BallTree(points, metric="chebyshev")

    return KDTree(points, metric="chebyshev", leaf_size=lf)
    # return KDTree(points, leafsize = lf)
    # return KDTree(points, copy_data=True, leafsize = 5)


def avgdigamma(points, dvec, lf=30, tree=None):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    if tree is None:
        tree = build_tree(points, lf=lf)

    dvec = dvec - 1e-15
    num_points = count_neighbors(tree, points, dvec)
    num_points = num_points.astype(float)

    zero_inds = np.where(num_points == 0)[0]
    if 1.0 * len(zero_inds) / len(num_points) > 0.01:
        raise Exception("No neighbours in more than 1% points, check input!")
    else:
        if len(zero_inds) != 0:
            num_points[zero_inds] = 0.5

    # inf_inds = np.where(digamma(num_points) == -np.inf)
    # print(num_points[inf_inds])

    digammas = list(map(py_fast_digamma, num_points))
    return np.mean(digammas)


# CONTINUOUS ESTIMATORS


def nonparam_entropy_c(x, k=DEFAULT_NN, base=np.e):
    """The classic K-L k-nearest neighbor continuous entropy estimator."""
    # assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    # xs_columns = np.expand_dims(xs, axis=0).T
    x = np.asarray(x)
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    n_elements, n_features = x.shape
    x = add_noise(x)
    tree = build_tree(x)
    nn = query_neighbors(tree, x, k)
    const = py_fast_digamma(n_elements) - py_fast_digamma(k) + n_features * log(2)
    return (const + n_features * np.log(nn).mean()) / log(base)


def nonparam_cond_entropy_cc(x, y, k=DEFAULT_NN, base=np.e):
    """The classic K-L k-nearest neighbor continuous entropy estimator for the
    entropy of X conditioned on Y.
    """
    xy = np.c_[x, y]
    entropy_union_xy = nonparam_entropy_c(xy, k=k, base=base)
    entropy_y = nonparam_entropy_c(y, k=k, base=base)
    return entropy_union_xy - entropy_y


def nonparam_mi_cc(
    x,
    y,
    z=None,
    k=DEFAULT_NN,
    base=np.e,
    alpha=0,
    lf=5,
    precomputed_tree_x=None,
    precomputed_tree_y=None,
):
    """
    Mutual information of x and y (conditioned on z if z is not None)
    """

    assert len(x) == len(y), "Arrays should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"

    x, y = np.asarray(x), np.asarray(y)
    x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
    x = add_noise(x)
    y = add_noise(y)

    points = [x, y]
    if z is not None:
        z = np.asarray(z)
        z = z.reshape(z.shape[0], -1)
        points.append(z)

    points = np.hstack(points)

    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = build_tree(points, lf=lf)
    dvec = query_neighbors(tree, points, k)

    if z is None:
        a = avgdigamma(x, dvec, tree=precomputed_tree_x, lf=lf)
        b = avgdigamma(y, dvec, tree=precomputed_tree_y, lf=lf)
        c = py_fast_digamma(k)
        d = py_fast_digamma(len(x))

        # print(a, b, c, d)

        if alpha > 0:
            d += lnc_correction(tree, points, k, alpha)
    else:
        xz = np.c_[x, z]
        yz = np.c_[y, z]
        a, b, c, d = (
            avgdigamma(xz, dvec),
            avgdigamma(yz, dvec),
            avgdigamma(z, dvec),
            py_fast_digamma(k),
        )

    return (-a - b + c + d) / log(base)


def lnc_correction(tree, points, k, alpha):
    e = 0
    n_sample = points.shape[0]
    for point in points:
        # Find k-nearest neighbors in joint space, p=inf means max norm
        knn = tree.query(point[None, :], k=k + 1, return_distance=False)[0]
        knn_points = points[knn]
        # Substract mean of k-nearest neighbor points
        knn_points = knn_points - knn_points[0]
        # Calculate covariance matrix of k-nearest neighbor points, obtain eigen vectors
        covr = knn_points.T @ knn_points / k
        _, v = la.eig(covr)
        # Calculate PCA-bounding box using eigen vectors
        V_rect = np.log(np.abs(knn_points @ v).max(axis=0)).sum()
        # Calculate the volume of original box
        log_knn_dist = np.log(np.abs(knn_points).max(axis=0)).sum()

        # Perform local non-uniformity checking and update correction term
        if V_rect < log_knn_dist + np.log(alpha):
            e += (log_knn_dist - V_rect) / n_sample
    return e
