"""
Utility functions.

Authors:
    Junki Ishikawa, Takahiro Inagaki, Shuhei Yokoo 
"""

import numpy as np
from scipy.linalg import eigh
from scipy.linalg import svd

try:
    import torch
except ImportError:
    pass


def mean_square_singular_values(X):
    """
    calculate mean square of singular values of X

    Parameters:
    -----------
    X : array-like, shape: (n, m)

    Returns:
    --------
    c: mean square of singular values
    """

    # _, s, _ = np.linalg.svd(X)
    # mssv = (s ** 2).mean()

    # Frobenius norm means square root of
    # sum of square singular values
    mssv = (X * X).sum() / min(X.shape)
    return mssv


def canonical_angle(X, Y):
    """
    Calculate cannonical angles beween subspaces

    Parameters
    ----------
    X: basis matrix, array-like, shape: (n_subdim_X, n_dim)
    Y: basis matrix, array-like, shape: (n_subdim_Y, n_dim)

    Returns
    -------
    c: float, similarity of X, Y

    """

    return mean_square_singular_values(Y @ X.T)


def canonical_angle_matrix(X, Y):
    """Calculate canonical angles between subspaces
    example     similarity = MathUtils.calc_basis_vector(X, Y)

    Parameters
    ----------
    X: set of basis matrix, array-like, shape: (n_set_X, n_subdim, n_dim)
        n_subdim can be variable on each subspaces
    Y: set of basis matrix, array-like, shape: (n_set_Y, n_subdim, n_dim)
        n_set can be variable from n_set of X
        n_subdim can be variable on each subspaces

    Returns:
        C: similarity matrix, array-like, shape: (n_set_X, n_set_Y)

    """

    n_set_X, n_set_Y = len(X), len(Y)
    C = np.zeros((n_set_X, n_set_Y))
    for x in range(n_set_X):
        for y in range(n_set_Y):
            C[x, y] = canonical_angle(X[x], Y[y])

    return C


# faster method
def canonical_angle_matrix_f(X, Y):
    """Calculate canonical angles between subspaces
    example     similarity = MathUtils.calc_basis_vector(X, Y)

    Parameters
    ----------
    X: set of basis matrix, array-like, shape: (n_set_X, n_subdim, n_dim)
        n_subdim can be variable on each subspaces
    Y: set of basis matrix, array-like, shape: (n_set_Y, n_subdim, n_dim)
        n_set can be variable from n_set of X
        n_subdim can be variable on each subspaces

    Returns:
        C: similarity matrix, array-like, shape: (n_set_X, n_set_Y)

    """
    X = np.transpose(X, (0, 2, 1))
    D = np.dot(Y, X)
    _D = np.transpose(D, (0, 2, 1, 3))
    _, C, _ = np.linalg.svd(_D)
    sim = C**2
    return sim.mean(2)


def _eigh(X, eigvals=None):
    """
    A wrapper function of numpy.linalg.eigh and scipy.linalg.eigh

    Parameters
    ----------
    X: array-like, shape (a, a)
        target symmetric matrix
    eigvals: tuple, (lo, hi)
        Indexes of the smallest and largest (in ascending order) eigenvalues and corresponding eigenvectors
        to be returned: 0 <= lo <= hi <= M-1. If omitted, all eigenvalues and eigenvectors are returned.

    Returns
    -------
    e: array-like, shape (a) or (n_dims)
        eigenvalues with descending order
    V: array-like, shape (a, a) or (a, n_dims)
        eigenvectors
    """

    if eigvals != None:
        e, V = eigh(X, eigvals=eigvals)
    else:
        # numpy's eigh is faster than scipy's when all calculating eigenvalues and eigenvectors
        e, V = np.linalg.eigh(X)

    e, V = e[::-1], V[:, ::-1]

    return e, V


def _eigen_basis(X, eigvals=None):
    """
    Return subspace basis using PCA

    Parameters
    ----------
    X: array-like, shape (a, a)
        target matrix
    n_dims: integer
        number of basis

    Returns
    -------
    e: array-like, shape (a) or (n_dims)
        eigenvalues with descending order
    V: array-like, shape (a, a) or (a, n_dims)
        eigenvectors
    """

    try:
        e, V = _eigh(X, eigvals=eigvals)
    except np.linalg.LinAlgError:
        # if it not converges, try with tiny salt
        salt = 1e-8 * np.eye(X.shape[0])
        e, V = eigh(X + salt, eigvals=eigvals)

    return e, V


def _get_eigvals(n, n_subdims, higher):
    """
    Culculate eigvals for eigh
    
    Parameters
    ----------
    n: int
    n_subdims: int, dimension of subspace
    higher: boolean, if True, use higher `n_subdim` basis

    Returns
    -------
    eigvals: tuple of 2 integers
    """

    if n_subdims is None:
        return None

    if higher:
        low = max(0, n - n_subdims)
        high = n - 1
    else:
        low = 0
        high = min(n - 1, n_subdims - 1)

    return low, high


def subspace_bases(X, n_subdims=None, higher=True):
    """
    Return subspace basis using PCA

    Parameters
    ----------
    X: array-like, shape (n_dimensions, n_vectors)
        data matrix
    n_subdims: integer
        number of subspace dimension

    Returns
    -------
    V: array-like, shape (n_dimensions, n_subdims)
        bases matrix
    """
    eigvals = _get_eigvals(X.shape[0], n_subdims, higher)

    # get eigenvector of autocorrelation matrix X @ X.T
    _, V = _eigen_basis(np.dot(X, X.T), eigvals=eigvals)

    return V


def dual_vectors(K, n_subdims=None, higher=True, eps=1e-20):
    """
    Calc dual representation of vectors in kernel space

    Parameters:
    -----------
    K :  array-like, shape: (n_samples, n_samples)
        Grammian Matrix of X: K(X, X)
    n_subdims: int, default=None
        Number of vectors of dual vectors to return
    higher: boolean, default=None
        If True, this function returns eigenbasis corresponding to 
            higher `n_subdims` eigenvalues in descending order.
        If False, this function returns eigenbasis corresponding to 
            lower `n_subdims` eigenvalues in descending order.
    eps: float, default=1e-20
        lower limit of eigenvalues

    Returns:
    --------
    A : array-like, shape: (n_samples, n_samples)
        Dual replesentation vectors.
        it satisfies lambda[i] * A[i] @ A[i] == 1, where lambda[i] is i-th biggest eigenvalue
    e:  array-like, shape: (n_samples, )
        Eigen values descending sorted
    """

    eigvals = _get_eigvals(K.shape[0], n_subdims, higher)
    e, A = _eigen_basis(K, eigvals=eigvals)

    # replace if there are too small eigenvalues
    e[(e < eps)] = eps

    A = A / np.sqrt(e)

    return A, e


def cross_similarities(refs, inputs):
    """
    Calc similarities between each reference spaces and each input subspaces

    Parameters:
    -----------
    refs: list of array-like (n_dims, n_subdims_i)
    inputs: list of array-like (n_dims, n_subdims_j)

    Returns:
    --------
    similarities: array-like, shape (n_refs, n_inputs)
    """

    similarities = []
    for _input in inputs:
        sim = []
        for ref in refs:
            sim.append(mean_square_singular_values(ref.T @ _input))
        similarities.append(sim)

    similarities = np.array(similarities)

    return similarities


def randomized_time_warping(
    inputs,
    n_sampling: int,
    n_concat: int=2,
    n_components: int=None,
    backend='numpy',
):
    """
    RTW(Randomized Time Warping) implementation.
    
    Please see this paper for more details.
    Suryanto, C. H., Xue, J. H., & Fukui, K. (2016).
    Randomized time warping for motion recognition.
    Image and Vision Computing, 54, 1-11.

    Parameters:
    -----------
    inputs: list of array-like, shape (n_frames, n_dimensions)
    n_sampling: int
    n_concat: int
    n_components: 

    Returns:
    --------
    v: array-like, shape (n_dimensions*n_concat, n_components)
        right singular matrix of RTW array
    s: array-like, shape (n_components)
        sigular values of RTW array
    
    Example:
    --------
    x = np.random.rand(100, 100)
    v, s = randomized_time_warping(x, n_sampling=10, n_components=5)
    """
    
    assert n_components <= n_sampling
    assert len(inputs.shape) == 2
    
    n_frames = len(inputs)
    
    if backend == 'numpy':
        random_idx = np.random.randint(low=0, high=n_frames, size=(n_sampling, n_concat))
        sorted_random_idx = np.sort(random_idx, axis=1)
        inputs_sorted = inputs[sorted_random_idx].reshape(n_sampling, -1)
        u, s, v = svd(inputs_sorted, lapack_driver='gesdd')
    elif backend == 'torch':
        random_idx = torch.randint(low=0, high=n_frames, size=(n_sampling, n_concat))
        sorted_random_idx, _ = torch.sort(random_idx, dim=1)
        inputs_sorted = inputs[sorted_random_idx].view(n_sampling, -1)
        u, s, v = torch.svd(inputs_sorted)
    else:
        raise NotImplementedError

    if n_components is None:
        topk_components = v
        topk_sv = s
    else:
        topk_components = v[:, :n_components]
        topk_sv = s[:n_components]
    
    return topk_components, topk_sv
