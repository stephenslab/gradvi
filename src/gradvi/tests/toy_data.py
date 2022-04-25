"""
Toy data for testing
"""

import numpy as np

def center_and_scale(Z):
    dim = Z.ndim
    if dim == 1:
        Znew = Z / np.std(Z)
        Znew = Znew - np.mean(Znew)
    elif dim == 2:
        Znew = Z / np.std(Z, axis = 0)
        Znew = Znew - np.mean(Znew, axis = 0).reshape(1, -1)
    return Znew


def get_linear_model(n = 100, p = 200, p_causal = 50, pve = 0.5, rho = 0.4, standardize = True):

    # This is specific to linear model
    def sd2_from_pve (X, b, pve):
        return np.var(np.dot(X, b)) * (1 - pve) / pve

    np.random.seed(100)

    """
    Equicorr predictors
    X is sampled from a multivariate normal, with covariance matrix V.
    V has unit diagonal entries and constant off-diagonal entries rho.
    """
    iidX    = np.random.normal(size = n * p).reshape(n, p)
    comR    = np.random.normal(size = n).reshape(n, 1)
    X       = comR * np.sqrt(rho) + iidX * np.sqrt(1 - rho)
    if standardize:
        X   = center_and_scale(X)
    bidx    = np.random.choice(p, p_causal, replace = False)
    b       = np.zeros(p)
    b[bidx] = np.random.normal(size = p_causal)
    s2      = sd2_from_pve(X, b, pve)
    y       = np.dot(X, b) + np.sqrt(s2) * np.random.normal(size = n)
    return X, y, b, s2


def sample_normal_means(mean, var):
    p = mean.shape[0]
    if not isinstance(var, np.ndarray):
        var = np.repeat(var, p)
    y = np.random.multivariate_normal(mean, np.diag(var))
    return y


def get_normal_means(prior, n = 100, s2 = 1.44, dj = None):
    np.random.seed(100)
    #dj = np.square(np.random.normal(1, 0.5, size = n)) * n
    if dj is None:
        dj  = np.ones(n) * n
    b   = prior.sample(n, seed = 200, scale = s2)
    sj2 = s2 / dj
    z   = sample_normal_means(b, sj2)
    return z, sj2, s2, dj

