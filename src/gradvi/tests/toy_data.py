"""
Toy data for testing
"""

import numpy as np
import collections
import patsy
from gradvi.models import basis_matrix as gvbm

ChangepointData = collections.namedtuple('CData', ['H', 'Hinv', 'x', 'y', 'ytest', 'ytrue', 'btrue', 'bspline_bases', 'bspline_beta', 'snr'])

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


def get_normal_means(prior, n = 100, s2 = 1., dj = 1., seed = 100):
    np.random.seed(seed)
    b   = prior.sample(n, seed = seed * 2, scale = s2)
    sj2 = s2 / dj
    z   = sample_normal_means(b, sj2)
    return z, sj2


def sample_coefs (p, method="normal", bfix=None):
    '''
    Sample coefficientss from a distribution (method = normal / gamma)
    or use a specified value for all betas:
        bfix = const -> all betas will have beta = const
        bfix = [a, b, c, ...] -> all betas can be specified using an array
    Note: 
        when sampling from the gamma distribution,
        a random sign (+/-) will be assigned
    '''
    beta = np.zeros(p)

    # helper function to obtain random sign (+1, -1) with equal proportion (f = 0.5)
    def sample_sign(n, f = 0.5):
        return np.random.choice([-1, 1], size=n, p=[f, 1 - f])

    # sample beta from Gaussian(mean = 0, sd = 1)
    if method == "normal":
        beta = np.random.normal(size = p)

    # receive fixed beta input
    elif method == "fixed":
        assert bfix is not None, "bfix is not specified for fixed signal"
        if isinstance(bfix, (collections.abc.Sequence, np.ndarray)):
            assert len(bfix) == p, "Length of input coefficient sequence is different from the number of non-zero coefficients"
            beta = bfix
        else:
            beta = np.repeat(bfix, p)

    # sample beta from a Gamma(40, 0.1) distribution and assign random sign
    elif method == "gamma":
        params = [40, 0.1]
        beta = np.random.gamma(params[0], params[1], size = p)
        beta = np.multiply(beta, sample_sign(p))

    return beta


def changepoint_from_bspline (x, knots, std,
                 degree = 0, signal = "gamma", seed = None,
                 include_intercept = False, bfix = None,
                 eps = 1e-8, get_bsplines = False):
    '''
    Generate trend-filtering data.

    Parameters
    ----------
    x: ndarray
        Data points

    knots: ndarray
        Location of changepoints on x

    std: float
        Standard deviation of the noise

    degree: integer, default 0
        Degree of the B-spline basis

    signal: str, default 'gamma'
        A distribution from which the coefficients are sampled
            - 'gamma' : Gamma(40, 0.1) with random signs (-1, 1)
            - 'normal' : Normal(0, 1)
            - 'fixed' : provide fixed values of the coefficients,
                        must be used with `bfix`.

    seed: integer
        Set a seed for reproducibility

    include_intercept: bool, default False
        Whether to include a basis function for intercept

    bfix: ndarray
        specify the values of the coefficients for the basis functions

    Note
    ----
    number of bases = k + degree + 1 if include_intercept = True
    number of bases = k + degree     if include_intercept = False

    '''
    if seed is not None: np.random.seed(seed)
    # ------------------------------
    n = x.shape[0]
    # ------------------------------
    # Generate B-spline bases given the knots and degree
    bspline_bases = patsy.bs(x, knots = knots, degree = degree, include_intercept = include_intercept)
    nbases = knots.shape[0] + degree + int(include_intercept)
    assert bspline_bases.shape[1] == nbases, "Number of B-spline bases does not match the number of knots + degree + interecept"
    # ------------------------------
    # Generate coefficients for the bases
    beta   = sample_coefs(nbases, method = signal, bfix = bfix)
    # ------------------------------
    # Generate the function without noise
    ytrue = np.dot(bspline_bases, beta)
    # ------------------------------
    # Map the data to trendfiltering bases
    # set low values of beta to zero and regenerate y
    H     = gvbm.trendfiltering(n, degree)
    Hinv  = gvbm.trendfiltering_inverse(n, degree)
    Hs, fscale, floc = gvbm.center_and_scale_tfbasis(H)
    Hinvs = Hinv * fscale.reshape(-1,1)
    Hinvs[0, :] = 1 / n
    #H     = gvbm.trendfiltering_scaled(n, degree)
    #Hinv  = gvbm.trendfiltering_inverse_scaled(n, degree)
    btrue = np.dot(Hinv, ytrue)
    btrue[np.abs(btrue) <= eps] = 0.
    noise = np.random.normal(0, std, size = n * 2)
    ytrue = np.dot(H, btrue)
    y     = ytrue + noise[:n]
    # ------------------------------
    # Some test data?
    ytest = ytrue + noise[n:]
    # ------------------------------
    # Signal to noise ratio
    # (experimental)
    signal = np.mean(np.square(btrue[btrue != 0]))
    snr    = signal / np.square(std)
    # ------------------------------
    data  = ChangepointData(H = H, Hinv = Hinv, x = x, y = y, ytest = ytest, ytrue = ytrue, btrue = btrue,
                            bspline_bases = bspline_bases, bspline_beta = beta, snr = snr)
    return data
