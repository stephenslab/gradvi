import numpy as np
import collections


def discrete_difference_operator_check(n, k):
    '''
    Returns discrete difference operator D(k)
    This is the exact definition used in [Tibshirani, 2014]
    and should only be used for checking the faster implementation below.
    '''
    if k == 0:
        D = np.eye(n)
    else:
        # define D(1)
        D = np.zeros((n-1, n))
        for i in range(n-1):
            D[i, i] = -1
            D[i, i+1] = 1
        D1 = D.copy()
        for j in range(1, k):
            Dj = D.copy()
            D = np.dot(D1[:n-j-1, :n-j], Dj)
    return D


def discrete_difference_operator(n, k, return_row = False):
    '''
    Returns discrete difference operator D(k)
    This is a fast implementation without any dot product.
    
    Parameters
    ----------
    n : int
        The number of observations

    k : int
        Order of the trendfiltering

    return_row : bool, optional 
        Whether to return only the non-zero values of first row 
        D[0, :k+2]. (The first row is required for calculating 
        the inverse trendfiltering basis matrix). Default is False.

    Returns
    -------
    D : array of shape (n - k, n)
        The discrete difference operator

    '''
    Drow = np.zeros((2, k + 2))
    for i in range(2):
        Drow[i, i] = 1
    for j in range(k):
        Drow[0, :]     = Drow[1,:] - Drow[0, :]
        Drow[1, 1:j+3] = Drow[0, :j+2]
    if not return_row:
        D = np.zeros((n - k, n))
        for irow in np.arange(n - k):
            D[irow, irow:irow + k + 1] = Drow[0, :k+1]
    else:
        D = np.zeros(n)
        D[:k+2] = Drow[0, :k+2]
    return D


def trendfiltering_inverse_check(n, k):
    '''
    Returns the inverse of the trendfiltering basis matrix H
    This is the exact definition used in [Tibshirani, 2014]
    and should only be used for checking the faster implementation below.
    See proof of Lemma 2 in Supplementary.
    '''
    Dk = discrete_difference_operator(n, k + 1)
    Minv = np.zeros((n, n))
    for i in range(k + 1):
        Drow = discrete_difference_operator(n, i, return_row = True)
        Minv[i, :] = Drow
    Minv[i+1:, :] = Dk
    #tconst = np.power(n, k) / np.math.factorial(k)
    return Minv


def trendfiltering_inverse(n, k):
    '''
    Returns the inverse of the trendfiltering basis matrix H', where

        bhat = argmin ||y - b||^2 + \lambda ||H'b||

    This is a faster implementation without any dot product.
    Check output with trendfiltering_inverse_check(n, k).
    See proof of Lemma 2 in Supplementary.
    '''
    Hinv = np.zeros((n, n))
    for i in range(2):
        Hinv[i, i] = 1
    for i in range(1, k + 2):
        Hinv[i, :i+2] = Hinv[i, :i+2] - Hinv[i-1, :i+2]
        Hinv[i+1, 1:i+3] = Hinv[i, :i+2]
    for j in range(1, n-k-2):
        irow = i + j + 1
        Hinv[irow, j+1:j+k+3] = Hinv[i, :k+2]
    return Hinv


def trendfiltering_check(n, k):
    '''
    Returns the forward trendfiltering basis matrix H, where

        bhat = argmin ||y - Hb||^2 + \lambda \sum |b_j|

    This is the exact definition used in [Tibshirani, 2014]
    and should only be used for checking the faster implementation below.
    See proof of Lemma 2 in Supplementary
    '''
    #tconst = np.power(n, k) / np.math.factorial(k)
    def getMi(n, i):
        M = np.zeros((n, n))
        M[:i, :i] = np.eye(i)
        M[i:, i:] = np.tril(np.ones((n-i, n-i)))
        return M
    M = getMi(n, 0)
    for i in range(1, k+1):
        M = np.dot(M, getMi(n, i))
    return M


def trendfiltering(n, k):
    '''
    Returns the trendfiltering basis matrix H
    This is a faster implementation without any dot product.
    Check output with trendfiltering_basis_matrix_check(n, k)
    '''
    H = np.zeros((n, n))
    A = list([np.ones(n) for i in range(k + 1)])
    for i in range(1, k + 1):
        A[i] = np.cumsum(A[i-1])
    for j in range(k):
        H[j:, j] = A[j][:n-j]
    for j in range(k, n):
        H[j:, j] = A[k][:n-j]
    return H


def trendfiltering_tibshirani(n, k):
    '''
    This is an alternate definition of the trendfiltering basis matrix H,
    see Eq 27 in [Tibshirani, 2014].
    I have not calculated the inverse of this matrix.
    '''
    # ----------------------------
    # Let's not delete the explicit version, slow
    # but this is what we are doing.
    # ----------------------------
    # H = np.zeros((n, n))
    # npowerk = np.power(n, k)
    # seq = np.arange(1, n+1).reshape(n, 1)
    # H[:, :k + 1] = np.power(np.tile(seq, k+1), np.arange(k+1)) / np.power(n, np.arange(k+1))
    # for j in range(k+1, n):
    #     for i in range(n):
    #         if i > j - 1:
    #             Hij = 1.0
    #             for l in range(1, k+1):
    #                 Hij *= (i - j + k - l + 1)
    #             H[i, j] = Hij #/ np.power(n, k)
    # ----------------------------
    # ----------------------------
    # A function for fast calculation of the lower triangular matrix
    # obtained from the third condition in Eq 27
    def lower_tril_from_vector(S):
        n = S.shape[0]
        X = np.zeros((n, n))
        X[:, 0] = S
        for j in range(1, n):
            X[j:, j] = S[:-j]
        return X
    # ----------------------------
    # instead of calculating each element
    # precalculate the vector of products in the third condition only once 
    # and fill up the lower triangular matrix of the basis
    npowerk = np.power(n, k)
    kfact = np.math.factorial(k)
    S = np.ones(n - k - 1)
    for i in range(1, n - k - 1):
        S[i] = S[i - 1] * (i + k) / i
    # ----------------------------
    H = np.zeros((n, n))
    seq = np.arange(1, n+1).reshape(n, 1)
    H[:, :k + 1] = np.power(seq, np.arange(k+1)) / np.power(n, np.arange(k+1))
    H[k+1:, k+1:] = lower_tril_from_vector(S * kfact / npowerk)
    return H
