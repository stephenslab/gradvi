import numpy as np
import collections


def truncated_power(n, k): 
    ''' 
    Truncated power basis matrix for degree k, order k + 1 with m data points
    adapted from [Tibshirani, 2014](https://doi.org/10.1214/13-AOS1189)
    Equation (22) [page 303]
    '''
    # Note: Python is zero indexed.
    X = np.zeros((n, n)) 
    if k == 0:
        for j in range(n):
            X[j:n, j] = 1 
    else:
        npowerk = np.power(n, k)
        '''
        j = 1, ..., k+1
        '''
        # create a nx1 matrix with 1, ..., n
        seq = np.arange(1, n+1).reshape(n, 1)
        # repeat (tile) the matrix k+1 times.
        # np.tile(seq, k+1) is a n x k+1 matrix whose each column contains 1..n
        # raise each column to the power of 0, ..., k 
        X[:, :k + 1] = np.power(np.tile(seq, k+1), np.arange(k+1)) / np.power(n, np.arange(k+1))
        '''
        j > k + 1
        '''
        for j in range(k+1, n): 
            khalf = int(k / 2) if k % 2 == 0 else int((k + 1) / 2)
            # non-zero value if row i > j - khalf, that is from i = j + 1 - khalf
            # for column j, one-base indices of those rows are np.arange(j + 1 - khalf, n) + 1
            X[(j - khalf + 1):, j] = np.power(np.arange(j - khalf + 1, n) - j + khalf, k) / npowerk
    return X
