import numpy as np

def haar(n):
    if n == 1: 
        return np.atleast_2d(1)
    H = np.concatenate( (np.kron(haar(n//2),        [1,1]), 
                         np.kron(np.identity(n//2), [1,-1])), axis = 0) / np.sqrt(2)
    return H

def haar_inverse(n):
    return haar(n).transpose()
