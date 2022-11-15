import numpy as np
import scipy.ndimage as ndimg

def moving_average(y, n = None, method = 'uniform_filter'):

    if y.shape[0] < 5:
        raise ValueError("Too few data points for moving average")

    if n is None:
        n = int(y.shape[0] / 20)
        n = max(4, n)
        if (n % 2 != 0): n += 1

    # Reflect
    k = n // 2
    y_start = y[ : k ][::-1]
    y_end   = y[ -k + 1 : ]
    yref    = np.concatenate([y_start, y, y_end])

    if method == 'uniform_filter':
        x = ma_uniform_filter1d(yref, n)
    elif method == 'cumsum':
        x = ma_cumsum(yref, n)
    elif method == 'convolve':
        x = ma_convolve(yref, n)
    else:
        raise ValueError(f"Method {method} not defined for moving average")
    return x

def ma_convolve(y, n):
    x = np.convolve(y, np.ones(n) / float(n), 'valid')
    return x

def ma_cumsum(y, n):
    cm = np.cumsum(np.insert(y, 0, 0))
    x  = (cm[n:] - cm[:-n]) / float(n)
    return x

def ma_uniform_filter1d(y, n):
    x = ndimg.uniform_filter1d(y, n, mode='constant', origin=-(n//2))[:-(n-1)]
    return x
