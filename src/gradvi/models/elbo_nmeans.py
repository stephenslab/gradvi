"""
Use the Normal Means model to calculate ELBO
"""

import numpy as np
from ..normal_means import NormalMeans

def ash(X, y, b, sigma2, prior, dj = None, phijk = None, mujk = None, varjk = None, eps = 1e-8):
    n, p    = X.shape
    k       = prior.k
    w       = prior.w
    y       = y - np.mean(y) ## hack to match ELBO from varbvsmix
    sk2     = np.square(prior.sk.reshape(1, k))

    if dj is None: 
        dj = np.sum(np.square(X), axis = 0)

    r       = y - np.dot(X, b)
    btilde  = b + np.dot(X.T, r) / dj
    
    if any([x is None for x in [phijk, mujk, varjk]]):
        nmash = NormalMeans.create(btilde, prior, sigma2 / dj, scale = sigma2, d = dj)
        phijk, mujk, varjk = nmash.posterior()
    
    mujk2   = np.square(mujk)
    bmean   = np.sum(phijk * mujk, axis = 1)
    r       = y - np.dot(X, bmean)
    rTr     = np.dot(r.T, r)
    bvar    = np.sum(phijk * (mujk2 + varjk), axis = 1) - np.square(bmean)
    elbo1   = - 0.5 * n * np.log(2.0 * np.pi * sigma2) \
              - 0.5 * (rTr +  np.dot(dj, bvar)) / sigma2 \
              - 0.5 * np.log(n)
    elbo2   = - np.sum(phijk * np.log(phijk + eps)) + np.sum(phijk * np.log(w + eps))
    
    t5inner = 1 + np.log(varjk[:, 1:]) - np.log(sigma2) - np.log(sk2[:, 1:]) \
                - (mujk2[:, 1:] + varjk[:, 1:]) / sigma2 / sk2[:, 1:]
    elbo3   = np.sum(phijk[:, 1:] * t5inner) / 2
    elbo    = elbo1 + elbo2 + elbo3
    return  -elbo
