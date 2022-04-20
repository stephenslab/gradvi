"""
Performs one coordinate descent step from the current parameters
for multiple linear regression with ash prior.
In the process, it calculates the ELBO.

Adapted from mr.ash R package of Yongseok Kim.
"""

import numpy as np

def updatebetaj(xj, betaj, dj, r, w, sigma2, sk2, s2invj, p, epstol):
    bjdj   = np.dot(r, xj) + betaj * dj
    muj    = bjdj * s2invj
    muj[0] = 0
    phij   = np.log(w + epstol) - np.log(1 + sk2 * dj) / 2 + muj * (bjdj / 2 / sigma2)
    phij   = np.exp(phij - np.max(phij))
    phij   = phij / np.sum(phij)
    betaj  = np.dot(phij, muj)
    a1     = bjdj * betaj
    a2     = np.dot(phij, np.log(phij + epstol)) - np.dot(phij[1:], np.log(s2invj[1:])) / 2
    return betaj, a1, a2, phij


def elbo(X, y, b, sigma2, sk, w, dj = None, epstol = 1e-12, s2inv = None):
    n, p = X.shape
    k    = sk.shape[0]
    y    = y - np.mean(y) ## hack to match ELBO from mr.ash.alpha
    r    = y - np.dot(X, b)
    sk2  = np.square(sk)
    if dj is None: 
        dj = np.sum(np.square(X), axis = 0)
    if s2inv is None:
        s2inv = np.zeros((p, k))
        s2inv[:, 1:] = 1 / (dj.reshape(p, 1) + 1 / sk2[1:].reshape(1, k - 1))

    a1 = 0
    a2 = 0
    phijk = np.zeros((p, k))
    bnew = b.copy()
    rnew = r.copy()
    wnew = np.zeros(k)
    for i in range(p):
        betaj, a1j, a2j, phij = updatebetaj(X[:, i], bnew[i],
                                            dj[i], rnew, w, sigma2, sk2, s2inv[i, :], p, epstol)
        bnew[i]     = betaj
        rnew        = rnew + X[:, i] * b[i] - X[:, i] * bnew[i]
        phijk[i, :] = phij
        wnew        += phij / p
        a1 += a1j
        a2 += a2j
    
    t1 = np.dot(rnew, rnew) - np.dot(np.square(bnew), dj) + a1
    t1 = t1 / sigma2 / 2.0
    t2 = np.log(2.0 * np.pi * sigma2) * n / 2.0
    t3 = - np.dot(wnew, np.log(w + epstol)) * p + a2 \
         + np.sum(wnew[1:] * np.log(sk2[1:]) * p / 2)
    #varobj = t1 / sigma2 / 2.0 \
    #         + np.log(2.0 * np.pi * sigma2) * n / 2.0 \
    #         - np.dot(wnew, np.log(w + epstol)) * p + a2 \
    #         + np.sum(wnew[1:] * np.log(sk2[1:]) * p / 2)
    varobj = t1 + t2 + t3
    return varobj
