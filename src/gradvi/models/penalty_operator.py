
"""
This provides the standalone penalty operator rho(b) 
given any vector b, some prior g(w) and the variance of NM model.

Note the difference with NormalMeans.penalty_operator() 
which calculates the penalty function rho(M(b)),
where is the Normal Means Shrinkage Operator (also called
Posterior Mean Operator).
"""


def penalty_func(b, ak, std, sk, dj,
                 jac = True,
                 softmax_base = np.exp(1),
                 method = 'fssi-cubic',
                 ngrid = 1000
                ):
    wk = softmax(ak, base = softmax_base)
    k = ak.shape[0]
    p = b.shape[0]
    Minv = shrinkage_operator_inverse(b, std, wk, sk, dj, method = method, ngrid = ngrid)
    t = Minv.x
    nm = NormalMeansASHScaled(t, std, wk, sk, d = dj)
    s2 = nm.yvar
    #rhoMtj = - nm.logML #- 0.5 * nm.yvar * np.square(nm.logML_deriv)
    rhoMtj = - nm.logML - 0.5 * np.square(t - b) / nm.yvar
    rhoMt = np.sum(rhoMtj)
    if jac:
        dHdb = (t - b) / nm.yvar
        # dMdw = nm.yvar.reshape(-1, 1) * nm.logML_deriv_wderiv
        # dMdtinv = 1 / (1 + nm.yvar * nm.logML_deriv2)
        # dtdw = - nm.logML_deriv_wderiv * (nm.yvar * dMdtinv).reshape(-1, 1)
        ## Derivative of -0.5(t-b)^2 / s2
        # dHdw = - ((t - b) / nm.yvar).reshape(-1, 1) * dtdw
        ## Derivative of -logML(t)
        # dHdw = - nm.logML_wderiv - (nm.logML_deriv).reshape(-1, 1) * dtdw
        dHdw = - nm.logML_wderiv
        dHdw = np.sum(dHdw, axis = 0)
        akjac = np.log(softmax_base) * wk.reshape(-1, 1) * (np.eye(k) - wk)
        dHda = np.sum(dHdw * akjac, axis = 1)
        return rhoMt, dHdb, dHda
    else:
        return rhoMt
