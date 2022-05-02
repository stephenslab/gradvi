
import numpy as np
import copy

from . import Prior
from ..normal_means.nm_point_normal import NMPointNormal

class PointNormal(Prior):

    @property
    def prior_type(self):
        return 'point_normal'

    def __init__(self, sparsity = None, s2 = None):
        w = self.initialize(sparsity, s2)
        # Update self.w and self.wmod
        self.update_w(w)
        # Keep the initial values in memory
        self.w_init = self.w.copy()
        self.wmod_init = self.wmod.copy()
        # ================================
        # Normal Means model depends on the choice of prior.
        # ================================
        self.normal_means = NMPointNormal
        return


    def copy(self):
        newcopy = PointNormal(
            sparsity = 1 - copy.deepcopy(self.w[0]),
            s2 = copy.deepcopy(self.w[1])
            )
        return newcopy


    def update_wmod(self, wnew):
        self.wmod = wnew
        self.w = self.inverse_transform(wnew)
        return


    def update_w(self, wnew):
        self.w = wnew
        self.wmod = self.transform(wnew)
        return


    def wmod_jac(self):
        ajac = np.log(self.smbase) * self.w.reshape(-1, 1) * (np.eye(self.k) - self.w)
        return ajac


    #def wmod_grad(self, wgrad):
    #    agrad = np.sum(wgrad * self.wmod_jac(), axis = 1)
    #    return agrad


    def wmod_grad(self, wgrad):
        dfdpi = wgrad[0]
        dfds2 = wgrad[1]
        dfda0 = self.w[0] * (1 - self.w[0]) * dfdpi
        dfda1 = self.w[1] * dfds2
        return np.array([dfda0, dfda1])


    def transform(self, w, eps = np.finfo(float).eps):
        pi1 = w[0]
        s2  = w[1]
        if (pi1 > 0) and (pi1 < 1):
            a0 = np.log(pi1) - np.log(1 - pi1)
        elif pi1 == 0:
            a0 = np.log(eps)
        elif pi1 == 1:
            a0 = - np.log(eps)
        else:
            raise ValueError("Error in point normal prior. Weight must be within (0, 1)")
        a1 = np.log(s2)
        return np.array([a0, a1])


    def inverse_transform(self, wmod, eps = np.finfo(float).eps):
        a0  = wmod[0]
        a1  = wmod[1]
        pi1 = 1 / (1 + np.exp(-a0))
        s2  = np.exp(a1)
        return np.array([pi1, s2])


    def initialize(self, sparsity, s2):
        pi0 = 0.5 if sparsity is None else sparsity
        s2  = 1.0 if s2 is None else s2
        return np.array([1 - pi0, s2])


    def sample(self, size, scale = None, seed = None):
        if seed is not None:
            np.random.seed(seed)
        pi1 = self.w[0]
        s2  = self.w[1]
        gcomp = np.random.binomial(1, pi1, size = size)
        x     = np.zeros(size)
        ione  = np.where(gcomp == 1)[0]
        x[ione] = np.random.normal(0, np.sqrt(s2), size = ione.shape[0])
        return x
