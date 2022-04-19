
import numpy as np

from . import Prior
from .normal_means import NMAsh, NMAshScaled

class Ash(Prior):

    @property
    def prior_type(self):
        if self.is_scaled:
            return 'ash_scaled'
        else:
            return 'ash'

    def __init__(self, sk, wk = None, smbase = np.exp(1), sparsity = None, scaled = True):
        self.smbase = smbase
        self.sk = sk
        if wk is None: 
            wk = self.initialize(sk.shape[0], sparsity)
        self.update_w(wk)
        self.w_init = self.wmod.copy()
        self.is_scaled = scaled
        # ================================
        # Normal Means model depends on the choice of prior.
        # ================================
        if scaled:
            self.normal_means = NMAshScaled
        else:
            self.normal_means = NMAsh
        return


    def update_wmod(self, wnew):
        self.wmod = wnew
        self.w = self.softmax(wnew)
        return


    def update_w(self, wnew):
        self.w = wnew
        self.wmod = self.softmax_inverse(wnew)
        return


    def wmod_jac(self):
        ajac = np.log(self.smbase) * self.w.reshape(-1, 1) * (np.eye(self.k) - self.w)
        return ajac


    def wmod_grad(self, wgrad):
        agrad = np.sum(wgrad * self.wmod_jac(), axis = 1)
        return agrad


    def softmax(self, a):
        if self.smbase is not None:
            beta = np.log(self.smbase)
            a = a * beta
        e_a = np.exp(a - np.max(a))
        w   = e_a / np.sum(e_a, axis = 0, keepdims = True)
        return w


    def softmax_inverse(self, w, eps = 1e-8):
        a = np.log(w + eps) / np.log(self.smbase)
        return a


    def initialize(self, k, sparsity):
        w = np.zeros(k)
        if sparsity is None:
            w[0] = 1. / k
        else:
            w[0] = sparsity
        w[1:(k-1)] = np.repeat((1 - w[0])/(k-1), (k - 2))
        w[k-1] = 1 - np.sum(w)
        return w
