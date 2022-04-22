
import numpy as np

from . import Prior
from ..normal_means import NMAsh, NMAshScaled

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


    def sample(self, size, scale = None, seed = None):
        if seed is not None:
            np.random.seed(seed)
        runif = np.random.uniform(0, 1, size = size)

        # Get index of mixture component for all elements.
        # gcomp[j] is the mixture component from which
        # the j-th element should be sampled.
        #
        # Example
        # >>> x = np.array([0.2, 6.4, 3.0, 1.6])
        # >>> bins = np.array([1.0, 2.5, 4.0, 10.0])
        # >>> inds = np.digitize(x, bins)
        # >>> inds
        # array([0, 3, 2, 1])
        # >>> wk = np.array([0.8, 0.1, 0.1])
        # >>> np.cumsum(wk)
        # array([0.8, 0.9, 1. ])
        # >>> runif = np.random.uniform(0, 1, size = 100)
        # >>> np.digitize(runif, np.cumsum(wk))
        # array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0,
        #        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2,
        #        0, 0, 1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 2, 0, 1, 0, 0, 1, 1, 0, 0, 0,
        #        2, 0, 2, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0,
        #        2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        gcomp = np.digitize(runif, np.cumsum(self.w))

        # Scaled and unscaled prior
        sk = self.sk.copy()
        if self.is_scaled and scale is not None:
            sk *= np.sqrt(scale)

        # Sample each element from its corresponding 
        # mixture components
        x = np.zeros(size)
        for i, gc in enumerate(gcomp):
            if sk[gc] > 0:
                x[i] = np.random.normal(0, sk[gc])
        return x
