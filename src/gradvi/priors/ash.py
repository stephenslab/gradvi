class Ash(Prior):

    def __init__(self, sk, wk = None, smbase = None, sparsity = None): 
        self.smbase = smbase
        self.sk = sk
        if wk is None: 
            self.w = self.initialize(sk.shape[0], sparsity)
        else:
            self.w = wk.copy()
        self.wmod = self.softmax_inverse(w)
        self.w_init = self.wmod.copy() 
        return


    @property
    def sk(self):
        return self.sk


    def update_wmod(self, wnew):
        self.wmod = wnew
        self.w = self.softmax(wnew)
        return


    def update_w(self, wnew):
        self.w = wnew
        self.wmod = self.softmax_inverse(w)
        return


    def softmax(self, a):
        if self.smbase is not None:
            beta = np.log(base)
            a = a * beta
        e_a = np.exp(a - np.max(a))
        w   = e_a / np.sum(e_a, axis = 0, keepdims = True)
        return w


    def softmax_inverse(self, w, eps = 1e-8)
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
