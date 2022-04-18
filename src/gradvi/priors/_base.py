class Prior:

    def __init__(self, w):
        self.w_init = w
        self.w = w
        self.wmod = w
        return


    def update_wmod(self, wnew):
        self.wmod = wnew
        self.w = wnew
        return


    def update_w(self, wnew):
        self.w = wnew
        self.wmod = wnew
        return


    @property
    def k(self):
        """
        Number of parameters to be estimated
        """
        return self.w.shape[0]


    @property
    def w(self):
        """
        Current value of the parameters of the prior
        """
        return self.w


    @propery
    def wmod(self):
        """
        Current value of the reparametrized parameters of the prior
        """
        return self.wmod


    @property
    def w_init(self):
        """
        Initial value of the parameters
        """
        return self.w_init


    @property
    def bounds(self):
        """
        Bounds for the parameters,
        to be used in L-BFGS-B minimization
        """
        bd = [(None, None) for x in self.w]
        return bd
