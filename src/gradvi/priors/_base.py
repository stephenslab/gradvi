class Prior:

    def __init__(self, w):
        """
        Each prior class must initiate the following
        """
        # Initial value of the parameters
        self.w_init = w
        # Parameters of the prior
        self.w = w
        # Reparametrized parameters of the prior
        self.wmod = w
        return


    def update_wmod(self, wnew):
        """
        Update the reparametrized values of the prior parameters
        (for example, during gradient descent iterations)
        """
        self.wmod = wnew
        self.w = wnew
        return


    def update_w(self, wnew):
        """
        Update the values of the prior parameters
        (for example, during gradient descent iterations)
        """
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
    def bounds(self):
        """
        Bounds for the parameters,
        to be used in L-BFGS-B minimization
        """
        bd = [(None, None) for x in self.w]
        return bd
