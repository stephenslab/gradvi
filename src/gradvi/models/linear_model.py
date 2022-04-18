class LinearModel:
    """
    LinearModel calculates the objective function 
    and gradients of a linear model (normal distribution).
        y ~ Normal(Xb, s2)
        b ~ g(w)
    given the data {X, y, b} and the prior distribution family g(w),
    where w are the parameters of the prior.
    """

    def __init__(self, X, y, b, s2, prior, dj = None, objtype = "reparametrize", debug = False):
        self._X = X
        self._y = y
        self._b = b
        self._s2 = s2
        self._prior = prior
        self._objtype = objtype
        self._dj = self.set_xvar(dj)

        # set debug options
        self._is_debug = debug
        logging_level  = logging.DEBUG if debug else logging.INFO
        self.logger    = MyLogger(__name__, level = logging_level)


    def set_xvar(self, dj):
        if dj is None:
            dj = np.sum(np.square(self._X), axis = 0)
        return dj


    @run_once
    def calc_obj_jac(self):
        """
        Calculates the objective function and gradients for the linear model
        """
        self.logger.debug(f"Calculating PLR objective with sigma2 = {self._s2}")

        # Initialize the Normal Means model
        norm_mean = NormalMeans(self._b, self._prior, self._s2 / self._dj, s2 = self._s2, d = self._dj)

        # shrinkage operator and penalty operator
        # M(b) and rho(b)
        Mb, Mb_bgrad, Mb_wgrad, Mb_sj2grad = norm_mean.shrinkage_operator()
        lj, l_bgrad,  l_wgrad,  l_sj2grad  = norm_mean.penalty_operator()

        # gradients with respect to s2

        # Objective function
        r = self._y - np.dot(self._X, Mb)
        rTr  = np.sum(np.square(r))
        rTX  = np.dot(r.T, self._X)
        obj  = (0.5 * rTr / self._s2) + np.sum(lj)
        obj += 0.5 * (self._n - self._p) * np.log(2 * np.pi * self._s2)

        # Gradients
        bgrad  = - (rTX * Mb_bgrad / self._s2) + l_bgrad
        wgrad  = - np.dot(rTX, Mb_wgrad) / self._s2  + l_wgrad
        s2grad = - 0.5 * rTr / (self._s2 * self._s2) \
                 - np.dot(rTX, Mb_s2grad) / self._s2 \
                 + np.sum(l_s2grad) \
                 + 0.5 * (self._n - self._p) / self._s2

        self._objective = obj
        self._bgrad     = bgrad
        self._wgrad     = wgrad
        self._s2grad    = s2grad
        return


    # =====================================================
    # Attributes
    # =====================================================
    @property
    def objective(self):
        return self._objective


    @property
    def gradients(self):
        return self._bgrad, self._wgrad, self._s2grad


    @property
    def coef(self):
        return self._coef


    @property
    def coef_inv(self):
        return self._theta
