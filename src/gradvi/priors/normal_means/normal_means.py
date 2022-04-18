"""
Unified interfaces to Normal Means models corresponding to different priors
"""

class NormalMeans:

    subclasses = {}

    @classmethod
    def register_nm(cls, prior_type):
        def decorator(subclass):
            cls.subclasses[prior_type] = subclass
            return subclass
        return decorator

    @classmethod
    def create(cls, prior_type, params):
        if prior_type not in cls.subclasses:
            raise ValueError('Bad prior type {}'.format(message_type))
        return cls.subclasses[message_type](params)

    def __init__(self, y, prior, s2):
        self.y = y
        self.yvar = s2


    def shrinkage_operator(self, jac = True):
        """
        Calculate the posterior expectation of b under NM model
        using Tweedie's formula.

        Returns shrinkage operator M(b)
        Dimensions:
            M: vector of size P
            M_bgrad: vector of size P
            M_wgrad: matrix of size P x K
            M_sgrad: vector of size P
        """
        M       = self.y + self.yvar * self.logML_deriv
        if jac:
            M_bgrad  = 1      + self.yvar * self.logML_deriv2
            M_wgrad  = self.yvar.reshape(-1, 1) * self.logML_deriv_wderiv
            M_s2grad = self.logML_deriv + self.yvar * self.logML_deriv_s2deriv
            return M, M_bgrad, M_wgrad, M_s2grad
        return M


    def penalty_operator(self, jac = True):
        """
        Returns the penalty operator, defined as sum_j lambda_j = rho(M(b)) / sj^2
        Dimensions:
            lambdaj: vector of size P
            l_bgrad: vector of size P
            l_wgrad: vector of size K
            l_sgrad: vector of size P 
        Note: lambdaj to be summed outside this function for sum_j lambda_j
              l_sgrad to be summed outside this function for sum_j d/ds2 lambda_j
        """
        lambdaj = - self.logML - 0.5 * self.yvar * np.square(self.logML_deriv)
        if jac:
            # Gradient with respect to b
            l_bgrad = - self.logML_deriv  - self.yvar * self.logML_deriv * self.logML_deriv2

            # Gradient with repect to w
            v2_ld_ldwd = nm.yvar.reshape(-1, 1) * nm.logML_deriv.reshape(-1, 1) * nm.logML_deriv_wderiv
            l_wgrad = - nm.logML_wderiv - v2_ld_ldwd
            l_wgrad = np.sum(l_wgrad, axis = 0)

            # Gradient with respect to reparametrized w 
            # depending upon the choice of prior
            l_agrad = prior.jacobian(l_wgrad)

            # Gradient with respect to s2
            v2_ld_lds2d = nm.yvar * nm.logML_deriv * nm.logML_deriv_s2deriv
            l_s2grad = - (nm.logML_s2deriv + 0.5 * np.square(nm.logML_deriv) + v2_ld_lds2d 
            #if self._is_prior_scaled:
            #    l_sgrad = - (nm.logML_s2deriv + v2_ld_lds2d + 0.5 * np.square(nm.logML_deriv) / self._dj)
            #else:
            #    #l_sgrad = - (nm.logML_s2deriv - 0.5 * np.square(nm.logML_deriv) - v2_ld_lds2d) / self._dj
            #    l_sgrad = - (nm.logML_s2deriv + 0.5 * np.square(nm.logML_deriv) + v2_ld_lds2d) / self._dj
            return lambdaj, l_bgrad, l_wgrad, l_s2grad
        return lambdaj
