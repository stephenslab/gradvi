"""
Unified interfaces to Normal Means models.
This is a factory method which creates the required Normal Means model
based on the prior, and returns the two main operators we need for GradVI:
    - Shrinkage Operator :math:`M(x)`
    - Penalty Operator :math:`\rho_j(x_j)`

References:
https://medium.com/@vadimpushtaev/python-choosing-subclass-cf5b1b67c696
https://stackoverflow.com/questions/27322964

"""

import numpy as np

class NormalMeans:

    @classmethod
    def create(self, y, prior, sj2, **kwargs):
        return prior.normal_means(y, prior, sj2, **kwargs)


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
        M  = self.y + self.yvar * self.logML_deriv
        if jac:
            M_bgrad  = 1 + self.yvar * self.logML_deriv2
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
            l_wgrad: vector of size P x K
            l_sgrad: vector of size P 
        Note: lambdaj to be summed outside this function for sum_j lambda_j
              l_sgrad to be summed outside this function for sum_j d/ds2 lambda_j
        """
        lambdaj = - self.logML - 0.5 * self.yvar * np.square(self.logML_deriv)
        if jac:
            # Gradient with respect to b
            l_bgrad = - self.logML_deriv  - self.yvar * self.logML_deriv * self.logML_deriv2

            # Gradient with repect to w
            v2_ld_ldwd = self.yvar.reshape(-1, 1) * self.logML_deriv.reshape(-1, 1) * self.logML_deriv_wderiv
            l_wgrad = - self.logML_wderiv - v2_ld_ldwd
            #l_wgrad = np.sum(l_wgrad, axis = 0)

            # Gradient with respect to s2
            v2_ld_lds2d = self.yvar * self.logML_deriv * self.logML_deriv_s2deriv
            l_s2grad = - self.logML_s2deriv - 0.5 * np.square(self.logML_deriv) - v2_ld_lds2d 

            return lambdaj, l_bgrad, l_wgrad, l_s2grad

        return lambdaj
