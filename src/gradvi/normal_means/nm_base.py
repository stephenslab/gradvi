"""
This is the parent class of all Normal Means models for all priors.
It provides the two main operators we need for GradVI:
    - Shrinkage Operator :math:`M(x)`
    - Penalty Operator :math:`\rho_j(x_j)`
These are constructed from the log marginal likelihood and its derivatives,
which need to be defined in each child class of Normal Means for each prior.

TO-DO: Provide autodiff options for the derivatives.
"""

import numpy as np

class NMBase:


    def shrinkage_operator(self, jac = True, hess = False):
        """
        Calculate the posterior expectation of b under NM model
        using Tweedie's formula.

        Parameters
        ----------
        jac : boolean, default True
            Whether to return the derivatives

        Returns
        -------
        M : ndarray of shape (n_features,)
            M(y) - the shrinkage operator applied on the input self.y

        M_bgrad : ndarray of shape (n_features,)
            The derivative of M(y) with respect to y

        M_wgrad : ndarray of shape (n_features, K)
            The derivative of M(y) with respect to w, where w are the
            parameters of the prior to be estimated. Note, there are
            K unknown parameters for the prior.

        M_s2grad : ndarray of shape (n_features,)
            The derivative of M(y) with respect to sj2, where sj2 is
            the variance of the Normal Means model. 
        """
        M  = self.y + self.yvar * self.logML_deriv
        if jac or hess:
            M_bgrad  = 1 + self.yvar * self.logML_deriv2
            M_wgrad  = self.yvar.reshape(-1, 1) * self.logML_deriv_wderiv
            M_s2grad = self.logML_deriv + self.yvar * self.logML_deriv_s2deriv

            if hess:
                M_bgrad2 = self.yvar * self.logML_deriv3
                return M, M_bgrad, M_wgrad, M_s2grad, M_bgrad2

            return M, M_bgrad, M_wgrad, M_s2grad
        return M


    def penalty_operator(self, jac = True, hess = False):
        """
        Calculate the penalty operator, defined as 
            sum_j L_j,   where
            L_j = rho(M(y_j)) / s_j^2

        Parameters
        ----------
        jac : boolean, default True
            Whether to return the derivatives

        Returns
        -------
        lambdaj : ndarray of shape (n_features,)
            The penalty operator applied on the input self.y

        l_bgrad : ndarray of shape (n_features,)
            The derivative of L with respect to y

        l_wgrad : ndarray of shape (n_features, K)
            The derivative of L with respect to w, where w are the
            parameters of the prior to be estimated. Note, there are
            K unknown parameters for the prior.

        l_s2grad : ndarray of shape (n_features,)
            The derivative of L with respect to sj2, where sj2 is
            the variance of the Normal Means model. 
        """
        lambdaj = - self.logML - 0.5 * self.yvar * np.square(self.logML_deriv)
        if jac or hess:
            # Gradient with respect to b
            l_bgrad = - self.logML_deriv * ( 1 + self.yvar * self.logML_deriv2 )

            # Gradient with repect to w
            v2_ld_ldwd = self.yvar.reshape(-1, 1) * self.logML_deriv.reshape(-1, 1) * self.logML_deriv_wderiv
            l_wgrad = - self.logML_wderiv - v2_ld_ldwd
            #l_wgrad = np.sum(l_wgrad, axis = 0)

            # Gradient with respect to s2
            v2_ld_lds2d = self.yvar * self.logML_deriv * self.logML_deriv_s2deriv
            l_s2grad = - self.logML_s2deriv - 0.5 * np.square(self.logML_deriv) - v2_ld_lds2d

            if hess:
                l_b2grad = - self.logML_deriv2 * (1 + self.yvar * self.logML_deriv2) - self.yvar * self.logML_deriv * self.logML_deriv3

                return lambdaj, l_bgrad, l_wgrad, l_s2grad, l_b2grad

            return lambdaj, l_bgrad, l_wgrad, l_s2grad

        return lambdaj
