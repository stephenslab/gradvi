class GradVIResult(dict):
    """ Represents the optimization result.
    Attributes
    ----------
    b_post : ndarray
        The posterior mean of the parameters in the problem.

    b_inv : ndarray
        The optimized value of theta in the problem,
        obtained by applying the inverse shrinkage operator on
        the posterior mean of b.
        `M(b_inv) = b_post`, where M is the shrinkage operator.

    residual_var : float
        The optimized value of the residual variance.

    prior : object
        The optimized value of the prior. See the Prior class for details.

    success : bool
        Whether or not the gradient descent solver exited successfully.

    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.

    message : str
        Description of the cause of the termination.

    fitobj : object
        The optimization object

    Notes
    -----
    `GradVIResult` may have additional attributes not listed here depending
    on the specific problem being solved. Since this class is essentially a
    subclass of dict with attribute accessors, one can see which
    attributes are available using the `GradVIResult.keys` method.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


class GradVI():

    """
    Base class for GradVI containing the attributes that are required
    to be returned by each task.
    """

    @property
    def theta(self):
        return self._res.b_inv


    @property
    def coef(self):
        return self._res.b_post


    @property
    def prior(self):
        return self._res.prior


    @property
    def success(self):
        return self._res.success


    @property
    def obj_path(self):
        return self._hpath


    @property
    def niter(self):
        return self._niter


    @property
    def nfev(self):
        return self._nfev_count


    @property
    def njev(self):
        return self._njev_count


    @property
    def fun(self):
        return self._res.fun


    @property
    def grad(self):
        return self._res.grad


    @property
    def fitobj(self):
        return self._res.fitobj
