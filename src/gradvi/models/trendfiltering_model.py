
import numpy as np

from . import LinearModel

class TrendfilteringModel(LinearModel):

    def __init__(
            self, X, y, b, s2, prior,
            dj = None,
            objtype = "reparametrize",
            v2inv = None,
            debug = False,
            invert_method = "trisection",
            invert_options = {},
            tf_degree = 0,
            tfbasis_matrix = None,
            tfbasis_scale_factors = (None, None),
            standardize_basis = False,
            scale_basis = False
            ):

        super().__init__(X, y, b, s2, prior, 
            dj = dj, objtype = objtype, v2inv = v2inv, debug = debug, 
            invert_method = invert_method, invert_options = invert_options)

        self._tf_standardize_basis = standardize_basis
        self._tf_scale_basis = scale_basis
        self._tf_degree = tf_degree
        self._tf_X    = tfbasis_matrix
        self._tf_fstd = tfbasis_scale_factors[0]
        self._tf_floc = tfbasis_scale_factors[1]
        return


    def Xdotv_unscaled(self, v):
        '''
        X.v = | X0  X1 | . | v0 | = | X0v0 + X1v1 |
              | X2  X3 |   | v1 |   | X2v0 + X3v1 |

        [ X0v0, X2v0]' = [X0, X2]' . v0

        X0v0, X2v0 -- calculate the dot product explicitly
        X1v1 -- zero
        X3v1 -- repetitive cumulative sum
        '''
        d    = self._tf_degree
        X3v1 = np.zeros_like(v)
        v0 = v[:d]
        v1 = v[d:]
        X3v1[d:] = np.cumsum(v1)
        for i in range(d):
            X3v1 = np.cumsum(X3v1)
        if d == 0:
            Xv = X3v1
        elif d == 1:
            Xv = v[0] + X3v1
        else:
            X0v0 = np.dot(self._tf_X[:, :d], v0)
            Xv = X0v0 + X3v1
        return Xv


    def XTdotv_unscaled(self, v):
        d = self._tf_degree
        vrev = v[::-1]
        X3v1 = np.cumsum(vrev)
        for i in range(d):
            X3v1 = np.cumsum(X3v1)
        XTv = X3v1[::-1]
        ## degree 0 do not need any change
        if d == 0:
            # do nothing, but Python cares about indentation
            _ = -1
        elif d == 1:
            XTv[0] = np.sum(v)
        else:
            XTv[:d] = np.dot(self._tf_X[:, :d].T, v)
        return XTv


    def Xdotv(self, v):
        if self._tf_standardize_basis:
            return self.Xdotv_unscaled( v / self._tf_fstd ) - np.dot( v, self._tf_floc )
        elif self._tf_scale_basis:
            return self.Xdotv_unscaled(v) / self._tf_fstd
        else:
            return self.Xdotv_unscaled(v)


    def XTdotv(self, v):
        if self._tf_standardize_basis:
            return self.XTdotv_unscaled(v) / self._tf_fstd - (self._tf_floc * np.sum(v))
        elif self._tf_scale_basis:
            return self.XTdotv_unscaled(v) / self._tf_fstd 
        else:
            return self.XTdotv_unscaled(v)
