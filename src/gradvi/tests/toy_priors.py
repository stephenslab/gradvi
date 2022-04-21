
"""
Toy priors for testing
"""
import numpy as np
from gradvi.priors import Ash

def get_all(k = 6, sparsity = 0.6, scale = 2.0):
    """
    Get a list of priors for testing
    """
    priors = list()
    # ==============================
    # Manually list all priors here.
    # ==============================
    #
    # Ash
    wk = np.zeros(k)
    wk[0] = sparsity
    wk[1:(k-1)] = np.repeat((1 - wk[0])/(k-1), (k - 2))
    wk[k-1] = 1 - np.sum(wk)
    sk = (np.power(scale, np.arange(k) / k) - 1)
    prior = Ash(sk, wk = wk, scaled = False)
    priors.append(prior)
    #
    # Ash scaled
    prior = Ash(sk, wk = wk, scaled = True)
    priors.append(prior)
    #
    return priors


def get_from_same_class(prior, wk):
    """
    Return a new_prior from the same class as `prior`
    with a new set of parameters `wk`
    """
    if prior.prior_type == 'ash':
        new_prior = Ash(prior.sk, wk = wk, scaled = False)
    elif prior.prior_type == 'ash_scaled':
        new_prior = Ash(prior.sk, wk = wk, scaled = True)
    return new_prior
