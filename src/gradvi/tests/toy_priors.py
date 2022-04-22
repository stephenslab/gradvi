
"""
Toy priors for testing
"""
import numpy as np
from gradvi.priors import Ash

def get_all(**kwargs):
    """
    Get a list of priors for testing
    """
    priors = list()
    # ==============================
    # Manually list all priors here.
    # ==============================
    #
    # Ash
    prior = get_ash(**kwargs)
    priors.append(prior)
    #
    # Ash scaled
    prior = get_ash_scaled(**kwargs)
    priors.append(prior)
    #
    return priors


def get_ash(k = 6, sparsity = 0.6, skbase = 2.0, is_scaled = False, **kwargs):
    wk = np.zeros(k)
    wk[0] = 1.0 / k if sparsity is None else sparsity
    wk[1:(k-1)] = np.repeat((1 - wk[0])/(k-1), (k - 2))
    wk[k-1] = 1 - np.sum(wk)
    sk = (np.power(skbase, np.arange(k) / k) - 1)
    prior = Ash(sk, wk = wk, scaled = is_scaled)
    return prior


def get_ash_scaled(k = 6, sparsity = 0.6, skbase = 2.0, **kwargs):
    return get_ash(k = k, sparsity = sparsity, skbase = skbase, is_scaled = True, **kwargs)


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
