
"""
Some utility functions for Normal Means model
"""

def guess_nm_scale(sj2, s2, dj):
    if s2 is None:
        if dj is None:
            s2 = 1.0
            dj = 1.0 / sj2
        else:
            s2 = sj2 * dj
    if dj is None:
        dj = s2 / sj2
    return s2, dj


def get_optional_arg(key, default, **kwargs):
    x = kwargs[key] if key in kwargs.keys() else default
    return x
