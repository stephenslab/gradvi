
"""
test_method_comparsion
    All inversion methods yield similar result.

test_fssi_homogeneous_variance
    FSSI throws error if NM variance are unequal.

test_expectations
    Results should match some of our expectations:
        .. M(x) = b where x is obtained after inverting b
        .. For a NM model z ~ N(a, sj2), calculate posterior mean.
            Then, invert of the posterior mean equals z.
"""
