
"""
This is a factory method which creates the required Normal Means model
based on the prior, and returns the Normal Means class.
The Normal Means class provides two main operators we need for GradVI:
    - Shrinkage Operator :math:`M(x)`
    - Penalty Operator :math:`\rho_j(x_j)`

See the following for other ways of creating factory:
https://medium.com/@vadimpushtaev/python-choosing-subclass-cf5b1b67c696
https://stackoverflow.com/questions/27322964

"""

def NormalMeans(y, prior, sj2, **kwargs):
        return prior.normal_means(y, prior, sj2, **kwargs)

#class NormalMeans:
#
#    @classmethod
#    def create(self, y, prior, sj2, **kwargs):
#        return prior.normal_means(y, prior, sj2, **kwargs)
