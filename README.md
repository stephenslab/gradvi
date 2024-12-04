# GradVI - Gradient Descent Variational Inference

GradVI provides tools for Bayesian variational inference using gradient descent methods.
It is a modular software which provides the boilerplate for variational inference.
The user specifies a prior family of distribution and a task (e.g. linear regression, trendfiltering),
observes data and runs posterior inference.
The goal is to learn the parameters of the corresponding variational posterior family.

Currently, two different prior distributions, namely (1) adaptive shrinkage (ASH) prior,
and (2) point-normal prior are provided within the software.
For any other choice, the user has to define the prior distribution following the examples
provided within the framework.

### Related software
- [mr.ash.alpha](https://github.com/stephenslab/mr.ash.alpha) A coordinate ascent algorithm for multiple linear regression with ASH prior.
- [mr-ash-pen](https://github.com/banskt/mr-ash-pen) A fast FORTRAN core for GradVI multiple regression using ASH prior.

<!-- Future work includes extension to other types of distributions -->
Theory for GradVI: [Link to Overleaf](https://www.overleaf.com/project/60d0d9301e098e4dbe8e3521)

## Installation

The software can be installed directly from github using `pip`:
```bash
pip install git+https://github.com/stephenslab/gradvi
```

For development, download this repository and install using the `-e` flag:
```bash
git clone https://github.com/stephenslab/gradvi.git # or use the SSH link
cd gradvi
pip install -e .
```

## Quick Start

The software provides several classes for performing variational inference.
Try running the following small examples that illustrates using some classes.

__Example of Linear regression__

Simulate some data:

```python
import numpy
import matplotlib.pyplot
from gradvi.priors import Ash
from gradvi.inference import  LinearRegression

n = 100
p = 200
pcausal = 20
s2 = 1.4
k = 10
sk = (numpy.power(2.0, numpy.arange(k) / k) - 1)
numpy.random.seed(100)

X = numpy.random.normal(0, 1, size = n * p).reshape(n, p)
b = numpy.zeros(p)
b[:pcausal] = numpy.random.normal(0, 1, size = pcausal)
err = numpy.random.normal(0, numpy.sqrt(s2), size = n)
y = numpy.dot(X, b) + err
```

Perform regression:

```python
prior = Ash(sk, scaled = True)
gvlin = LinearRegression(debug = False, display_progress = True)
gvlin.fit(X, y, prior)
b_hat = gvlin.coef
```

Compare the true regression coefficients against the estimated
coefficients:

```python
matplotlib.pyplot.scatter(b,b_hat,s = 10,color = "black")
matplotlib.pyplot.axline((0,0),slope = 1,color = "magenta",linestyle = ":")
matplotlib.pyplot.xlabel("true")
matplotlib.pyplot.ylabel("estimated")
matplotlib.pyplot.show()
```

## Credits

The GradVI Python package was developed by
[Saikat Banerjee](https://github.com/banskt) at the
[University of Chicago](https://www.uchicago.edu/), with contributions
from [Peter Carbonetto](https://github.com/pcarbo) and
[Matthew Stephens](https://stephenslab.uchicago.edu/).

<!--
__Defaults__
```python
from gradvi.inference import LinearRegression
gvlin = LinearRegression()
gvlin.fit(X, y)

from gradvi.inference import Trendfilter
gvtf = Trendfilter(order = 1)
gvtf.fit(y)

from gradvi.inference import GLMRegression
gvglm = GLMRegression(model = "Poisson")
gvglm.fit(X, y)
```

__Linear Regression with minimization options and specified prior__
```python
from gradvi.inference import LinearRegression, Minimizer
from gradvi.priors import ASH

minimizer = Minimizer(method = 'L-BFGS-B')
prior = ASH(wk, sk, scaled = True)
gvlin = LinearRegression(prior = prior, debug = True)
gvlin.fit(X, y)
```
-->

<!--
### Demonstration

[Link](https://banskt.github.io/iridge-notes/2021/08/24/mrash-penalized-trend-filtering-demo.html) 
to demonstration on simple examples of linear data and trend-filtering data.

### How to use

Functions are not documented yet. Here is only a quick start.

```
from mrashpen.inference.penalized_regression import PenalizedRegression as PLR
plr = PLR(method = 'L-BFGS-B', optimize_w = True, optimize_s = True, is_prior_scaled = True, debug = False)
plr.fit()
```
| Returns | Description |
| --- | --- |
|`plr.coef` | optimized regression coefficients |
|`plr.prior` | optimized Mr.ASH prior mixture coefficients |
|`plr.obj_path` | Value of the objective function for all iterations |
|`plr.theta` | optimized parameter `theta` from the objective function |
|`plr.fitobj` | [OptimizeResult](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult) object from scipy.optimize |
| --- | --- |

### Running tests
Run the unittest from the `/path/to/download/mr-ash-pen` directory.
```
python -m unittest
```
-->
