# GradVI - Gradient Descent Variational Inference

GradVI provides tools for Bayesian variational inference using gradient descent methods.
The user specifies a prior and a task 
(e.g. linear regression, GLM regression, trendfiltering),
observes data and runs posterior inference.
The goal is to learn the parameters of the corresponding variational posterior family.

Currently, only adaptive shrinkage (ASH) prior has been implemented.
A coordinate ascent algorithm for multiple linear regression with ASH prior is available here: 
[mr.ash.alpha](https://github.com/stephenslab/mr.ash.alpha).

<!-- Future work includes extension to other types of distributions -->
Theory for GradVI: [Link to Overleaf](https://www.overleaf.com/project/60d0d9301e098e4dbe8e3521)

### Installation
For development, download this repository and install using `pip`:

```bash
git clone https://github.com/stephenslab/gradvi.git # or use the SSH link
cd gradvi
pip install -e .
```

### How to use
Functions are not documented yet. Here, I show an example to get started:

__Example of Linear regression__

Simulate some data:

```python
import numpy as np
from gradvi.priors import Ash
from gradvi.inference import  LinearRegression

n = 100
p = 200
pcausal = 20
s2 = 1.4
np.random.seed(100)

X = np.random.normal(0, 1, size = n * p).reshape(n, p)
b = np.zeros(p)
b[:pcausal] = np.random.normal(0, 1, size = pcausal)
err = np.random.normal(0, np.sqrt(s2), size = n)
y = np.dot(X, b) + err
```

Perform regression:

```python
prior = Ash(sk, scaled = True)
gvlin = LinearRegression(debug = False, display_progress = False)
gvlin.fit(X, y, prior)

b_hat = gvlin.coef
```

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
