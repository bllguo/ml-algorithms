import numpy as np
from numpy.random import normal, uniform
from scipy.stats import multivariate_normal as mv_norm

# goal: recover true parameters a, b
# recall in linear regression we assume the target variable t is given by deterministic y(x, w) with additive noise
# t = y(x, w) + epsilon
# epsilon is a zero mean Gaussian with precision beta
# THUS, it follows that p(t|x, w, beta) = N(t|y(x, w), beta^-1)
# First we generate some data with normally distributed noise
# following Bishop, a=0.5 and b=-0.3 and sigma=.2


def generator(x, a, b, sigma):
    N = len(x)
    if sigma == 0:
        return a*x + b
    else:
        return a*x + b + normal(0, sigma, N)


# generate x from uniform distribution U(x|-1, 1)

np.random.seed(1089)
x = uniform(-1, 1, 1000)
t = generator(x, .5, -.3, .2)

# we start with a noninformative prior for w - zero mean isotropic Gaussian. p(w|alpha) = N(w|0, (alpha^-1)I)
# set alpha to 2
# assume sigma was known. Then the precision parameter beta is 1/sigma^2 = 1/0.2^2 = 25

m0 = np.array([0, 0])
alpha = 2.0
s0 = (1/alpha)*np.identity(2)
beta = 25

# the closed-form solution for the posterior is another Gaussian. The relevant equations can be found in Bishop 3.3
# I would type them but this is really not a good place.
# TODO: make this a notebook
# TODO: step through derivation of posterior for multivariate Gaussian


class BayesianLinReg:
    def __init__(self, m0, s0, beta):
        self.prior = mv_norm(m0, s0)
        self.m0 = m0
        self.s0 = s0
        self.beta = beta
        self.mn = m0
        self.sn = s0
        self.posterior = self.prior

    def update_posterior(self, x, t):
        x = np.array(x)
        if x.ndim <= 1:
            x = x.reshape((-1, 1))
        t = np.array(t)
        x = np.concatenate((np.ones([x.shape[0], 1]), x), axis=1)
        t = t.reshape((-1, 1))
        self.sn = np.linalg.inv(np.linalg.inv(self.s0) + self.beta*x.T.dot(x))
        self.mn = self.sn.dot(np.linalg.inv(self.s0).dot(self.m0).reshape(-1, 1) + self.beta*x.T.dot(t))
        self.posterior = mv_norm(self.mn.flatten(), self.sn)

# TODO: add plots

test = BayesianLinReg(m0, s0, beta)
test.update_posterior(x[0:1], t[0:1])