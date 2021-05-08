import numpy as np
from scipy.stats import multivariate_normal as norm
from scipy.special import binom
from tqdm import tqdm


def calculate_binom_logs(n):
    logs = np.cumsum(np.insert(np.log(np.arange(1, n + 1)), 0, 0))
    return logs[n] - logs - np.flip(logs)


class DistrClass:
    def __init__(self, beta, max_size=1000, J=0.15):
        self.sigma = (1 - beta) * np.eye(max_size) + beta * np.ones((max_size, max_size))
        self.distributions = [norm(cov=self.sigma[:k, :k]) for k in range(1, max_size + 1)]
        self.max_size = max_size
        self.J = J
        self.log_binoms = calculate_binom_logs(max_size)

    def segment_prob(self, a, b, n):
        '''
        Parameters:
        a: float
        b: float, a <= b
        n: number of variables
        Gives a approximate logarithm of probability P(x1, ... xn in [a, b]^n),
        if x1, ... xn are drawn from multivariate normal
        distribution with covariance beta.
        '''
        return self.distributions[n - 1].logpdf(np.full(n, b)) + n * np.log(b - a)
    
    def get_mean(self, f):
        probs = [self.segment_prob(f - 2 * self.J * s / self.max_size, f, s - 1) for s in range(1, 