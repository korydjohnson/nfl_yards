
import numpy as np
from scipy.stats import norm

def f1(params, observed):
    '''
    Optimize CRPS via normal distribution

    :param params: mean and variance of normal distribution
    :param observed: observed target
    :return: CRPS
    '''
    p1, p2 = params
    vals = np.arange(-99,100)
    vals_scaled = np.log(vals.clip(-14,99)+15)
    predicted = norm.cdf(vals_scaled, p1, p2)
    predicted = np.tile(predicted, (len(observed), 1))
    score = np.mean((observed - predicted) ** 2)
    return score
