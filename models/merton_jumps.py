import math
from scipy.stats import poisson

import models.black_scholes as bs

def equity_value(V, D, sigma_V, r, T, lambda_, gamma, delta, tol=1e-6):
    k = math.exp(gamma) - 1
    lambda_k = lambda_ * (1 + k)
    sigma2 = sigma_V ** 2
    delta2 = delta ** 2

    diff = 10
    n = 0
    sum = 0.0
    while diff > tol:
        r_n = r - lambda_ * k + n * gamma / T
        sigma_n = math.sqrt(sigma2 + n * delta2 / T)
        prob_n = poisson.pmf(n, mu=lambda_k*T)
        diff = prob_n * bs.call(spot=V, strike=D, vol=sigma_n, rate=r_n, ttm=T)
        sum += diff
        n += 1

    return sum

def debt_value(V, D, sigma_V, r, T, lambda_, gamma, delta, tol=1e-6):
    S = equity_value(V, D, sigma_V, r, T, lambda_, gamma, delta, tol=1e-6)
    return V - S
    
def credit_spread(V, D, sigma_V, r, T, lambda_, gamma, delta, tol=1e-6):
    if T == 0:
        # in the case V > D, we use the limit T -> 0
        return lambda_ * (V / D) * math.exp(gamma) * bs.put(spot=1.0, strike=D/V, vol=delta, rate=gamma, ttm=1.0) if V > D else math.inf
    else:
        B = debt_value(V, D, sigma_V, r, T, lambda_, gamma, delta, tol=1e-6)
        return - math.log(B / D) / T - r
