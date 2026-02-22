import numpy as np
from scipy.stats import norm

def call(S, K, sigma, r, T):
    if T < 0:
        return ValueError("Time to maturity must be non-negative")
    elif T > 0:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return max(S - K, 0)

def put(S, K, sigma, r, T):
    C = call(S, K, sigma, r, T)
    return C - (S - K*np.exp(-r*T))

def call_delta(S, K, sigma, r, T):
    if T < 0:
        return ValueError("Time to maturity must be non-negative")
    elif T > 0:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma*np.sqrt(T))
        return norm.cdf(d1)
    else:
        if S > K:
            return 1
        elif S < K:
            return 0
        else:
            return np.nan