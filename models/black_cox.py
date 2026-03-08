import math
import models.black_scholes as bs

from scipy.stats import norm

def equity_value(V, sigma_V, r, T, D, C, gamma, t=0.0, a=0.0):
    B = debt_value(V, sigma_V, r, T, D, C, gamma, t, a)
    return V - B

def B_m(V, sigma_V, r, T, D, C, gamma, t, a):
    K = D * math.exp(-gamma*T)
    L = C * math.exp(-gamma*T)

    exp_gammat = math.exp(gamma * t)

    V_LO = exp_gammat * bs.down_and_out_stock(spot = V / exp_gammat, barrier = L, vol = sigma_V, rate = r - gamma, ttm = T - t, div = a)
    C_LO = exp_gammat * bs.down_and_out_call(spot = V / exp_gammat, strike = K, barrier = L, vol = sigma_V, rate = r - gamma, ttm = T - t, div = a)

    return V_LO - C_LO

def B_b(V, sigma_V, r, ttm, C, gamma, a):
    b = (math.log(C / V) - gamma * ttm) / sigma_V
    mu = (r - a - gamma - 0.5 * sigma_V**2) / sigma_V
    muTilde = math.sqrt(mu**2 + 2 * (r - gamma))

    return C * math.exp(-gamma*ttm + (mu-muTilde)*b) * (norm.cdf((b - muTilde*ttm)/math.sqrt(ttm)) + math.exp(2*muTilde*b)*norm.cdf((b + muTilde*ttm)/math.sqrt(ttm)))

def debt_value(V, sigma_V, r, T, D, C, gamma, t=0.0, a=0.0):

    return B_m(V, sigma_V, r, T, D, C, gamma, t, a) + B_b(V, sigma_V, r, T-t, C, gamma, a)
    
def credit_spread(V, sigma_V, r, T, D, C, gamma, t=0.0, a=0.0):
    if T-t == 0:
        return 0.0 if V > D else math.inf
    else:
        B = debt_value(V, sigma_V, r, T, D, C, gamma, t, a)
        return - math.log(B / D) / (T-t) - r