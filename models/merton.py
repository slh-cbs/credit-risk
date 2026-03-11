import numpy as np
from scipy.stats import norm

import models.black_scholes as bs
import utils.estimation as ue

def equity_value(V, D, sigma_V, r, T):
    return bs.call(spot=V, strike=D, vol=sigma_V, rate=r, ttm=T)

def equity_delta(V, D, sigma_V, r, T):
    return bs.call_delta(spot=V, strike=D, vol=sigma_V, rate=r, ttm=T)

def debt_value(V, D, sigma_V, r, T):
    return V - bs.call(spot=V, strike=D, vol=sigma_V, rate=r, ttm=T)
    
def credit_spread(V, D, sigma_V, r, T):
    if T == 0:
        return 0.0 if V > D else np.inf
    else:
        B = debt_value(V, D, sigma_V, r, T)
        return - np.log(B / D) / T - r
    
def default_probability(V, D, sigma_V, mu, T):
    if T < 0:
        return ValueError("Time to maturity must be non-negative")
    elif T > 0:
        d2 = (np.log(V / D) + (mu - 0.5 * sigma_V**2)*T) / (sigma_V*np.sqrt(T))
        return norm.cdf(-d2)
    else:
        return 0.0 if V >= D else 1.0
    
def mle_estimation_from_asset_values(dates, values, days_in_year):
    return ue.mle_estimation_gbm(dates, values, days_in_year)

def mle_estimation_from_equity_values(dates, equity_values, face_values, short_rates, T, days_in_year, tol):

    params = [{'D': D, 'r': r, 'T': T} for D, r in zip(face_values, short_rates)]
    
    return ue.mle_estimation_gbm_from_transformation(
        dates=dates,
        values=equity_values,
        T_function=equity_value,
        T_derivative=equity_delta,
        T_params=params,
        days_in_year=days_in_year,
        tol=tol
    )

def implied_asset_value(S, D, sigma_V, r, T):
    params = {'sigma_V': sigma_V, 'D': D, 'r': r, 'T': T}
    return ue.invert_function(S, equity_value, params)

def vassalou_xing_estimation(dates, equity_values, face_values, short_rates, T, days_in_year, tol):

    # we use sigma from a GBM estimation as our starting guess
    mle_estimates = ue.mle_estimation_gbm(dates=dates, values=equity_values, days_in_year=days_in_year)
    sigma = mle_estimates['sigma']

    V = [implied_asset_value(S, D, sigma, r, T) for S, D, r in zip(equity_values, face_values, short_rates)]

    diff = 10.0
    while diff > tol:
        mle_estimates = ue.mle_estimation_gbm(dates=dates, values=V, days_in_year=days_in_year)
        sigma_new = mle_estimates['sigma']
        V = [implied_asset_value(S, D, sigma_new, r, T) for S, D, r in zip(equity_values, face_values, short_rates)]
        diff = abs(sigma_new - sigma)
        sigma = sigma_new
    
    return mle_estimates
