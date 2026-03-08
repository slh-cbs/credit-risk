import numpy as np
from scipy.stats import norm


# =====================================================
# Helpers
# =====================================================

def _d1(spot, strike, vol, rate, div, ttm):
    return (np.log(spot / strike) + (rate - div + 0.5 * vol**2)*ttm) / (vol*np.sqrt(ttm))


def _d2(spot, strike, vol, rate, div, ttm):
    return _d1(spot, strike, vol, rate, div, ttm) - vol*np.sqrt(ttm)


# =====================================================
# Vanilla Black–Scholes
# =====================================================

def call(spot, vol, rate, ttm, strike, div=0.0):
    if ttm < 0:
        raise ValueError("Time to maturity must be non-negative")
    elif ttm > 0:
        d1 = _d1(spot, strike, vol, rate, div, ttm)
        d2 = _d2(spot, strike, vol, rate, div, ttm)
        return (
            spot*np.exp(-div*ttm)*norm.cdf(d1)
            - strike*np.exp(-rate*ttm)*norm.cdf(d2)
        )
    else:
        return max(spot - strike, 0.0)


def put(spot, vol, rate, ttm, strike, div=0.0):
    C = call(spot, vol, rate, ttm, strike, div)
    return C - (spot*np.exp(-div*ttm) - strike*np.exp(-rate*ttm))


def call_delta(spot, vol, rate, ttm, strike, div=0.0):
    if ttm < 0:
        return ValueError("Time to maturity must be non-negative")
    elif ttm > 0:
        d1 = _d1(spot, strike, vol, rate, div, ttm)
        return norm.cdf(d1)
    else:
        if spot > strike:
            return 1
        elif spot < strike:
            return 0
        else:
            return np.nan
        

# =====================================================
# Generic Down-and-Out Claim
# =====================================================

def down_and_out_claim(
    truncated_valuation_no_barrier,
    spot,
    barrier,
    vol,
    r,
    ttm,
    *args,
    div=0.0
):
    """
    Generic down-and-out valuation using Björk's book.

    Parameters
    ----------
    truncated_valuation_no_barrier : callable
        Function implementing Black-Scholes-type valuation WITHOUT barrier.
        Must have signature:
            f(spot, vol, r, T, *args, div=div)

    spot : float
        Spot
    barrier : float
        Lower barrier
    vol : float
        Volatility
    r : float
        Risk-free rate
    ttm : float
        Time to maturity
    div : float
        Continuous dividend yield

    *args
        Passed to truncated_valuation_no_barrier (e.g. strike K)

    Returns
    -------
    float
    """
    if ttm < 0:
        return ValueError("Time to maturity must be non-negative")

    if spot <= barrier:
        return 0.0

    if ttm == 0:
        return truncated_valuation_no_barrier(
            spot, barrier, vol, r, ttm, *args, div=div
        )

    vanilla = truncated_valuation_no_barrier(
        spot, barrier, vol, r, ttm, *args, div=div
    )

    mirror = (
        (barrier / spot)**(2 * (r - div - 0.5 * vol**2) / vol**2)
        * truncated_valuation_no_barrier(
            barrier**2 / spot, barrier, vol, r, ttm, *args, div=div
        )
    )

    return vanilla - mirror


# =====================================================
# Down-and-Out Wrappers
# =====================================================


def truncated_stock(spot, barrier, vol, rate, ttm, div=0.0):
    return spot * np.exp(-div*ttm) * norm.cdf((_d1(spot, barrier, vol, rate, div, ttm)))

def down_and_out_stock(spot, barrier, vol, rate, ttm, div=0.0):
    return down_and_out_claim(
        truncated_stock,
        spot,
        barrier,
        vol,
        rate,
        ttm,
        div=div
    )

def truncated_call(spot, barrier, vol, rate, ttm, strike, div=0.0):
    return call(spot, vol, rate, ttm, strike, div)

def down_and_out_call(spot, barrier, vol, rate, ttm, strike, div=0.0):
    # assuming strike > barrier

    return down_and_out_claim(
        truncated_call,
        spot,
        barrier,
        vol,
        rate,
        ttm,
        strike,
        div=div
    )