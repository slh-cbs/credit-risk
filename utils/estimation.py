import math
import numpy as np
from scipy.optimize import brentq, bisect, newton, minimize_scalar

def invert_function(
    value,
    function,
    params,
    V_min=1e-6,
    V_max=1e6,
    method="brentq"
):
    def objective(V):
        return function(V=V, **params) - value

    method = method.lower()

    if method == "brentq":
        return brentq(objective, V_min, V_max)

    elif method == "bisection":
        return bisect(objective, V_min, V_max)

    elif method == "secant":
        # When no derivative is passed to newton, it uses the secant method
        V0 = V_min
        V1 = V_max
        return newton(objective, V0, V1)

    else:
        raise ValueError(
            "Unknown method. Choose from: 'brentq', 'bisection', 'secant'"
        )

def mle_estimation_gbm(dates, values, days_in_year=365):
    # assume data is sorted
    dates, values = zip(*sorted(zip(dates, values)))

    dt = np.array([
        (dates[i] - dates[i-1]).days / days_in_year
        for i in range(1, len(dates))
    ])

    x = np.log(values)
    dx = np.diff(x)

    # MLE for alpha
    alpha_hat = (x[-1] - x[0]) / ((dates[-1] - dates[0]).days / days_in_year)

    # MLE for sigma^2
    sigma2_hat = np.mean((dx - alpha_hat * dt)**2 / dt)

    # Recover mu and sigma
    mu_hat = alpha_hat + 0.5 * sigma2_hat
    sigma_hat = np.sqrt(sigma2_hat)

    return {
        "mu": mu_hat,
        "sigma": sigma_hat
    }

def mle_estimation_gbm_from_transformation(dates, values, T_function, T_derivative, T_params, days_in_year=365, tol=1e-6):
    if len(values) < 2:
        raise ValueError("At least two observations are required for MLE.")

    # Ensure sorted by date
    dates, values, T_params = zip(*sorted(zip(dates, values, T_params)))
    
    N = len(values) - 1
    total_time = ((dates[-1] - dates[0]).days / days_in_year)

    def neg_log_likelihood(log_sigma):
        sigma = math.exp(log_sigma)
        params = {**T_params[0], 'sigma_V': sigma}

        v0 = invert_function(values[0], T_function, params)
        x0 = math.log(v0)


        sum1 = sum2 = sum3 = sum4 = 0.0
        x_prev = x0
        for i in range(1, N + 1):
            params = {**T_params[i], 'sigma_V': sigma}
            dt = (dates[i] - dates[i-1]).days / days_in_year

            vi = invert_function(values[i], T_function, params)
            x_curr = math.log(vi)

            dx = x_curr - x_prev

            sum1 += math.log(dt)
            sum2 += dx**2 / dt
            sum3 += math.log(T_derivative(vi, **params))
            sum4 += x_curr

            x_prev = x_curr

        xN = x_prev
        alpha_hat = (xN - x0) / total_time
        sum2 -= (alpha_hat ** 2) * total_time

        l = 0.5 * N * math.log(2 * math.pi) + N * log_sigma
        l += 0.5 * sum1
        l += 0.5 * sum2 / (sigma ** 2)
        l += sum3
        l += sum4
        return l

    result = minimize_scalar(
        neg_log_likelihood,
        method='brent',
        tol=tol
    )
    
    sigma_hat = math.exp(result.x)
    v0 = invert_function(values[0], T_function, {**T_params[0], 'sigma_V': sigma_hat})
    vN = invert_function(values[-1], T_function, {**T_params[-1], 'sigma_V': sigma_hat})

    alpha_hat = (math.log(vN) - math.log(v0)) / total_time
    mu_hat = alpha_hat + 0.5 * sigma_hat**2

    return {
        "mu": mu_hat,
        "sigma": sigma_hat
    }
