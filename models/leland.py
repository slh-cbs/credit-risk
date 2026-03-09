
import math

def _beta2(r, delta, sigma):
    sigma2 = sigma ** 2
    drift_V = r - delta
    drift_logV = drift_V - 0.5 * sigma2
    return (-drift_logV - math.sqrt(drift_logV**2 + 2 * sigma2 * r)) / sigma2

def _arrow_debreu_price(V, V_B, beta2):
    return (V / V_B)**beta2

def _optimal_default_boundary(C, tau, r, beta2):
    return (beta2 / (beta2 - 1)) * (1-tau) * (C / r)

def optimal_default_boundary(C, tau, r, delta, sigma):
    beta2 = _beta2(r, delta, sigma)
    return _optimal_default_boundary(C, tau, r, beta2)

def _optimal_coupon(V, tau, alpha, r, beta2):
    return V * ((beta2 - 1) / beta2) * (r / (1 - tau)) * (((1-beta2)*tau - alpha*beta2*(1-tau))/tau) ** (1 / beta2)

def optimal_coupon(V, tau, alpha, r, delta, sigma):
    beta2 = _beta2(r, delta, sigma)
    return _optimal_coupon(V, tau, alpha, r, beta2)

def _equity_value(V, V_B, C, tau, r, beta2):
    E_inf = V - (1 - tau) * C / r
    E_B = V_B - (1 - tau) * C / r
    P_B = _arrow_debreu_price(V, V_B, beta2)
    return E_inf - E_B * P_B

def equity_value(V, tau, alpha, r, delta, sigma):
    C = optimal_coupon(V, tau, alpha, r, delta, sigma)
    V_B = optimal_default_boundary(C, tau, r, delta, sigma)
    beta2 = _beta2(r, delta, sigma)
    return _equity_value(V, V_B, C, tau, r, beta2)

def _debt_value(V, V_B, C, alpha, r, beta2):
    D_inf = C / r
    D_B = (1-alpha) * V_B
    P_B = _arrow_debreu_price(V, V_B, beta2)
    return D_inf + (D_B - D_inf) * P_B

def debt_value(V, tau, alpha, r, delta, sigma):
    C = optimal_coupon(V, tau, alpha, r, delta, sigma)
    V_B = optimal_default_boundary(C, tau, r, delta, sigma)
    beta2 = _beta2(r, delta, sigma)
    return _debt_value(V, V_B, C, alpha, r, beta2)

def _tax_benefit_value(V, V_B, C, tau, r, beta2):
    TB_inf = tau * C / r
    TB_B = 0.0
    P_B = _arrow_debreu_price(V, V_B, beta2)
    return TB_inf + (TB_B - TB_inf) * P_B

def tax_benefit_value(V, tau, alpha, r, delta, sigma):
    C = optimal_coupon(V, tau, alpha, r, delta, sigma)
    V_B = optimal_default_boundary(C, tau, r, delta, sigma)
    beta2 = _beta2(r, delta, sigma)
    return _tax_benefit_value(V, V_B, C, tau, r, beta2)

def _bankruptcy_costs_value(V, V_B, alpha, beta2):
    BC_inf = 0.0
    BC_B = alpha * V_B
    P_B = _arrow_debreu_price(V, V_B, beta2)
    return BC_inf + (BC_B - BC_inf) * P_B

def bankruptcy_costs_value(V, tau, alpha, r, delta, sigma):
    C = optimal_coupon(V, tau, alpha, r, delta, sigma)
    V_B = optimal_default_boundary(C, tau, r, delta, sigma)
    beta2 = _beta2(r, delta, sigma)
    return _bankruptcy_costs_value(V, V_B, alpha, beta2)

def all_output(V, tau, alpha, r, delta, sigma):
    C = optimal_coupon(V, tau, alpha, r, delta, sigma)
    V_B = optimal_default_boundary(C, tau, r, delta, sigma)
    beta2 = _beta2(r, delta, sigma)
    E = _equity_value(V, V_B, C, tau, r, beta2)
    D = _debt_value(V, V_B, C, alpha, r, beta2)
    TB = _tax_benefit_value(V, V_B, C, tau, r, beta2)
    BC = _bankruptcy_costs_value(V, V_B, alpha, beta2)
    return E, D, TB, BC
