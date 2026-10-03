import numpy as np
from pricing.heston_cf import price_heston_call


def price_sensitivity(params, K, T, S0, rate, dividend_yield):
    """
    Numerical gradient of the Heston price
    to the five model parameters.
    Euler method
    returns Jacobian
    """
    params = np.asarray(params, dtype=float)
    sensitivities = np.zeros(5)

    for j in range(5):
        # infinitesimal change
        h = 1e-5

        params_up = params.copy()
        params_down = params.copy()
        params_up[j] += h
        params_down[j] -= h

        price_up = price_heston_call(S0=S0, K=K, v0=params_up[0], rate=rate, dividend_yield=dividend_yield,
            kappa=params_up[1], theta=params_up[2], ksi=params_up[3], rho=params_up[4], T=T)
        price_down = price_heston_call(S0=S0, K=K, v0=params_down[0], rate=rate, dividend_yield=dividend_yield, kappa=params_down[1],
                theta=params_down[2], ksi=params_down[3], rho=params_down[4], T=T)

        sensitivities[j] = (price_up-price_down)/(2*h)

    return sensitivities

def scale_fisher_matrix(fisher_matrix, scales):
    scales = np.asarray(scales)
    scaling_matrix = np.diag(scales)
    return scaling_matrix @ fisher_matrix @ scaling_matrix
