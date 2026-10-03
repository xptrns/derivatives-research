import numpy as np
from scipy.optimize import least_squares
from pricing.heston_cf import price_heston_call

"""
Module for calibration of heston model parameters from market data using least squares regression
"""

def heston_residuals(params, market_data, S0, rate, dividend_yield):
    """
    Residual function
    Uses heston model and market data
    """
    v0, kappa, theta, ksi, rho = params
    residuals = []
    for row in market_data:
        strike_price = row["strike"]
        maturity = row["maturity"]
        market_price = row["price"]

        model_price = price_heston_call(S0=S0, K=strike_price, v0=v0, rate=rate, dividend_yield=dividend_yield,
                     kappa=kappa, theta=theta, ksi=ksi, rho=rho, T=maturity)
        residuals.append(model_price - market_price)

    return np.array(residuals)


def calibrate_heston(initial_guess, market_data, S0, rate, dividend_yield):
    result = least_squares(heston_residuals, x0=initial_guess, args=(market_data, S0, rate, dividend_yield,),
                            bounds=([1e-6, 1e-6, 1e-6, 1e-6, -0.999],[2.0, 20.0, 2.0, 5.0, 0.999]))
    return result