import numpy as np
from scipy.integrate import quad

"""
Implements pricing from Heston model through the characteristic function
Semi-analytical from numerical integration
to speed up pricing compared to Monte Carlo
"""


def heston_characteristic_function(u, S0, v0, rate, dividend_yield, kappa, theta, ksi, rho, T):
    """
    Calculates Heston characteristic function to evaluate integrals for P1 and P2
    """
    x0 = np.log(S0)

    a = kappa*theta
    b = kappa-rho*ksi*1j*u
    d = np.sqrt(b**2+ksi**2*(1j*u+u**2))
    g = (b-d)/(b+d)

    C = ((rate - dividend_yield) * 1j * u * T + (a / ksi**2) * ((b - d)*T-2 * np.log((1 - g * np.exp(-d * T))/ (1 - g))))

    D = ((b - d) / ksi**2* ((1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))))

    return np.exp(C + D * v0 + 1j * u * x0)

def _heston_probability(j, S0, K, v0, rate, dividend_yield, kappa, theta, ksi, rho, T):
    """
    Calculates P1 and P2 in call price as numerical integrals using Heston characteristic function
    """
    log_K = np.log(K)

    def integrand(u):
        if j == 1:
            numerator = heston_characteristic_function(u - 1j, S0, v0, rate, dividend_yield, kappa, theta, ksi, rho, T)
            denominator = heston_characteristic_function(-1j, S0, v0, rate, dividend_yield, kappa, theta, ksi, rho, T)
        else:
            numerator = heston_characteristic_function(u, S0, v0, rate, dividend_yield, kappa, theta, ksi, rho, T)
            denominator = 1.0

        return np.real(np.exp(-1j*u*log_K)*numerator/(1j*u*denominator))

    integral, _ = quad(integrand, 1e-8, 100.0, limit=200)

    return 0.5 + integral / np.pi

def price_heston_call(S0, K, v0,rate, dividend_yield, kappa, theta, ksi, rho, T):
    """
    Call price
    C = S0*exp(-q*T)*P1 - K*exp(-r*T)*P2
    """
    P1 = _heston_probability(1, S0, K, v0, rate, dividend_yield, kappa, theta, ksi, rho, T)
    P2 = _heston_probability(2, S0, K, v0, rate, dividend_yield, kappa, theta, ksi, rho, T)
    return (S0 * np.exp(-dividend_yield * T) * P1-K * np.exp(-rate * T) * P2)

def price_heston_put(S0, K, v0, rate, dividend_yield, kappa, theta, ksi, rho, T):
    """
    From put call parity
    """
    call = price_heston_call(S0, K, v0, rate, dividend_yield, kappa, theta, ksi, rho, T)
    return (call-S0 * np.exp(-dividend_yield * T)+K*np.exp(-rate * T))