import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

"""CALL, PUT PRICES"""
def _d1(S0, K, rate, sigma, T, dividend_yield):
    return (np.log(S0/K)+(rate-dividend_yield+sigma**2/2)*T)/(sigma*np.sqrt(T))

def _d2(S0, K, rate, sigma, T, dividend_yield):
    return (np.log(S0/K)+(rate-dividend_yield-sigma**2/2)*T)/(sigma*np.sqrt(T))

def european_call_price(S0, K, rate, sigma, T, dividend_yield=0.0):
    d1 = _d1(S0, K, rate, sigma,T, dividend_yield)
    d2 = _d2(S0, K, rate, sigma, T, dividend_yield)
    return S0 * np.exp(-dividend_yield * T)* norm.cdf(d1) - K*np.exp(-rate*T)*norm.cdf(d2)

def european_put_price(S0, K, rate, sigma, T, dividend_yield=0.0):
    d1 = _d1(S0, K, rate, sigma,T, dividend_yield)
    d2 = _d2(S0, K, rate, sigma, T, dividend_yield)
    return K*np.exp(-rate*T) * norm.cdf(-d2) - S0*np.exp(-dividend_yield * T)*norm.cdf(-d1)

"""GREEKS"""
# Delta
def call_delta(S0, K, rate, sigma, T, dividend_yield=0.0):
    d1 = _d1(S0, K, rate, sigma, T, dividend_yield)
    return np.exp(-dividend_yield * T)*norm.cdf(d1)

def put_delta(S0, K, rate, sigma, T, dividend_yield=0.0):
    d1 = _d1(S0, K, rate, sigma, T, dividend_yield)
    return np.exp(-dividend_yield * T) * (norm.cdf(d1) - 1)

# Gamma
def gamma(S0, K, rate, sigma, T, dividend_yield=0.0):
    d1 = _d1(S0, K, rate, sigma, T, dividend_yield)
    return (np.exp(-dividend_yield * T)*norm.pdf(d1)/(S0 * sigma * np.sqrt(T)))

# Vega
def vega(S0, K, rate, sigma, T, dividend_yield=0.0):
    d1 = _d1(S0, K, rate, sigma, T, dividend_yield)
    return (S0*np.exp(-dividend_yield * T)*norm.pdf(d1)*np.sqrt(T))

# Theta
def call_theta(S0, K, rate, sigma, T, dividend_yield=0.0):
    d1 = _d1(S0, K, rate, sigma, T, dividend_yield)
    d2 = _d2(S0, K, rate, sigma, T, dividend_yield)
    return (-S0 * np.exp(-dividend_yield * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))- 
            rate * K * np.exp(-rate * T) * norm.cdf(d2) + dividend_yield * S0 * np.exp(-dividend_yield * T) * norm.cdf(d1))

def put_theta(S0, K, rate, sigma, T, dividend_yield=0.0):
    d1 = _d1(S0, K, rate, sigma, T, dividend_yield)
    d2 = _d2(S0, K, rate, sigma, T, dividend_yield)
    return (-S0 * np.exp(-dividend_yield * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        + rate * K * np.exp(-rate * T) * norm.cdf(-d2)-dividend_yield * S0 * np.exp(-dividend_yield * T) * norm.cdf(-d1))

# Rho
def call_rho(S0, K, rate, sigma, T, dividend_yield=0.0):
    d2 = _d2(S0, K, rate, sigma, T, dividend_yield)
    return K * T * np.exp(-rate * T) * norm.cdf(d2)
def put_rho(S0, K, rate, sigma, T, dividend_yield=0.0):
    d2 = _d2(S0, K, rate, sigma, T, dividend_yield)
    return -K * T * np.exp(-rate * T) * norm.cdf(-d2)

"""Implied volatility"""

def implied_volatility(market_price, S0, K, rate, T, dividend_yield=0.0):
    def objective(sigma):
        return european_call_price(S0, K, rate, sigma, T, dividend_yield) - market_price
    return brentq(objective, a=1e-6, b=5.0)
