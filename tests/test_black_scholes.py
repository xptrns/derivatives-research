import numpy as np

from pricing.blackscholes import *

def test_at_the_money_call():
    price = european_call_price(S0=100,K=100,rate=0.05,sigma=0.20,T=1.0)
    assert np.isclose(price, 10.4506, atol=1e-4)

def test_put_call_parity():
    S0 = 100
    K = 105
    rate = 0.05
    sigma = 0.20
    T = 1.5
    q = 0.02
    call = european_call_price(S0, K, rate, sigma, T, q)
    put = european_put_price(S0, K, rate, sigma, T, q)
    lefthandside = call - put
    righthandside = S0*np.exp(-q*T) - K*np.exp(-rate*T)

    assert np.isclose(lefthandside, righthandside)

def test_prices_are_positive():
    call = european_call_price(100, 100, 0.05, 0.20, 1.0)
    put = european_put_price(100, 100, 0.05, 0.20, 1.0)
    assert call > 0
    assert put > 0

def test_prices_increase_with_volatility_strike():
    call_lowstrike = european_call_price(100, 70, 0.05, 0.20, 1.0)
    call_highstrike = european_call_price(100, 130, 0.05, 0.20, 1.0)
    assert call_highstrike<call_lowstrike
    call_lowvol = european_call_price(100, 100, 0.05, 0.10, 1.0)
    call_highvol = european_call_price(100, 100, 0.05, 0.40, 1.0)
    assert call_lowvol < call_highvol
    put_lowstrike = european_put_price(100, 70, 0.05, 0.20, 1.0)
    put_highstrike = european_put_price(100, 130, 0.05, 0.20, 1.0)
    assert put_highstrike>put_lowstrike
    put_lowvol = european_put_price(100, 100, 0.05, 0.10, 1.0)
    put_highvol = european_put_price(100, 100, 0.05, 0.40, 1.0)
    assert put_lowvol < put_highvol

def test_call_delta_finite_difference():
    S0 = 100
    K = 105
    rate = 0.05
    sigma = 0.20
    T = 1.5
    q = 0.02
    h = 1e-4
    derivative_delta = (european_call_price(S0 + h, K, rate, sigma, T, q) - european_call_price(S0 - h, K, rate, sigma, T, q)) / (2 * h)
    function_delta = call_delta(S0, K, rate, sigma, T, q)
    assert np.isclose(function_delta, derivative_delta, rtol=1e-5)

def test_gamma_against_finite_difference():
    S0 = 100
    K = 105
    rate = 0.05
    sigma = 0.20
    T = 1.5
    q = 0.02
    h = 1e-3
    derivative_gamma = (european_call_price(S0 + h, K, rate, sigma, T, q) - 2 * european_call_price(S0, K, rate, sigma, T, q) + european_call_price(S0 - h, K, rate, sigma, T, q)) / h**2
    function_gamma = gamma(S0, K, rate, sigma, T, q)
    assert np.isclose(function_gamma, derivative_gamma, rtol=1e-4)

def test_vega_against_finite_difference():
    S0 = 100
    K = 105
    rate = 0.05
    sigma = 0.20
    T = 1.5
    q = 0.02
    h = 1e-5
    derivative_vega = (european_call_price(S0, K, rate, sigma + h, T, q) - european_call_price(S0, K, rate, sigma - h, T, q)) / (2 * h)
    function_vega = vega(S0, K, rate, sigma, T, q)
    assert np.isclose(function_vega, derivative_vega, rtol=1e-5)

def test_implied_volatility():
    S0 = 100
    K = 100
    rate = 0.05
    sigma = 0.20
    T = 1.0
    price = european_call_price(S0, K, rate, sigma, T)
    implied_sigma = implied_volatility(price, S0, K, rate, T)
    assert np.isclose(implied_sigma, sigma, atol=1e-6)