import numpy as np
from pricing.heston_cf import *
from pricing.heston import *

def test_characteristic_function_at_zero():
    phi = heston_characteristic_function( u=0.0, S0=100, v0=0.04, rate=0.05, dividend_yield=0.0, kappa=2.0, theta=0.04, ksi=0.5, rho=-0.7, T=1.0)
    assert np.isclose(phi, 1.0)

def test_heston_price_is_positive():
    call = price_heston_call(S0=100, K=100, v0=0.04, rate=0.05, dividend_yield=0.0, kappa=2.0, theta=0.04, ksi=0.5, rho=-0.7, T=1.0)
    assert call > 0
    put = price_heston_put(S0=100, K=100, v0=0.04, rate=0.05, dividend_yield=0.0, kappa=2.0, theta=0.04, ksi=0.5, rho=-0.7, T=1.0)
    assert put > 0
    lefthand = call - put
    righthand = 100 * np.exp(-0 * 1) - 100* np.exp(-0.05 * 1)
    assert np.isclose(lefthand, righthand, rtol=1e-8)

def test_heston_cf_matches_monte_carlo():
    S0 = 100
    K = 100
    v0 = 0.04
    rate = 0.05
    q = 0.0
    kappa = 2.0
    theta = 0.04
    ksi = 0.5
    rho = -0.7
    T = 1.0
    cf_price = price_heston_call(S0=S0, K=K, v0=v0, rate=rate, dividend_yield=q, kappa=kappa, theta=theta, ksi=ksi, rho=rho, T=T)
    mc_price, mc_error = price_heston_call_mc(S0=S0, K=K, v0=v0, rate=rate, dividend_yield=q, kappa=kappa, theta=theta,
        ksi=ksi, rho=rho, T=T, n_steps=252, n_paths=100_000, seed=42)
    assert abs(cf_price - mc_price) < 4 * mc_error