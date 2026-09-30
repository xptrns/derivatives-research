import numpy as np

from pricing.heston import *

def test_heston_initial_conditions():
    S0 = 100
    v0 = 0.04

    S, v = simulate_heston_paths(S0=S0, v0=v0, rate=0.05,dividend_yield=0.0, kappa=2.0, theta=0.04,
        ksi=0.3,rho=-0.7, T=1.0, n_steps=100, n_paths=1_000, seed=42)

    assert np.all(S[:, 0] == S0)
    assert np.all(v[:, 0] == v0)
    assert np.all(v >= 0)

def test_heston_put_call_parity():
    S0 = 100
    K = 105
    rate = 0.05
    q = 0.02
    T = 1.0

    call, call_error = price_heston_call_mc(S0=S0, K=K, v0=0.04, rate=rate, dividend_yield=q, kappa=2.0,
        theta=0.04, ksi=0.3, rho=-0.7, T=T, n_steps=100, n_paths=100_000, seed=42)

    put, put_error = price_heston_put_mc(S0=S0, K=K, v0=0.04, rate=rate, dividend_yield=q, kappa=2.0,
        theta=0.04, ksi=0.3, rho=-0.7, T=T, n_steps=100, n_paths=100_000, seed=42)
    
    lefthand = call - put
    righthand = S0 * np.exp(-q * T) - K * np.exp(-rate * T)

    tolerance = 4 * np.sqrt(call_error**2 + put_error**2)

    assert abs(lefthand - righthand) < tolerance