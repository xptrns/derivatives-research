from pricing.blackscholes import *
from pricing.montecarlo import *

def test_monte_carlo_call_black_scholes():
    S0 = 100
    K = 100
    rate = 0.05
    sigma = 0.20
    T = 1.0

    bs_price = european_call_price(S0, K, rate, sigma, T)
    mc_price, standard_error = price_european_call_mc(S0, K, rate, sigma, T, n_paths=500_000, seed=42)
    assert abs(mc_price - bs_price) < 4 * standard_error

def test_monte_carlo_put_black_scholes():
    S0 = 100
    K = 100
    rate = 0.05
    sigma = 0.20
    T = 1.0

    bs_price = european_put_price(S0, K, rate, sigma, T)
    mc_price, standard_error = price_european_put_mc(S0, K, rate, sigma, T, n_paths=500_000, seed=42)
    assert abs(mc_price - bs_price) < 4 * standard_error