import numpy as np


def simulate_maturity_prices(S0, rate, sigma, T, n_paths, dividend_yield=0.0, seed=None):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    return S0 * np.exp((rate - dividend_yield - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)

def price_european_call_mc(S0, K, rate, sigma, T, n_paths, dividend_yield=0.0, seed=None):
    S_T = simulate_maturity_prices(S0, rate, sigma, T, n_paths, dividend_yield, seed)
    payoffs = np.maximum(S_T - K, 0)
    discounted_payoffs = np.exp(-rate * T) * payoffs
    price = np.mean(discounted_payoffs)

    standard_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths)
    return price, standard_error

def price_european_put_mc(S0, K, rate, sigma, T, n_paths, dividend_yield=0.0, seed=None):
    S_T = simulate_maturity_prices(S0, rate, sigma, T, n_paths, dividend_yield, seed)
    payoffs = np.maximum(K - S_T, 0)
    discounted_payoffs = np.exp(-rate * T) * payoffs
    price = np.mean(discounted_payoffs)

    standard_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths)
    return price, standard_error
