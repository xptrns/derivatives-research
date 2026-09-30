import numpy as np


def simulate_heston_paths(S0, v0, rate, dividend_yield, kappa, theta, ksi, rho, T, n_steps, n_paths, seed=None):
    """
    Monte Carlo for Heston equations:
    dSt = (r-q)*St*dt + sqrt(vt)*St*dW_S
    dvt = kappa*(theta-vt)*dt + ksi*sqrt(vt)*dW_v
    """
    dt = T / n_steps
    rng = np.random.default_rng(seed)

    S = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))

    S[:, 0] = S0
    v[:, 0] = v0

    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)

        # Correlated random shocks
        dW_S = np.sqrt(dt) * z1
        dW_v = np.sqrt(dt) * (rho * z1 + np.sqrt(1 - rho**2) * z2)

        v_current = v[:, t]
        v[:, t + 1] = np.maximum(v[:, t] + kappa * (theta - v_current) * dt + ksi * np.sqrt(v_current) * dW_v, 0.0)
        v_next = np.maximum(v[:, t + 1], 0.0)
        S[:, t + 1] = (S[:, t] + (rate - dividend_yield) * S[:, t] * dt + np.sqrt(v_current) * S[:, t] * dW_S)
    return S, v

def simulate_heston_terminal_prices(S0, v0, rate, dividend_yield, kappa, theta, ksi, rho, T, n_steps, n_paths, seed=None):
    S, v = simulate_heston_paths(S0=S0, v0=v0, rate=rate, dividend_yield=dividend_yield, kappa=kappa,
        theta=theta, ksi=ksi, rho=rho, T=T, n_steps=n_steps, n_paths=n_paths, seed=seed)
    return S[:, -1], v[:, -1]

def price_heston_call_mc(S0, K, v0, rate, dividend_yield, kappa, theta, ksi, rho, T, n_steps, n_paths, seed=None):
    S_T, _ = simulate_heston_terminal_prices(S0=S0, v0=v0, rate=rate, dividend_yield=dividend_yield, kappa=kappa, \
        theta=theta, ksi=ksi, rho=rho, T=T, n_steps=n_steps, n_paths=n_paths, seed=seed)

    payoffs = np.maximum(S_T - K, 0)
    discounted_payoffs = np.exp(-rate * T) * payoffs
    price = np.mean(discounted_payoffs)

    standard_error = (np.std(discounted_payoffs, ddof=1)/ np.sqrt(n_paths))
    return price, standard_error

def price_heston_put_mc(S0, K, v0, rate, dividend_yield, kappa, theta, ksi, rho, T, n_steps, n_paths, seed=None):
    S_T, _ = simulate_heston_terminal_prices(S0=S0, v0=v0, rate=rate, dividend_yield=dividend_yield, kappa=kappa, \
        theta=theta, ksi=ksi, rho=rho, T=T, n_steps=n_steps, n_paths=n_paths, seed=seed)

    payoffs = np.maximum(K - S_T, 0)
    discounted_payoffs = np.exp(-rate * T) * payoffs
    price = np.mean(discounted_payoffs)

    standard_error = (np.std(discounted_payoffs, ddof=1)/ np.sqrt(n_paths))
    return price, standard_error
