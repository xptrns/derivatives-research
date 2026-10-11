from pricing.heston_cf import price_heston_call
import numpy as np

def heston_greeks(params, S0, K, T, rate, dividend_yield, price_heston_call):
    """
    Calculate Heston Delta, Gamma and Vega using finite differences.

    Vega is sensitivity to initial variance v0.
    """

    v0, kappa, theta, ksi, rho = params

    # Spot bump
    dS = 1e-3 * S0

    price_up = price_heston_call(S0=S0 + dS, K=K, v0=v0, rate=rate, dividend_yield=dividend_yield, kappa=kappa,
        theta=theta, ksi=ksi, rho=rho, T=T)

    price_down = price_heston_call(S0=S0 - dS, K=K, v0=v0, rate=rate, dividend_yield=dividend_yield, kappa=kappa,
        theta=theta, ksi=ksi, rho=rho, T=T)

    price_mid = price_heston_call(S0=S0, K=K, v0=v0, rate=rate, dividend_yield=dividend_yield, kappa=kappa,
        theta=theta, ksi=ksi, rho=rho, T=T)

    delta = (price_up - price_down) / (2 * dS)

    gamma = (price_up - 2 * price_mid + price_down) / dS**2

    # Initial variance bump for vega
    dv = max(1e-5, 1e-3 * v0)

    price_v_up = price_heston_call(S0=S0, K=K, v0=v0 + dv, rate=rate, dividend_yield=dividend_yield, kappa=kappa,
        theta=theta, ksi=ksi, rho=rho, T=T)

    price_v_down = price_heston_call(S0=S0, K=K, v0=max(v0 - dv, 1e-8), rate=rate, dividend_yield=dividend_yield,
        kappa=kappa, theta=theta, ksi=ksi, rho=rho, T=T)

    # Sensitivity to variance, rather than volatility
    variance_vega = (price_v_up - price_v_down) / (2 * dv)

    # Convert to conventional volatility vega:
    # v0 = sigma0^2
    sigma0 = np.sqrt(v0)

    vega = variance_vega * 2 * sigma0

    return {"delta": delta, "gamma": gamma, "vega": vega}