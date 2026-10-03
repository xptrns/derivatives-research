import numpy as np


def calibrate_subset(selected_options, market_data, initial_guess, S0, rate, dividend_yield, calibrate_heston):
    """
    Calibrate Heston parameters using only the selected option points.

    Parameters
    ----------
    selected_options : set
        Set of (maturity, strike) tuples.

    market_data : list of dict
        Market observations containing 'maturity', 'strike', and 'price'.

    initial_guess : np.ndarray
        Initial Heston parameter guess.

    S0, rate, dividend_yield : float
        Market/model parameters.

    calibrate_heston : callable
        Heston calibration function.

    Returns
    -------
    result : scipy.optimize.OptimizeResult
        Calibration result.
    """

    subset_data = [observation for observation in market_data if (observation["maturity"], observation["strike"]) in selected_options]
    result = calibrate_heston(initial_guess=initial_guess, market_data=subset_data, S0=S0, rate=rate, dividend_yield=dividend_yield)
    return result


def parameter_rmse(estimated_params, true_params, scales):
    """
    Calculate dimensionless RMSE of estimated Heston parameters.
    """

    estimated_params = np.asarray(estimated_params)
    true_params = np.asarray(true_params)
    scales = np.asarray(scales)

    return np.sqrt(np.mean(((estimated_params - true_params) / scales) ** 2))


def price_surface(params, market_data, S0, rate, dividend_yield, price_heston_call):
    """
    Price every option in a market-data surface using given Heston parameters.

    Returns
    -------
    np.ndarray
        Model prices in the same order as market_data.
    """

    params = np.asarray(params)
    prices = []

    for observation in market_data:
        price = price_heston_call(S0=S0, K=observation["strike"], v0=params[0], rate=rate, dividend_yield=dividend_yield, kappa=params[1],
                        theta=params[2], ksi=params[3], rho=params[4], T=observation["maturity"])
        prices.append(price)
    return np.asarray(prices)


def pricing_rmse(predicted_prices, true_prices):
    """
    Calculate RMSE between model and true option prices.
    """

    predicted_prices = np.asarray(predicted_prices)
    true_prices = np.asarray(true_prices)

    return np.sqrt(np.mean((predicted_prices - true_prices) ** 2))


def evaluate_subset(selected_options, calibration_data, test_data, initial_guess, true_params, scales, S0, rate, dividend_yield, calibrate_heston,  price_heston_call):
    """
    Calibrate using a selected subset and evaluate parameter
    recovery and pricing performance on an independent test surface.
    """

    # Calibration
    result = calibrate_subset(selected_options=selected_options, market_data=calibration_data, initial_guess=initial_guess,
                            S0=S0, rate=rate, dividend_yield=dividend_yield, calibrate_heston=calibrate_heston)

    estimated_params = result.x

    # Parameter error
    parameter_error = parameter_rmse(estimated_params=estimated_params, true_params=true_params, scales=scales)

    # Price independent test surface
    predicted_prices = price_surface(params=estimated_params, market_data=test_data, S0=S0, rate=rate, dividend_yield=dividend_yield, price_heston_call=price_heston_call)

    true_prices = np.asarray([observation["true_price"] for observation in test_data])

    # Test-surface pricing error
    test_rmse = pricing_rmse(predicted_prices, true_prices)

    return {"estimated_params": estimated_params, "parameter_rmse": parameter_error, "test_rmse": test_rmse, "optimizer_result": result}