# optimization.py

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import pickle
from data_loader import load_data
from indicators import (
    calculate_rpm,
    calculate_macd,
    calculate_rsi,
    calculate_cci,
    calculate_supertrend,
    calculate_stochastic,
    calculate_bears_power,
)
from signals import generate_signals
from backtesting import backtest


def objective(params, config):
    dataframe = load_data(config["data"]["file_path"])

    # Calculate indicators with parameters from Hyperopt
    dataframe = calculate_rpm(dataframe, src="close", period=int(params["rpm_period"]))
    dataframe = calculate_macd(
        dataframe,
        src="close",
        fast=int(params["macd_fast"]),
        slow=int(params["macd_slow"]),
        signal=int(params["macd_signal"]),
        macd_signal_selector="MACD",
    )
    dataframe = calculate_rsi(dataframe, src="close", period=int(params["rsi_period"]))
    dataframe = calculate_cci(dataframe, src="close", period=int(params["cci_period"]))
    dataframe = calculate_supertrend(
        dataframe,
        period=int(params["supertrend_period"]),
        multiplier=float(params["supertrend_multiplier"]),
    )
    dataframe = calculate_stochastic(
        dataframe,
        k_period=int(params["stochastic_k_period"]),
        d_period=int(params["stochastic_d_period"]),
        k_smoothing=int(params["stochastic_k_smoothing"]),
    )
    dataframe = calculate_bears_power(
        dataframe, ema_period=int(params["bears_power_ema_period"])
    )

    # Generate signals using the scoring system
    dataframe = generate_signals(
        dataframe,
        bullish_threshold=int(params["bullish_threshold"]),
        bearish_threshold=int(params["bearish_threshold"]),
    )

    # Drop NaN values
    dataframe.dropna(inplace=True)

    # Run backtest
    results = backtest(
        dataframe,
        initial_balance=config["backtest"]["initial_balance"],
        fee_rate=config["backtest"]["fee_rate"],
    )

    # Check for trades
    if results["Number of Trades"] == 0:
        return {"loss": 1e6, "status": STATUS_OK}

    # Calculate a composite loss
    loss = (
        -results["Total Return (%)"]
        + results["Max Drawdown (%)"] * 0.5
        - results["Sharpe Ratio"] * 10
    )

    return {"loss": loss, "status": STATUS_OK}


def optimize_parameters(config):
    search_space = {
        "rpm_period": hp.quniform("rpm_period", 10, 30, 1),
        "macd_fast": hp.quniform("macd_fast", 8, 15, 1),
        "macd_slow": hp.quniform("macd_slow", 20, 30, 1),
        "macd_signal": hp.quniform("macd_signal", 7, 12, 1),
        "rsi_period": hp.quniform("rsi_period", 10, 20, 1),
        "cci_period": hp.quniform("cci_period", 14, 30, 1),
        "bears_power_ema_period": hp.quniform("bears_power_ema_period", 5, 30, 1),
        "supertrend_period": hp.quniform("supertrend_period", 7, 14, 1),
        "supertrend_multiplier": hp.uniform("supertrend_multiplier", 2, 5),
        "stochastic_k_period": hp.quniform("stochastic_k_period", 10, 20, 1),
        "stochastic_d_period": hp.quniform("stochastic_d_period", 3, 5, 1),
        "stochastic_k_smoothing": hp.quniform("stochastic_k_smoothing", 1, 5, 1),
        "bullish_threshold": hp.quniform("bullish_threshold", 4, 7, 1),
        "bearish_threshold": hp.quniform("bearish_threshold", -7, -4, 1),
    }

    trials = Trials()

    best = fmin(
        fn=lambda params: objective(params, config),
        space=search_space,
        algo=tpe.suggest,
        max_evals=config["optimization"]["max_evals"],
        trials=trials,
    )

    # Save Trials
    with open("hyperopt_trials.pkl", "wb") as f:
        pickle.dump(trials, f)

    return best
