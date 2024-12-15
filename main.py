# main.py

import warnings
from data_loader import load_data
from indicators import (
    calculate_rpm, calculate_macd, calculate_rsi, calculate_cci,
    calculate_supertrend, calculate_stochastic, calculate_bbo
)
from signals import generate_signals
from backtesting import backtest
from utils import plot_results
from optimization import optimize_parameters
from config_loader import load_config

warnings.filterwarnings('ignore')


def apply_best_parameters(best_params, config):
    dataframe = load_data(config['data']['file_path'])
    dataframe = calculate_rpm(dataframe, src='close',
                              period=int(best_params.get('rpm_period', config['parameters']['rpm_period'])))
    dataframe = calculate_macd(
        dataframe,
        src='close',
        fast=int(best_params.get('macd_fast', config['parameters']['macd_fast'])),
        slow=int(best_params.get('macd_slow', config['parameters']['macd_slow'])),
        signal=int(best_params.get('macd_signal', config['parameters']['macd_signal'])),
        macd_signal_selector='MACD'
    )
    dataframe = calculate_rsi(dataframe, src='close',
                              period=int(best_params.get('rsi_period', config['parameters']['rsi_period'])))
    dataframe = calculate_cci(dataframe, src='close',
                              period=int(best_params.get('cci_period', config['parameters']['cci_period'])))
    dataframe = calculate_supertrend(
        dataframe,
        period=int(best_params.get('supertrend_period', config['parameters']['supertrend_period'])),
        multiplier=float(best_params.get('supertrend_multiplier', config['parameters']['supertrend_multiplier']))
    )
    dataframe = calculate_stochastic(
        dataframe,
        k_period=int(best_params.get('stochastic_k_period', config['parameters']['stochastic_k_period'])),
        d_period=int(best_params.get('stochastic_d_period', config['parameters']['stochastic_d_period'])),
        k_smoothing=int(config['parameters']['stochastic_k_smoothing'])
    )
    dataframe = calculate_bbo(dataframe,
                              length_difference=int(best_params.get('bbo_period', config['parameters']['bbo_period'])))
    dataframe = generate_signals(dataframe)
    dataframe.dropna(inplace=True)
    results = backtest(
        dataframe,
        initial_balance=config['backtest']['initial_balance'],
        fee_rate=config['backtest']['fee_rate']
    )
    return dataframe, results


if __name__ == "__main__":
    # Load Configuration
    config = load_config()

    # Optimize Parameters
    best_params = optimize_parameters(config)
    print("Best Parameters:", best_params)

    # Apply Best Parameters
    df, results = apply_best_parameters(best_params, config)

    # Save the DataFrame with signals and indicators
    df.to_csv('signals_with_indicators.csv')

    # Access and save the trade history
    trade_history = results['Trade History']
    trade_history.to_csv('trade_history.csv', index=False)

    # Display Results
    print("Optimized Backtest Results:")
    for key, value in results.items():
        if key != 'Trade History':
            print(f"{key}: {value}")

    # Optionally, display the trade history
    print("\nTrade History:")
    print(trade_history)

    # Plot Results
    plot_results(df)