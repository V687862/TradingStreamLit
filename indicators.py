# indicators.py

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD as ta_MACD, CCIIndicator
from ta.volatility import AverageTrueRange
import requests


# Custom Exception for Order Book Errors
class OrderBookFetchError(Exception):
    def __init__(self, status_code, message):
        super().__init__(f"OrderBookFetchError: Status Code {status_code} - {message}")


# Helper function to fetch order book data


def fetch_order_book_data(product_id="CLV-USD"):
    url = f"https://api.exchange.coinbase.com/products/{product_id}/book"
    headers = {"Accept": "application/json"}
    params = {"level": 2}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise OrderBookFetchError(response.status_code, response.text)
    data = response.json()
    try:
        bids = pd.DataFrame(
            data["bids"], columns=["price", "size", "num_orders"]
        ).astype(float)
        asks = pd.DataFrame(
            data["asks"], columns=["price", "size", "num_orders"]
        ).astype(float)
    except ValueError as e:
        raise ValueError(f"Data conversion error: {e}")
    return bids, asks


# New Order Book Imbalance Indicator


def calculate_order_book_imbalance(dataframe, product_id="CLV-USD"):
    bids, asks = fetch_order_book_data(product_id)
    if bids.empty or asks.empty:
        raise ValueError("Order book data is empty. Cannot calculate imbalance.")
    total_bid_volume = bids["size"].sum()
    total_ask_volume = asks["size"].sum()
    dataframe["OrderBookImbalance"] = (total_bid_volume - total_ask_volume) / (
        total_bid_volume + total_ask_volume
    )
    return dataframe


# New VWAP Calculation


def calculate_vwap(dataframe):
    if dataframe.empty:
        raise ValueError("Dataframe is empty. Cannot calculate VWAP.")
    vwap_numerator = (dataframe["close"] * dataframe["volume"]).cumsum()
    vwap_denominator = dataframe["volume"].cumsum()
    dataframe["VWAP"] = vwap_numerator / vwap_denominator
    return dataframe


# New Whale Activity Monitor


def calculate_whale_activity(dataframe, recent_trades):
    if recent_trades.empty:
        raise ValueError(
            "Recent trades data is empty. Cannot calculate whale activity."
        )
    dataframe["LargeTrades"] = recent_trades["size"] > recent_trades["size"].quantile(
        0.99
    )
    dataframe["WhaleActivity"] = (
        dataframe["LargeTrades"].rolling(window=10, min_periods=1).sum()
    )
    return dataframe


# Dynamic ATR Calculation for Tuning Indicators


def calculate_atr(dataframe, period=14):
    if dataframe.empty:
        raise ValueError("Dataframe is empty. Cannot calculate ATR.")
    atr_indicator = AverageTrueRange(
        high=dataframe["high"],
        low=dataframe["low"],
        close=dataframe["close"],
        window=period,
    )
    dataframe["ATR"] = atr_indicator.average_true_range()
    return dataframe


# Dynamic Tuning for Supertrend


def calculate_dynamic_supertrend(dataframe, base_period=10, multiplier=3.0):
    dataframe = calculate_atr(dataframe, period=base_period)
    if dataframe.empty:
        raise ValueError(
            "Dataframe is empty after ATR calculation. Cannot calculate Supertrend."
        )
    # Adjust the Supertrend period based on ATR (e.g., increase period during high volatility)
    dynamic_multiplier = multiplier + dataframe["ATR"] / dataframe["ATR"].mean()
    dataframe["TR"] = np.maximum(
        dataframe["high"] - dataframe["low"],
        np.maximum(
            abs(dataframe["high"] - dataframe["close"].shift()),
            abs(dataframe["low"] - dataframe["close"].shift()),
        ),
    )
    dataframe["ATR"] = dataframe["TR"].rolling(window=base_period, min_periods=1).mean()
    dataframe["BasicUpperBand"] = ((dataframe["high"] + dataframe["low"]) / 2) + (
        dynamic_multiplier * dataframe["ATR"]
    )
    dataframe["BasicLowerBand"] = ((dataframe["high"] + dataframe["low"]) / 2) - (
        dynamic_multiplier * dataframe["ATR"]
    )
    dataframe["FinalUpperBand"] = dataframe["BasicUpperBand"].copy()
    dataframe["FinalLowerBand"] = dataframe["BasicLowerBand"].copy()
    dataframe["FinalUpperBand"] = dataframe["FinalUpperBand"].combine_first(
        dataframe["FinalUpperBand"].shift()
    )
    dataframe["FinalLowerBand"] = dataframe["FinalLowerBand"].combine_first(
        dataframe["FinalLowerBand"].shift()
    )
    dataframe["Supertrend"] = np.where(
        dataframe["close"] <= dataframe["FinalUpperBand"],
        dataframe["FinalUpperBand"],
        dataframe["FinalLowerBand"],
    )
    dataframe.drop(
        [
            "TR",
            "ATR",
            "BasicUpperBand",
            "BasicLowerBand",
            "FinalUpperBand",
            "FinalLowerBand",
        ],
        axis=1,
        inplace=True,
    )
    return dataframe


# Dynamic Tuning for MACD


def calculate_dynamic_macd(
    dataframe, src="close", base_fast=12, base_slow=26, base_signal=9
):
    dataframe = calculate_atr(dataframe, period=14)
    if dataframe.empty:
        raise ValueError(
            "Dataframe is empty after ATR calculation. Cannot calculate MACD."
        )
    # Adjust MACD periods based on ATR (e.g., increase signal smoothing during high volatility)
    atr_adjustment_factor = dataframe["ATR"] / dataframe["ATR"].mean()
    fast = max(1, int(base_fast + atr_adjustment_factor.mean()))
    slow = max(1, int(base_slow + atr_adjustment_factor.mean()))
    signal = max(1, int(base_signal + atr_adjustment_factor.mean()))
    macd_indicator = ta_MACD(
        close=dataframe[src], window_slow=slow, window_fast=fast, window_sign=signal
    )
    dataframe["MACD_line"] = macd_indicator.macd()
    dataframe["MACD_signal"] = macd_indicator.macd_signal()
    dataframe["MACD_hist"] = macd_indicator.macd_diff()
    dataframe["MACD"] = dataframe["MACD_line"]
    return dataframe


# Volume Filter for Signal Confirmation


def apply_volume_filter(dataframe, volume_period=20):
    if dataframe.empty:
        raise ValueError("Dataframe is empty. Cannot apply volume filter.")
    dataframe["VolumeSMA"] = (
        dataframe["volume"].rolling(window=volume_period, min_periods=1).mean()
    )
    dataframe["VolumeFilter"] = dataframe["volume"] > dataframe["VolumeSMA"]
    return dataframe


# Original Indicators with Dynamic Tuning Integrated


def calculate_rpm(dataframe, src="close", period=14):
    if len(dataframe) < period:
        raise ValueError(
            "Dataframe length is shorter than the RPM period. Cannot calculate RPM."
        )
    dataframe["PriceChangePercent"] = dataframe[src].pct_change() * 100
    dataframe["RPM"] = dataframe["PriceChangePercent"].rolling(window=period).sum()
    return dataframe


def calculate_rsi(dataframe, src="close", period=14):
    if len(dataframe) < period:
        raise ValueError(
            "Dataframe length is shorter than the RSI period. Cannot calculate RSI."
        )
    rsi_indicator = RSIIndicator(close=dataframe[src], window=period)
    dataframe["RSI"] = rsi_indicator.rsi()
    return dataframe


def calculate_cci(dataframe, src="close", period=20):
    if len(dataframe) < period:
        raise ValueError(
            "Dataframe length is shorter than the CCI period. Cannot calculate CCI."
        )
    cci_indicator = CCIIndicator(
        high=dataframe["high"],
        low=dataframe["low"],
        close=dataframe[src],
        window=period,
    )
    dataframe["CCI"] = cci_indicator.cci()
    return dataframe


def calculate_stochastic(dataframe, k_period=14, d_period=3, k_smoothing=3):
    if len(dataframe) < k_period:
        raise ValueError(
            "Dataframe length is shorter than the Stochastic period. Cannot calculate Stochastic Oscillator."
        )
    lowest_low = dataframe["low"].rolling(window=k_period, min_periods=1).min()
    highest_high = dataframe["high"].rolling(window=k_period, min_periods=1).max()
    dataframe["%K"] = 100 * (
        (dataframe["close"] - lowest_low) / (highest_high - lowest_low)
    )
    if k_smoothing > 1:
        dataframe["%K"] = (
            dataframe["%K"].rolling(window=k_smoothing, min_periods=1).mean()
        )
    dataframe["%D"] = dataframe["%K"].rolling(window=d_period, min_periods=1).mean()
    return dataframe


def f_t3(dataframe, src="close", length=3, volume_factor=0.7):
    if len(dataframe) < length:
        raise ValueError(
            "Dataframe length is shorter than the T3 length. Cannot calculate T3."
        )
    c1 = -(volume_factor**3)
    c2 = 3 * volume_factor**2 + 3 * volume_factor**3
    c3 = -6 * volume_factor**2 - 3 * volume_factor - 3 * volume_factor**3
    c4 = 1 + 3 * volume_factor + volume_factor**3 + 3 * volume_factor**2
    ema1 = dataframe[src].ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()
    ema4 = ema3.ewm(span=length, adjust=False).mean()
    ema5 = ema4.ewm(span=length, adjust=False).mean()
    ema6 = ema5.ewm(span=length, adjust=False).mean()
    t3 = c1 * ema6 + c2 * ema5 + c3 * ema4 + c4 * ema3
    dataframe["T3"] = t3
    return dataframe


def calculate_bears_power(dataframe, ema_period=13):
    """
    Calculates the Bears Power indicator.

    Parameters:
    - dataframe (DataFrame): The DataFrame containing price data.
    - ema_period (int): The period for the EMA used in the calculation.

    Returns:
    - dataframe (DataFrame): The DataFrame with a new 'BearsPower' column.
    """
    dataframe["EMA"] = dataframe["close"].ewm(span=ema_period, adjust=False).mean()
    dataframe["BearsPower"] = dataframe["low"] - dataframe["EMA"]
    return dataframe
