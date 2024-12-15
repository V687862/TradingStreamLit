# app.py

import logging
import tempfile
import warnings

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backtesting import backtest
from config_loader import load_config

# Import your custom modules
from data_loader import load_data  # Ensure this handles resampled data
from indicators import (
    calculate_rpm,
    calculate_rsi,
    calculate_cci,
    calculate_stochastic,
    calculate_bears_power,
    calculate_dynamic_supertrend,
    calculate_dynamic_macd,
    calculate_atr,
)
from signals import generate_signals

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for comprehensive logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger()

# Load configuration
config = load_config()

# Set page configuration
st.set_page_config(page_title="Personal Trading Bot", layout="wide")

# Title
st.title("Personal Trading Bot Application")

# Sidebar
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Choose an action",
    (
        "Generate Signals",
        "Backtest Strategy",
        "Optimize Parameters with Hyperopt",
        "Dashboard",
    ),
)
use_dynamic_adjustment = st.sidebar.checkbox(
    "Use Dynamic Adjustment Based on Volatility", value=False
)


# Helper functions for Plotly indicator plots
def make_macd_plot(df):
    fig = go.Figure()

    # MACD line
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["macd"], mode="lines", name="MACD", line=dict(color="blue")
        )
    )

    # Signal line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["macd_signal"],
            mode="lines",
            name="Signal Line",
            line=dict(color="orange"),
        )
    )

    # Histogram
    fig.add_trace(
        go.Bar(x=df.index, y=df["macd_hist"], name="Histogram", marker_color="grey")
    )

    fig.update_layout(
        title="MACD Indicator",
        xaxis_title="Time",
        yaxis_title="Value",
        legend=dict(x=0, y=1),
        hovermode="x unified",
    )

    return fig


def make_stochastic_plot(df):
    fig = go.Figure()

    # %K line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["stochastic_k"],
            mode="lines",
            name="%K",
            line=dict(color="purple"),
        )
    )

    # %D line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["stochastic_d"],
            mode="lines",
            name="%D",
            line=dict(color="red"),
        )
    )

    fig.update_layout(
        title="Stochastic Oscillator",
        xaxis_title="Time",
        yaxis_title="Value",
        legend=dict(x=0, y=1),
        hovermode="x unified",
    )

    return fig


# Function to load data (from database or resampled CSVs)
@st.cache_data
def get_data(
    file_type, product_id=None, resample_period=None, file_path=None, granularity=300
):
    try:
        if file_type == "Live Data":
            df = load_data(product_id=product_id, resample_period=resample_period)
        else:
            df = pd.read_csv(file_path, parse_dates=["time"])
        return df
    except Exception as e:
        logger.error(f"Error in get_data: {e}")
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


# Function to validate required columns
def validate_columns(df, required_columns):
    return all(col in df.columns for col in required_columns)


# Main content
if option == "Generate Signals":
    st.header("Generate Trading Signals")

    # File upload or select product and resample period
    st.subheader("Data Selection")
    data_source = st.selectbox("Select Data Source", ("Upload CSV", "Live Data"))

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Choose a CSV file", type=["csv"], key="signal_file"
        )

        if uploaded_file is not None:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                # Read the uploaded data
                df = get_data(file_type="Upload CSV", file_path=tmp_path)

                if df.empty:
                    st.error("Uploaded CSV is empty or invalid.")
                else:
                    # Check required columns
                    required_columns = [
                        "time",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ]
                    if not validate_columns(df, required_columns):
                        st.error(
                            f"The uploaded CSV must contain the following columns: {', '.join(required_columns)}"
                        )
                    else:
                        st.write("Data Preview:")
                        st.dataframe(df.head())

                        # Parameter inputs
                        st.subheader("Adjust Indicator Parameters (Optional)")
                        with st.expander("Indicator Parameters"):
                            params = {
                                "rpm_period": st.number_input(
                                    "RPM Period", min_value=1, max_value=50, value=14
                                ),
                                "macd_fast": st.number_input(
                                    "MACD Fast Period",
                                    min_value=1,
                                    max_value=50,
                                    value=12,
                                ),
                                "macd_slow": st.number_input(
                                    "MACD Slow Period",
                                    min_value=1,
                                    max_value=100,
                                    value=26,
                                ),
                                "macd_signal": st.number_input(
                                    "MACD Signal Period",
                                    min_value=1,
                                    max_value=50,
                                    value=9,
                                ),
                                "rsi_period": st.number_input(
                                    "RSI Period", min_value=1, max_value=50, value=14
                                ),
                                "cci_period": st.number_input(
                                    "CCI Period", min_value=1, max_value=50, value=20
                                ),
                                "bears_power_ema_period": st.number_input(
                                    "Bears Power EMA Period",
                                    min_value=1,
                                    max_value=50,
                                    value=13,
                                ),
                                "supertrend_period": st.number_input(
                                    "Supertrend Period",
                                    min_value=1,
                                    max_value=50,
                                    value=10,
                                ),
                                "supertrend_multiplier": st.number_input(
                                    "Supertrend Multiplier",
                                    min_value=0.1,
                                    max_value=10.0,
                                    value=3.0,
                                    step=0.1,
                                ),
                                "stochastic_k_period": st.number_input(
                                    "Stochastic %K Period",
                                    min_value=1,
                                    max_value=50,
                                    value=14,
                                ),
                                "stochastic_d_period": st.number_input(
                                    "Stochastic %D Period",
                                    min_value=1,
                                    max_value=50,
                                    value=3,
                                ),
                                "stochastic_k_smoothing": st.number_input(
                                    "Stochastic %K Smoothing",
                                    min_value=1,
                                    max_value=10,
                                    value=3,
                                ),
                                "bullish_threshold": st.number_input(
                                    "Bullish Threshold",
                                    min_value=1,
                                    max_value=7,
                                    value=5,
                                ),
                                "bearish_threshold": st.number_input(
                                    "Bearish Threshold",
                                    min_value=-7,
                                    max_value=-1,
                                    value=-5,
                                ),
                            }
                            # Optional thresholds for scoring system

                # Generate signals button
                if st.button("Generate Signals"):
                    with st.spinner("Calculating indicators and generating signals..."):
                        try:
                            # Process data
                            df["time"] = pd.to_datetime(df["time"])
                            df.set_index("time", inplace=True)

                            # Calculate indicators
                            if use_dynamic_adjustment:
                                st.write("Applying Dynamic Indicator Adjustments...")
                                df = calculate_atr(
                                    df, period=14
                                )  # Calculate ATR first for dynamic tuning
                                df = calculate_dynamic_macd(
                                    df,
                                    src="close",
                                    base_fast=int(params["macd_fast"]),
                                    base_slow=int(params["macd_slow"]),
                                    base_signal=int(params["macd_signal"]),
                                )
                                df = calculate_rsi(
                                    df, src="close", period=int(params["rsi_period"])
                                )
                                df = calculate_cci(
                                    df, src="close", period=int(params["cci_period"])
                                )
                                df = calculate_dynamic_supertrend(
                                    df,
                                    base_period=int(params["supertrend_period"]),
                                    multiplier=float(params["supertrend_multiplier"]),
                                )
                                df = calculate_stochastic(
                                    df,
                                    k_period=int(params["stochastic_k_period"]),
                                    d_period=int(params["stochastic_d_period"]),
                                    k_smoothing=int(params["stochastic_k_smoothing"]),
                                )
                                df = calculate_bears_power(
                                    df, ema_period=int(params["bears_power_ema_period"])
                                )
                            else:
                                df = calculate_rpm(
                                    df, src="close", period=int(params["rpm_period"])
                                )
                                df = calculate_dynamic_macd(
                                    df,
                                    src="close",
                                    base_fast=int(params["macd_fast"]),
                                    base_slow=int(params["macd_slow"]),
                                    base_signal=int(params["macd_signal"]),
                                )
                                df = calculate_rsi(
                                    df, src="close", period=int(params["rsi_period"])
                                )
                                df = calculate_cci(
                                    df, src="close", period=int(params["cci_period"])
                                )
                                df = calculate_dynamic_supertrend(
                                    df,
                                    base_period=int(params["supertrend_period"]),
                                    multiplier=float(params["supertrend_multiplier"]),
                                )
                                df = calculate_stochastic(
                                    df,
                                    k_period=int(params["stochastic_k_period"]),
                                    d_period=int(params["stochastic_d_period"]),
                                    k_smoothing=int(params["stochastic_k_smoothing"]),
                                )
                                df = calculate_bears_power(
                                    df, ema_period=int(params["bears_power_ema_period"])
                                )

                            # Generate signals using the scoring system
                            df = generate_signals(
                                df,
                                bullish_threshold=int(params["bullish_threshold"]),
                                bearish_threshold=int(params["bearish_threshold"]),
                            )

                            # Drop NaN values
                            df.dropna(inplace=True)

                            st.success("Signals generated successfully!")

                            # Display signals
                            st.subheader("Signals")
                            st.dataframe(
                                df[["close", "Signal"]].tail(50)
                            )  # Show the last 50 entries

                            # Plot signals on price chart using Plotly for interactivity
                            st.subheader("Price Chart with Signals")
                            fig = go.Figure()

                            # Price line
                            fig.add_trace(
                                go.Scatter(
                                    x=df.index,
                                    y=df["close"],
                                    mode="lines",
                                    name="Close Price",
                                )
                            )

                            # Buy signals
                            buy_signals = df[df["Signal"] == 1]
                            fig.add_trace(
                                go.Scatter(
                                    x=buy_signals.index,
                                    y=buy_signals["close"],
                                    mode="markers",
                                    marker=dict(
                                        symbol="triangle-up", color="green", size=10
                                    ),
                                    name="Buy Signal",
                                )
                            )

                            # Sell signals
                            sell_signals = df[df["Signal"] == -1]
                            fig.add_trace(
                                go.Scatter(
                                    x=sell_signals.index,
                                    y=sell_signals["close"],
                                    mode="markers",
                                    marker=dict(
                                        symbol="triangle-down", color="red", size=10
                                    ),
                                    name="Sell Signal",
                                )
                            )

                            fig.update_layout(
                                title="Trading Signals on Price Chart",
                                xaxis_title="Time",
                                yaxis_title="Price",
                                legend=dict(x=0, y=1, traceorder="normal"),
                                hovermode="x unified",
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Download signals as CSV
                            csv = df[["close", "Signal"]].to_csv().encode()
                            st.download_button(
                                label="Download Signals as CSV",
                                data=csv,
                                file_name="signals.csv",
                                mime="text/csv",
                            )
                        except Exception as e:
                            logger.error(f"Error in Generate Signals: {e}")
                            st.error(f"An error occurred: {e}")

            except Exception as e:
                logger.error(f"Error loading uploaded CSV: {e}")
                st.error(f"An error occurred while loading the CSV: {e}")

    else:
        # Handle "Live Data" option if implemented
        st.info("Live Data functionality is not yet implemented.")

elif option == "Backtest Strategy":
    st.header("Backtest Your Trading Strategy")

    # Data Selection
    st.subheader("Data Selection")
    data_source = st.selectbox(
        "Select Data Source", ("Upload CSV", "Live Data"), key="backtest_data_source"
    )

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Choose a CSV file", type=["csv"], key="backtest_file"
        )
    else:
        # Implement live data selection
        product_id = st.selectbox(
            "Select Product ID",
            ("BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "XYO-USD", "HBAR-USD"),
        )
        resample_period = st.selectbox("Select Resampling Period", ("5min", "1h", "1D"))
        df = get_data(
            file_type="Live Data",
            product_id=product_id,
            resample_period=resample_period,
        )
        if df.empty:
            st.warning("No data available for the selected product and period.")
        else:
            st.write("Data Preview:")
            st.dataframe(df.head())

    if data_source == "Upload CSV" and uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            # Read the uploaded data
            df = get_data(file_type="Upload CSV", file_path=tmp_path)

            if df.empty:
                st.error("Uploaded CSV is empty or invalid.")
            else:
                # Check required columns
                required_columns = ["time", "open", "high", "low", "close", "volume"]
                if not validate_columns(df, required_columns):
                    st.error(
                        f"The uploaded CSV must contain the following columns: {', '.join(required_columns)}"
                    )
                else:
                    st.write("Data Preview:")
                    st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading the uploaded file: {e}")

    elif data_source == "Live Data" and not df.empty:
        st.write("Data Preview:")
        st.dataframe(df.head())

    # Proceed only if df is loaded
    if (
        data_source == "Upload CSV"
        and uploaded_file is not None
        and validate_columns(df, ["time", "open", "high", "low", "close", "volume"])
    ) or (data_source == "Live Data" and not df.empty):
        # Parameter inputs
        st.subheader("Adjust Strategy Parameters")
        with st.expander("Indicator Parameters"):
            params = {
                "rpm_period": st.number_input(
                    "RPM Period", min_value=1, max_value=50, value=14
                ),
                "macd_fast": st.number_input(
                    "MACD Fast Period", min_value=1, max_value=50, value=12
                ),
                "macd_slow": st.number_input(
                    "MACD Slow Period", min_value=1, max_value=100, value=26
                ),
                "macd_signal": st.number_input(
                    "MACD Signal Period", min_value=1, max_value=50, value=9
                ),
                "rsi_period": st.number_input(
                    "RSI Period", min_value=1, max_value=50, value=14
                ),
                "cci_period": st.number_input(
                    "CCI Period", min_value=1, max_value=50, value=20
                ),
                "bears_power_ema_period": st.number_input(
                    "Bears Power EMA Period", min_value=1, max_value=50, value=13
                ),
                "supertrend_period": st.number_input(
                    "Supertrend Period", min_value=1, max_value=50, value=10
                ),
                "supertrend_multiplier": st.number_input(
                    "Supertrend Multiplier",
                    min_value=0.1,
                    max_value=10.0,
                    value=3.0,
                    step=0.1,
                ),
                "stochastic_k_period": st.number_input(
                    "Stochastic %K Period", min_value=1, max_value=50, value=14
                ),
                "stochastic_d_period": st.number_input(
                    "Stochastic %D Period", min_value=1, max_value=50, value=3
                ),
                "stochastic_k_smoothing": st.number_input(
                    "Stochastic %K Smoothing", min_value=1, max_value=10, value=3
                ),
                "bullish_threshold": st.number_input(
                    "Bullish Threshold", min_value=1, max_value=7, value=5
                ),
                "bearish_threshold": st.number_input(
                    "Bearish Threshold", min_value=-7, max_value=-1, value=-5
                ),
            }
            # Optional thresholds for scoring system

        # Backtest button
        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                try:
                    # Process data
                    df["time"] = pd.to_datetime(df["time"])
                    df.set_index("time", inplace=True)

                    # Calculate indicators
                    if use_dynamic_adjustment:
                        st.write("Applying Dynamic Indicator Adjustments...")
                        df = calculate_atr(
                            df, period=14
                        )  # Calculate ATR first for dynamic tuning
                        df = calculate_dynamic_macd(
                            df,
                            src="close",
                            base_fast=int(params["macd_fast"]),
                            base_slow=int(params["macd_slow"]),
                            base_signal=int(params["macd_signal"]),
                        )
                        df = calculate_rsi(
                            df, src="close", period=int(params["rsi_period"])
                        )
                        df = calculate_cci(
                            df, src="close", period=int(params["cci_period"])
                        )
                        df = calculate_dynamic_supertrend(
                            df,
                            base_period=int(params["supertrend_period"]),
                            multiplier=float(params["supertrend_multiplier"]),
                        )
                        df = calculate_stochastic(
                            df,
                            k_period=int(params["stochastic_k_period"]),
                            d_period=int(params["stochastic_d_period"]),
                            k_smoothing=int(params["stochastic_k_smoothing"]),
                        )
                        df = calculate_bears_power(
                            df, ema_period=int(params["bears_power_ema_period"])
                        )
                    else:
                        df = calculate_rpm(
                            df, src="close", period=int(params["rpm_period"])
                        )
                        df = calculate_dynamic_macd(
                            df,
                            src="close",
                            base_fast=int(params["macd_fast"]),
                            base_slow=int(params["macd_slow"]),
                            base_signal=int(params["macd_signal"]),
                        )
                        df = calculate_rsi(
                            df, src="close", period=int(params["rsi_period"])
                        )
                        df = calculate_cci(
                            df, src="close", period=int(params["cci_period"])
                        )
                        df = calculate_dynamic_supertrend(
                            df,
                            base_period=int(params["supertrend_period"]),
                            multiplier=float(params["supertrend_multiplier"]),
                        )
                        df = calculate_stochastic(
                            df,
                            k_period=int(params["stochastic_k_period"]),
                            d_period=int(params["stochastic_d_period"]),
                            k_smoothing=int(params["stochastic_k_smoothing"]),
                        )
                        df = calculate_bears_power(
                            df, ema_period=int(params["bears_power_ema_period"])
                        )

                    # Generate signals using the scoring system
                    df = generate_signals(
                        df,
                        bullish_threshold=int(params["bullish_threshold"]),
                        bearish_threshold=int(params["bearish_threshold"]),
                    )

                    # Drop NaN values
                    df.dropna(inplace=True)

                    # Run backtest
                    results = backtest(df)
                    trade_history = results["Trade History"]

                    # Display results
                    st.success("Backtest completed!")
                    st.header("Backtest Results")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Final Balance", f"${results['Final Balance']:.2f}")
                    col2.metric(
                        "Total Return (%)", f"{results['Total Return (%)']:.2f}%"
                    )
                    col3.metric("Number of Trades", f"{results['Number of Trades']}")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Profit Factor", f"{results['Profit Factor']:.2f}")
                    col2.metric(
                        "Max Drawdown (%)", f"{results['Max Drawdown (%)']:.2f}%"
                    )
                    col3.metric("Sharpe Ratio", f"{results['Sharpe Ratio']:.2f}")

                    # Display trade history
                    st.subheader("Trade History")
                    st.dataframe(trade_history)

                    # Plot signals on price chart using Plotly
                    st.subheader("Price Chart with Buy/Sell Signals")
                    fig = go.Figure()

                    # Price line
                    fig.add_trace(
                        go.Scatter(
                            x=df.index, y=df["close"], mode="lines", name="Close Price"
                        )
                    )

                    # Buy signals
                    buy_signals = df[df["Signal"] == 1]
                    fig.add_trace(
                        go.Scatter(
                            x=buy_signals.index,
                            y=buy_signals["close"],
                            mode="markers",
                            marker=dict(symbol="triangle-up", color="green", size=10),
                            name="Buy Signal",
                        )
                    )

                    # Sell signals
                    sell_signals = df[df["Signal"] == -1]
                    fig.add_trace(
                        go.Scatter(
                            x=sell_signals.index,
                            y=sell_signals["close"],
                            mode="markers",
                            marker=dict(symbol="triangle-down", color="red", size=10),
                            name="Sell Signal",
                        )
                    )

                    fig.update_layout(
                        title="Trading Signals on Price Chart",
                        xaxis_title="Time",
                        yaxis_title="Price",
                        legend=dict(x=0, y=1, traceorder="normal"),
                        hovermode="x unified",
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Plot equity curve
                    st.subheader("Equity Curve")
                    fig_eq = go.Figure()

                    fig_eq.add_trace(
                        go.Scatter(
                            x=results["Equity Curve"].index,
                            y=results["Equity Curve"]["Equity"],
                            mode="lines",
                            name="Equity Curve",
                        )
                    )

                    fig_eq.update_layout(
                        title="Equity Curve Over Time",
                        xaxis_title="Time",
                        yaxis_title="Equity",
                        hovermode="x unified",
                    )

                    st.plotly_chart(fig_eq, use_container_width=True)

                    # Provide a download link for the backtest results
                    csv = trade_history.to_csv().encode()
                    st.download_button(
                        label="Download Trade History as CSV",
                        data=csv,
                        file_name="trade_history.csv",
                        mime="text/csv",
                    )

                    # Provide a download link for the equity curve
                    equity_csv = results["Equity Curve"].to_csv().encode()
                    st.download_button(
                        label="Download Equity Curve as CSV",
                        data=equity_csv,
                        file_name="equity_curve.csv",
                        mime="text/csv",
                    )

                except Exception as e:
                    logger.error(f"Error in Backtest Strategy: {e}")
                    st.error(f"An error occurred: {e}")
    else:
        st.write("Please upload a CSV file or select live data to proceed.")

elif option == "Optimize Parameters with Hyperopt":
    st.header("Optimize Strategy Parameters with Hyperopt")

    # Data Selection
    st.subheader("Data Selection")
    data_source = st.selectbox(
        "Select Data Source", ("Upload CSV", "Live Data"), key="hyperopt_data_source"
    )

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Choose a CSV file", type=["csv"], key="hyperopt_file"
        )
    else:
        # Implement live data selection
        product_id = st.selectbox(
            "Select Product ID",
            ("BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "XYO-USD", "HBAR-USD"),
        )
        resample_period = st.selectbox("Select Resampling Period", ("5min", "1h", "1D"))
        df = get_data(
            file_type="Live Data",
            product_id=product_id,
            resample_period=resample_period,
        )
        if df.empty:
            st.warning("No data available for the selected product and period.")
        else:
            st.write("Data Preview:")
            st.dataframe(df.head())

    # Corrected condition to check columns in df, not in uploaded_file
    if data_source == "Upload CSV" and uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            # Read the uploaded data
            df = get_data(file_type="Upload CSV", file_path=tmp_path)

            if df.empty:
                st.error("Uploaded CSV is empty or invalid.")
            else:
                # Check required columns
                required_columns = ["time", "open", "high", "low", "close", "volume"]
                if not validate_columns(df, required_columns):
                    st.error(
                        f"The uploaded CSV must contain the following columns: {', '.join(required_columns)}"
                    )
                else:
                    st.write("Data Preview:")
                    st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading the uploaded file: {e}")

    elif data_source == "Live Data" and not df.empty:
        st.write("Data Preview:")
        st.dataframe(df.head())

    # Proceed only if df is loaded and columns are correct
    if (
        data_source == "Upload CSV"
        and uploaded_file is not None
        and validate_columns(df, ["time", "open", "high", "low", "close", "volume"])
    ) or (data_source == "Live Data" and not df.empty):
        # Optimization settings
        st.subheader("Optimization Settings")
        max_evals = st.number_input(
            "Maximum Evaluations", min_value=10, max_value=1000, value=50
        )
        st.write("")

        # Optimize button
        if st.button("Run Hyperopt Optimization"):
            with st.spinner("Running optimization..."):
                try:
                    from hyperopt import Trials, STATUS_OK
                    from hyperopt import fmin, tpe, hp

                    # Prepare data
                    df["time"] = pd.to_datetime(df["time"])
                    df.set_index("time", inplace=True)

                    # Define a class to keep track of progress
                    class OptimizationProgress:
                        def __init__(self, total_evals):
                            self.total_evals = total_evals
                            self.current_eval = 0
                            self.best_result = None
                            self.lock = False  # Simplified for Streamlit

                            # Streamlit elements
                            self.progress_bar = st.progress(0)
                            self.status_text = st.empty()
                            self.log_placeholder = st.empty()

                        def update(self, params, result):
                            self.current_eval += 1
                            progress = self.current_eval / self.total_evals
                            self.progress_bar.progress(progress)

                            # Update status text
                            self.status_text.text(
                                f"Running evaluation {self.current_eval}/{self.total_evals}..."
                            )

                            # Update best result
                            if (self.best_result is None) or (
                                result < self.best_result["loss"]
                            ):
                                self.best_result = {"params": params, "loss": result}

                            # Update log
                            log_text = f"**Evaluation {self.current_eval}:**\n"
                            log_text += f"- Parameters: {params}\n"
                            log_text += (
                                f"- Loss (Negative Total Return): {result:.4f}\n"
                            )
                            if self.best_result:
                                log_text += f"- Best Loss So Far: {self.best_result['loss']:.4f}\n"
                            self.log_placeholder.markdown(log_text)

                    # Instantiate the progress tracker
                    progress_tracker = OptimizationProgress(total_evals=int(max_evals))

                    # Define the objective function for optimization
                    def objective(params):
                        data = df.copy()

                        # Unpack parameters
                        rpm_period = int(params["rpm_period"])
                        macd_fast = int(params["macd_fast"])
                        macd_slow = int(params["macd_slow"])
                        macd_signal = int(params["macd_signal"])
                        rsi_period = int(params["rsi_period"])
                        cci_period = int(params["cci_period"])
                        bears_power_ema_period = int(params["bears_power_ema_period"])
                        supertrend_period = int(params["supertrend_period"])
                        supertrend_multiplier = float(params["supertrend_multiplier"])
                        stochastic_k_period = int(params["stochastic_k_period"])
                        stochastic_d_period = int(params["stochastic_d_period"])
                        stochastic_k_smoothing = int(params["stochastic_k_smoothing"])
                        bullish_threshold = int(params["bullish_threshold"])
                        bearish_threshold = int(params["bearish_threshold"])

                        # Calculate indicators
                        data = calculate_rpm(data, src="close", period=rpm_period)
                        data = calculate_dynamic_macd(
                            data,
                            src="close",
                            base_fast=int(params["macd_fast"]),
                            base_slow=int(params["macd_slow"]),
                            base_signal=int(params["macd_signal"]),
                        )
                        data = calculate_rsi(data, src="close", period=rsi_period)
                        data = calculate_cci(data, src="close", period=cci_period)
                        data = calculate_dynamic_supertrend(
                            data,
                            base_period=supertrend_period,
                            multiplier=supertrend_multiplier,
                        )
                        data = calculate_stochastic(
                            data,
                            k_period=stochastic_k_period,
                            d_period=stochastic_d_period,
                            k_smoothing=stochastic_k_smoothing,
                        )
                        data = calculate_bears_power(
                            data, ema_period=bears_power_ema_period
                        )

                        # Generate signals using the scoring system
                        data = generate_signals(
                            data,
                            bullish_threshold=bullish_threshold,
                            bearish_threshold=bearish_threshold,
                        )

                        # Drop NaN values
                        data.dropna(inplace=True)

                        # Check if there are enough data points
                        if data.empty or data["Signal"].nunique() == 1:
                            # No trades possible with these parameters
                            loss = float("inf")
                        else:
                            # Run backtest
                            results = backtest(data)

                            # Objective to minimize (negative total return)
                            loss = -results["Total Return (%)"]

                        # Update progress
                        progress_tracker.update(params, loss)

                        return {"loss": loss, "status": STATUS_OK}

                    # Define the search space
                    search_space = {
                        "rpm_period": hp.quniform("rpm_period", 5, 30, 1),
                        "macd_fast": hp.quniform("macd_fast", 5, 20, 1),
                        "macd_slow": hp.quniform("macd_slow", 20, 50, 1),
                        "macd_signal": hp.quniform("macd_signal", 5, 20, 1),
                        "rsi_period": hp.quniform("rsi_period", 5, 30, 1),
                        "cci_period": hp.quniform("cci_period", 5, 30, 1),
                        "bears_power_ema_period": hp.quniform(
                            "bears_power_ema_period", 5, 30, 1
                        ),
                        "supertrend_period": hp.quniform("supertrend_period", 5, 30, 1),
                        "supertrend_multiplier": hp.uniform(
                            "supertrend_multiplier", 1.0, 5.0
                        ),
                        "stochastic_k_period": hp.quniform(
                            "stochastic_k_period", 5, 30, 1
                        ),
                        "stochastic_d_period": hp.quniform(
                            "stochastic_d_period", 3, 10, 1
                        ),
                        "stochastic_k_smoothing": hp.quniform(
                            "stochastic_k_smoothing", 1, 5, 1
                        ),
                        "bullish_threshold": hp.quniform("bullish_threshold", 4, 7, 1),
                        "bearish_threshold": hp.quniform(
                            "bearish_threshold", -7, -4, 1
                        ),
                    }

                    trials = Trials()

                    # Run optimization
                    best_params = fmin(
                        fn=objective,
                        space=search_space,
                        algo=tpe.suggest,
                        max_evals=int(max_evals),
                        trials=trials,
                        show_progressbar=False,
                    )

                    st.success("Optimization completed!")

                    # Convert parameters from float to int where necessary
                    for key in best_params:
                        if key in ["supertrend_multiplier"]:
                            best_params[key] = float(best_params[key])
                        else:
                            best_params[key] = int(best_params[key])

                    st.subheader("Best Parameters Found:")
                    st.write(best_params)

                    # Run backtest with best parameters
                    data = df.copy()
                    data = calculate_rpm(
                        data, src="close", period=best_params["rpm_period"]
                    )
                    data = calculate_dynamic_macd(
                        data,
                        src="close",
                        base_fast=int(params["macd_fast"]),
                        base_slow=int(params["macd_slow"]),
                        base_signal=int(params["macd_signal"]),
                    )
                    data = calculate_rsi(
                        data, src="close", period=best_params["rsi_period"]
                    )
                    data = calculate_cci(
                        data, src="close", period=best_params["cci_period"]
                    )
                    data = calculate_dynamic_supertrend(
                        data,
                        base_period=best_params["supertrend_period"],
                        multiplier=best_params["supertrend_multiplier"],
                    )
                    data = calculate_stochastic(
                        data,
                        k_period=best_params["stochastic_k_period"],
                        d_period=best_params["stochastic_d_period"],
                        k_smoothing=best_params["stochastic_k_smoothing"],
                    )
                    data = calculate_bears_power(
                        data, ema_period=best_params["bears_power_ema_period"]
                    )

                    # Generate signals
                    data = generate_signals(
                        data,
                        bullish_threshold=best_params["bullish_threshold"],
                        bearish_threshold=best_params["bearish_threshold"],
                    )

                    # Drop NaN values
                    data.dropna(inplace=True)

                    # Run backtest
                    results = backtest(data)
                    trade_history = results["Trade History"]

                    # Display results
                    st.header("Backtest Results with Optimized Parameters")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Final Balance", f"${results['Final Balance']:.2f}")
                    col2.metric(
                        "Total Return (%)", f"{results['Total Return (%)']:.2f}%"
                    )
                    col3.metric("Number of Trades", f"{results['Number of Trades']}")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Profit Factor", f"{results['Profit Factor']:.2f}")
                    col2.metric(
                        "Max Drawdown (%)", f"{results['Max Drawdown (%)']:.2f}%"
                    )
                    col3.metric("Sharpe Ratio", f"{results['Sharpe Ratio']:.2f}")

                    # Display trade history
                    st.subheader("Trade History")
                    st.dataframe(trade_history)

                    # Plot signals on price chart using Plotly
                    st.subheader("Price Chart with Buy/Sell Signals")
                    fig = go.Figure()

                    # Price line
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data["close"],
                            mode="lines",
                            name="Close Price",
                        )
                    )

                    # Buy signals
                    buy_signals = data[data["Signal"] == 1]
                    fig.add_trace(
                        go.Scatter(
                            x=buy_signals.index,
                            y=buy_signals["close"],
                            mode="markers",
                            marker=dict(symbol="triangle-up", color="green", size=10),
                            name="Buy Signal",
                        )
                    )

                    # Sell signals
                    sell_signals = data[data["Signal"] == -1]
                    fig.add_trace(
                        go.Scatter(
                            x=sell_signals.index,
                            y=sell_signals["close"],
                            mode="markers",
                            marker=dict(symbol="triangle-down", color="red", size=10),
                            name="Sell Signal",
                        )
                    )

                    fig.update_layout(
                        title="Trading Signals on Price Chart",
                        xaxis_title="Time",
                        yaxis_title="Price",
                        legend=dict(x=0, y=1, traceorder="normal"),
                        hovermode="x unified",
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Plot equity curve
                    st.subheader("Equity Curve")
                    fig_eq = go.Figure()

                    fig_eq.add_trace(
                        go.Scatter(
                            x=results["Equity Curve"].index,
                            y=results["Equity Curve"]["Equity"],
                            mode="lines",
                            name="Equity Curve",
                        )
                    )

                    fig_eq.update_layout(
                        title="Equity Curve Over Time",
                        xaxis_title="Time",
                        yaxis_title="Equity",
                        hovermode="x unified",
                    )

                    st.plotly_chart(fig_eq, use_container_width=True)

                    # Provide a download link for the backtest results
                    csv = trade_history.to_csv().encode()
                    st.download_button(
                        label="Download Trade History as CSV",
                        data=csv,
                        file_name="trade_history.csv",
                        mime="text/csv",
                    )

                    # Provide a download link for the equity curve
                    equity_csv = results["Equity Curve"].to_csv().encode()
                    st.download_button(
                        label="Download Equity Curve as CSV",
                        data=equity_csv,
                        file_name="equity_curve.csv",
                        mime="text/csv",
                    )

                except Exception as e:
                    logger.error(f"Error in Optimize Parameters with Hyperopt: {e}")
                    st.error(f"An error occurred: {e}")
    else:
        st.write("Please upload a CSV file or select live data to proceed.")

elif option == "Dashboard":
    st.header("Trading Dashboard")

    # Data Selection
    st.subheader("Data Selection")
    data_source = st.selectbox(
        "Select Data Source", ("Upload CSV", "Live Data"), key="dashboard_data_source"
    )

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Choose a CSV file", type=["csv"], key="dashboard_file"
        )
    else:
        # Implement live data selection
        product_id = st.selectbox(
            "Select Product ID",
            ("BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "XYO-USD", "HBAR-USD"),
        )
        resample_period = st.selectbox("Select Resampling Period", ("5min", "1h", "1D"))
        df = get_data(
            file_type="Live Data",
            product_id=product_id,
            resample_period=resample_period,
        )
        if df.empty:
            st.warning("No data available for the selected product and period.")
        else:
            st.write("Data Preview:")
            st.dataframe(df.head())

    if data_source == "Upload CSV" and uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            # Read the uploaded data
            df = get_data(file_type="Upload CSV", file_path=tmp_path)

            if df.empty:
                st.error("Uploaded CSV is empty or invalid.")
            else:
                # Check required columns
                required_columns = ["time", "open", "high", "low", "close", "volume"]
                if not validate_columns(df, required_columns):
                    st.error(
                        f"The uploaded CSV must contain the following columns: {', '.join(required_columns)}"
                    )
                else:
                    st.write("Data Preview:")
                    st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading the uploaded file: {e}")

    elif data_source == "Live Data" and not df.empty:
        st.write("Data Preview:")
        st.dataframe(df.head())

    # Proceed only if df is loaded and columns are correct
    if (
        data_source == "Upload CSV"
        and uploaded_file is not None
        and validate_columns(df, ["time", "open", "high", "low", "close", "volume"])
    ) or (data_source == "Live Data" and not df.empty):
        # Select indicators to display
        st.subheader("Select Indicators to Display")
        indicators = st.multiselect(
            "Choose indicators",
            options=[
                "RPM",
                "MACD",
                "RSI",
                "CCI",
                "Supertrend",
                "Stochastic",
                "Bears Power",
                "ATR",
            ],
            default=["RPM", "MACD", "RSI"],
        )

        # Parameter inputs
        st.subheader("Adjust Indicator Parameters (Optional)")
        with st.expander("Indicator Parameters"):
            params = {
                "rpm_period": st.number_input(
                    "RPM Period", min_value=1, max_value=50, value=14
                ),
                "macd_fast": st.number_input(
                    "MACD Fast Period", min_value=1, max_value=50, value=12
                ),
                "macd_slow": st.number_input(
                    "MACD Slow Period", min_value=1, max_value=100, value=26
                ),
                "macd_signal": st.number_input(
                    "MACD Signal Period", min_value=1, max_value=50, value=9
                ),
                "rsi_period": st.number_input(
                    "RSI Period", min_value=1, max_value=50, value=14
                ),
                "cci_period": st.number_input(
                    "CCI Period", min_value=1, max_value=50, value=20
                ),
                "bears_power_ema_period": st.number_input(
                    "Bears Power EMA Period", min_value=1, max_value=50, value=13
                ),
                "supertrend_period": st.number_input(
                    "Supertrend Period", min_value=1, max_value=50, value=10
                ),
                "supertrend_multiplier": st.number_input(
                    "Supertrend Multiplier",
                    min_value=0.1,
                    max_value=10.0,
                    value=3.0,
                    step=0.1,
                ),
                "stochastic_k_period": st.number_input(
                    "Stochastic %K Period", min_value=1, max_value=50, value=14
                ),
                "stochastic_d_period": st.number_input(
                    "Stochastic %D Period", min_value=1, max_value=50, value=3
                ),
                "stochastic_k_smoothing": st.number_input(
                    "Stochastic %K Smoothing", min_value=1, max_value=10, value=3
                ),
            }

        # Display indicators
        if st.button("Display Indicators"):
            with st.spinner("Calculating indicators..."):
                try:
                    # Process data
                    df["time"] = pd.to_datetime(df["time"])
                    df.set_index("time", inplace=True)

                    # Calculate selected indicators
                    if "RPM" in indicators:
                        df = calculate_rpm(
                            df, src="close", period=int(params["rpm_period"])
                        )
                    if "MACD" in indicators:
                        df = calculate_dynamic_macd(
                            df,
                            src="close",
                            base_fast=int(params["macd_fast"]),
                            base_slow=int(params["macd_slow"]),
                            base_signal=int(params["macd_signal"]),
                        )
                    if "RSI" in indicators:
                        df = calculate_rsi(
                            df, src="close", period=int(params["rsi_period"])
                        )
                    if "CCI" in indicators:
                        df = calculate_cci(
                            df, src="close", period=int(params["cci_period"])
                        )
                    if "Supertrend" in indicators:
                        df = calculate_dynamic_supertrend(
                            df,
                            base_period=int(params["supertrend_period"]),
                            multiplier=float(params["supertrend_multiplier"]),
                        )
                    if "Stochastic" in indicators:
                        df = calculate_stochastic(
                            df,
                            k_period=int(params["stochastic_k_period"]),
                            d_period=int(params["stochastic_d_period"]),
                            k_smoothing=int(params["stochastic_k_smoothing"]),
                        )
                    if "Bears Power" in indicators:
                        df = calculate_bears_power(
                            df, ema_period=int(params["bears_power_ema_period"])
                        )
                    if "ATR" in indicators:
                        df = calculate_atr(df, period=14)  # Example period

                    st.success("Indicators calculated successfully!")

                    # Display indicators
                    st.subheader("Indicator Charts")
                    for indicator in indicators:
                        if indicator == "RPM":
                            fig = px.line(
                                df, x=df.index, y="rpm", title="RPM Indicator"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        elif indicator == "MACD":
                            fig = make_macd_plot(df)
                            st.plotly_chart(fig, use_container_width=True)
                        elif indicator == "RSI":
                            fig = px.line(
                                df, x=df.index, y="rsi", title="RSI Indicator"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        elif indicator == "CCI":
                            fig = px.line(
                                df, x=df.index, y="cci", title="CCI Indicator"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        elif indicator == "Supertrend":
                            fig = px.line(
                                df,
                                x=df.index,
                                y="supertrend",
                                title="Supertrend Indicator",
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        elif indicator == "Stochastic":
                            fig = make_stochastic_plot(df)
                            st.plotly_chart(fig, use_container_width=True)
                        elif indicator == "Bears Power":
                            fig = px.line(
                                df,
                                x=df.index,
                                y="bears_power",
                                title="Bears Power Indicator",
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        elif indicator == "ATR":
                            fig = px.line(
                                df, x=df.index, y="atr", title="ATR Indicator"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    logger.error(f"Error in Dashboard Display Indicators: {e}")
                    st.error(f"An error occurred while calculating indicators: {e}")

else:
    st.write("Invalid option selected.")
