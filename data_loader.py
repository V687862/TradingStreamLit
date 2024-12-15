# data_loader.py

import pandas as pd
import os
import requests
import logging
from datetime import datetime, timedelta
import pytz
from typing import Optional

# Configure logging
logging.basicConfig(
    filename='data_loader.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load historical data from a CSV file.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Processed DataFrame with required columns and proper data types.

    Raises:
    - ValueError: If required columns are missing.
    - FileNotFoundError: If the file does not exist.
    - Exception: For any other issues during loading.
    """
    logger.debug(f"Attempting to load CSV data from {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
        logger.debug("CSV file loaded successfully.")
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        raise Exception(f"Error reading CSV file: {e}")

    # Ensure required columns are present
    required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logger.error(f"Missing columns in data: {missing_columns}")
        raise ValueError(f"Missing columns in data: {missing_columns}")

    try:
        # Convert 'time' column to datetime and set as index
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        df.set_index('time', inplace=True)
        logger.debug("'time' column converted to datetime and set as index.")

        # Convert price and volume columns to float
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        df[price_columns] = df[price_columns].astype(float)
        logger.debug("Price and volume columns converted to float.")
    except Exception as e:
        logger.error(f"Error processing data types: {e}")
        raise Exception(f"Error processing data types: {e}")

    return df


def fetch_live_data_coinbase(product_id: str, granularity: int = 300) -> pd.DataFrame:
    """
    Fetch live historical data from Coinbase Pro API.

    Parameters:
    - product_id (str): The trading pair (e.g., 'BTC-USD').
    - granularity (int): Candlestick granularity in seconds (300 for 5 minutes).

    Returns:
    - pd.DataFrame: DataFrame containing historical data.

    Raises:
    - Exception: If the API request fails or data is invalid.
    """
    logger.debug(f"Fetching live data for {product_id} with granularity {granularity} seconds.")
    url = f"https://api.pro.coinbase.com/products/{product_id}/candles"

    # Define the time range (e.g., last 30 days)
    end = datetime.utcnow()
    start = end - timedelta(days=30)
    params = {
        'start': start.isoformat(),
        'end': end.isoformat(),
        'granularity': granularity
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        logger.debug(f"Received {len(data)} data points from Coinbase Pro API.")
    except requests.exceptions.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
        raise Exception(f"HTTP error occurred: {http_err}")
    except Exception as err:
        logger.error(f"Error fetching live data: {err}")
        raise Exception(f"Error fetching live data: {err}")

    if not data:
        logger.warning("No data received from Coinbase Pro API.")
        return pd.DataFrame()

    # Coinbase Pro API returns data in [time, low, high, open, close, volume]
    df = pd.DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])

    try:
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.sort_values('time').reset_index(drop=True)
        df.set_index('time', inplace=True)
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        df[price_columns] = df[price_columns].astype(float)
        logger.debug("Live data processed successfully.")
    except Exception as e:
        logger.error(f"Error processing live data: {e}")
        raise Exception(f"Error processing live data: {e}")

    return df


def resample_data(df: pd.DataFrame, resample_period: str) -> pd.DataFrame:
    """
    Resample the DataFrame to a different time period.

    Parameters:
    - df (pd.DataFrame): Original DataFrame with datetime index.
    - resample_period (str): Resampling frequency (e.g., '5T' for 5 minutes, '1H' for 1 hour, '1D' for 1 day).

    Returns:
    - pd.DataFrame: Resampled DataFrame.

    Raises:
    - ValueError: If the resample_period is invalid.
    - Exception: For any other issues during resampling.
    """
    logger.debug(f"Resampling data to period: {resample_period}")
    try:
        resampled_df = df.resample(resample_period).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        resampled_df.dropna(inplace=True)  # Drop periods with no data
        logger.debug("Data resampled successfully.")
    except Exception as e:
        logger.error(f"Error resampling data: {e}")
        raise Exception(f"Error resampling data: {e}")

    return resampled_df


def load_data(file_type: str,
              product_id: Optional[str] = None,
              resample_period: Optional[str] = None,
              file_path: Optional[str] = None,
              granularity: int = 300) -> pd.DataFrame:
    """
    Load data based on the specified file type.

    Parameters:
    - file_type (str): Type of data source ("Upload CSV" or "Live Data").
    - product_id (str, optional): Trading pair (e.g., 'BTC-USD') for live data.
    - resample_period (str, optional): Resampling frequency (e.g., '5T', '1H', '1D') for live data.
    - file_path (str, optional): Path to the CSV file for uploaded data.
    - granularity (int, optional): Granularity in seconds for live data (default is 300 seconds).

    Returns:
    - pd.DataFrame: Loaded and processed DataFrame.

    Raises:
    - ValueError: If required parameters are missing or invalid.
    - Exception: For any issues during data loading.
    """
    logger.debug(
        f"Loading data with file_type: {file_type}, product_id: {product_id}, resample_period: {resample_period}, file_path: {file_path}")

    if file_type == "Upload CSV":
        if not file_path:
            logger.error("file_path must be provided for 'Upload CSV' data_type.")
            raise ValueError("file_path must be provided for 'Upload CSV' data_type.")
        return load_csv_data(file_path)

    elif file_type == "Live Data":
        if not product_id:
            logger.error("product_id must be provided for 'Live Data' data_type.")
            raise ValueError("product_id must be provided for 'Live Data' data_type.")
        if not resample_period:
            logger.error("resample_period must be provided for 'Live Data' data_type.")
            raise ValueError("resample_period must be provided for 'Live Data' data_type.")

        # Fetch live data
        df = fetch_live_data_coinbase(product_id=product_id, granularity=granularity)
        if df.empty:
            logger.warning("No live data fetched.")
            return df

        # Resample if necessary
        if resample_period.lower() != '1d' and resample_period.lower() != '1h' and resample_period.lower() != '5T':
            logger.warning(f"Unsupported resample_period '{resample_period}'. Proceeding without resampling.")
            return df

        resampled_df = resample_data(df, resample_period)
        return resampled_df

    else:
        logger.error(f"Unsupported file_type: {file_type}")
        raise ValueError(f"Unsupported file_type: {file_type}")


def save_resampled_data(df: pd.DataFrame, product_id: str, resample_period: str,
                        directory: str = "data/historical_data/"):
    """
    Save resampled data to a CSV file.

    Parameters:
    - df (pd.DataFrame): DataFrame to save.
    - product_id (str): Trading pair (e.g., 'BTC-USD').
    - resample_period (str): Resampling frequency (e.g., '5T', '1H', '1D').
    - directory (str, optional): Directory to save the CSV files.

    Raises:
    - Exception: If there's an error during saving.
    """
    logger.debug(f"Saving resampled data for {product_id} at {resample_period} to directory {directory}")
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            logger.debug(f"Created directory {directory}")
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {e}")
            raise Exception(f"Error creating directory {directory}: {e}")

    file_name = f"{product_id}_{resample_period}_data.csv"
    file_path = os.path.join(directory, file_name)

    try:
        df.to_csv(file_path)
        logger.debug(f"Data saved successfully to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data to {file_path}: {e}")
        raise Exception(f"Error saving data to {file_path}: {e}")
