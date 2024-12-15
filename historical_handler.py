# historical_handler.py

import os
import pandas as pd
import logging

def resample_data(dataframe, period, output_dir="data/historical_data"):
    """
    Resample data to a specific time period and calculate OHLC and volume.

    Parameters:
        dataframe (pd.DataFrame): The live data to resample.
        period (str): Resampling period (e.g., '5min', '1h', '1D').
        output_dir (str): Directory to save the historical CSV files.
    """
    try:
        # Log DataFrame structure and content for debugging
        logging.debug(f"Resampling DataFrame columns: {dataframe.columns.tolist()}")
        logging.debug(f"Resampling DataFrame head:\n{dataframe.head()}")

        # Check if 'time' is a column; if not, check if it's the index
        if 'time' not in dataframe.columns:
            if 'time' in dataframe.index.names:
                dataframe = dataframe.reset_index()
                logging.debug("Reset index to include 'time' as a column.")
            else:
                raise KeyError("'time' column is missing from the DataFrame.")

        # Ensure 'time' is in datetime format
        dataframe["time"] = pd.to_datetime(dataframe["time"], errors='coerce')

        # Check for any NaT (Not a Time) entries after conversion
        if dataframe["time"].isnull().any():
            missing_times = dataframe[dataframe["time"].isnull()]
            logging.warning(f"Found {len(missing_times)} records with invalid 'time' values. These records will be skipped.")
            dataframe = dataframe.dropna(subset=["time"])

        # Set 'time' as the index
        dataframe.set_index("time", inplace=True)

        # Perform resampling with comprehensive aggregation
        resampled = dataframe.resample(period).agg({
            "price": ["first", "max", "min", "last"],  # OHLC
            "volume": "sum"                            # Total Volume
        })

        # Flatten MultiIndex columns
        resampled.columns = ["open", "high", "low", "close", "volume"]
        resampled.dropna(inplace=True)  # Remove periods with no data
        resampled.reset_index(inplace=True)

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Define the file path
        file_path = os.path.join(output_dir, f"{period}_data.csv")

        # Check if file exists to write headers accordingly
        file_exists = os.path.isfile(file_path)

        # Save to historical file
        resampled.to_csv(file_path, mode="a", header=not file_exists, index=False)

        logging.info(f"Resampled data for period '{period}' saved to '{file_path}'.")

    except KeyError as ke:
        logging.error(f"Key error during resampling: {ke}")
    except Exception as e:
        logging.error(f"Error during resampling data for period '{period}': {e}")
