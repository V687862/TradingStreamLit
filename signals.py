import numpy as np


def generate_signals(dataframe, bullish_threshold=5, bearish_threshold=-5):
    dataframe["Score"] = 0  # Initialize the Score column

    # Assign points based on each indicator
    dataframe["Score"] += np.where(dataframe["RPM"] > 0, 1, -1)
    dataframe["Score"] += np.where(dataframe["MACD"] > 0, 1, -1)
    dataframe["Score"] += np.where(dataframe["RSI"] > 50, 1, -1)
    dataframe["Score"] += np.where(dataframe["%K"] > 50, 1, -1)
    dataframe["Score"] += np.where(dataframe["CCI"] > 0, 1, -1)
    dataframe["Score"] += np.where(dataframe["BearsPower"] > 0, 1, -1)
    dataframe["Score"] += np.where(dataframe["Supertrend"] < dataframe["close"], 1, -1)

    # Remove hardcoded thresholds
    # total_indicators = 7  # This can be kept for reference if needed

    # Generate signals based on the total score
    dataframe["Signal"] = 0
    dataframe.loc[dataframe["Score"] >= bullish_threshold, "Signal"] = 1
    dataframe.loc[dataframe["Score"] <= bearish_threshold, "Signal"] = -1

    return dataframe


def analyze_signal_conditions(dataframe, desired_frequency=5):
    """
    Analyzes the frequency of each condition in the signal generation logic and determines if conditions are too strict.

    Parameters:
    dataframe (DataFrame): The DataFrame containing price data and calculated indicators.
    desired_frequency (float): Desired minimum percentage frequency for signal generation.

    Returns:
    None
    """
    # Define each condition separately
    conditions = {
        "RPM_Positive": dataframe["RPM"] > 0,
        "MACD_Positive": dataframe["MACD"] > 0,
        "RSI_Above_50": dataframe["RSI"] > 50,
        "Stochastic_K_Above_50": dataframe["%K"] > 50,
        "CCI_Positive": dataframe["CCI"] > 0,
        "BearsPower_Positive": dataframe["BearsPower"] > 0,
        "Supertrend_Below_Close": dataframe["Supertrend"] < dataframe["close"],
    }

    total_bars = len(dataframe)
    print("**Individual Condition Frequencies (% of total bars):**\n")
    condition_frequencies = {}
    for name, condition in conditions.items():
        true_count = condition.sum()
        percentage = (true_count / total_bars) * 100
        condition_frequencies[name] = percentage
        print(f"{name}: {percentage:.2f}%")

    # Calculate combined long condition frequency
    combined_long = np.logical_and.reduce(list(conditions.values()))
    combined_long_count = combined_long.sum()
    combined_long_percentage = (combined_long_count / total_bars) * 100

    print(f"\n**Combined Long Condition Frequency:** {combined_long_percentage:.4f}%")

    # Determine if combined condition is too strict
    if combined_long_percentage < desired_frequency:
        print(
            "\nThe combined long condition is met less frequently than the desired frequency."
        )
        print("Consider relaxing one or more of the following conditions:")
        # Identify conditions that are the most restrictive
        sorted_conditions = sorted(condition_frequencies.items(), key=lambda x: x[1])
        for name, freq in sorted_conditions:
            if freq < (desired_frequency * 2):  # Adjust the multiplier as needed
                print(f"- {name} is only met {freq:.2f}% of the time.")
    else:
        print("\nThe combined long condition frequency meets the desired threshold.")

    # Optionally, you can return the frequencies for further analysis
    return condition_frequencies, combined_long_percentage
