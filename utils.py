# utils.py

import matplotlib.pyplot as plt

def plot_results(dataframe):
    plt.figure(figsize=(14, 7))
    plt.plot(dataframe.index, dataframe['close'], label='Close Price', alpha=0.5)
    buy_signals = dataframe[dataframe['Signal'] == 1]
    sell_signals = dataframe[dataframe['Signal'] == -1]
    plt.scatter(buy_signals.index, buy_signals['close'], color='green', label='Buy Signal', marker='^')
    plt.scatter(sell_signals.index, sell_signals['close'], color='red', label='Sell Signal', marker='v')
    plt.title('Trade Signals on Price Chart')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
