# backtesting.py

import pandas as pd
import numpy as np

def backtest(df, initial_balance=1000, fee_rate=0.001):
    balance = initial_balance
    equity = initial_balance  # Includes current balance and unrealized P&L
    position = 0  # 1 for long, -1 for short, 0 for flat
    position_size = 0
    entry_price = 0
    entry_time = None
    trade_log = []
    trade_history = []  # List to store trade details
    equity_curve = []  # Track equity over time

    for i, row in df.iterrows():
        price = row['close']
        signal = row['Signal']

        # Close position if an opposite signal is generated
        if position != 0 and ((position == 1 and signal == -1) or (position == -1 and signal == 1)):
            exit_price = price
            # Calculate P&L
            gross_profit = (exit_price - entry_price) * position_size * position
            entry_fee = entry_price * position_size * fee_rate
            exit_fee = exit_price * position_size * fee_rate
            net_profit = gross_profit - entry_fee - exit_fee

            balance += net_profit
            trade_log.append(net_profit)
            trade_history.append({
                'Entry Time': entry_time,
                'Exit Time': row.name,
                'Position': 'Long' if position == 1 else 'Short',
                'Entry Price': entry_price,
                'Exit Price': exit_price,
                'Position Size': position_size,
                'Gross Profit': gross_profit,
                'Net Profit': net_profit
            })
            position = 0
            position_size = 0
            entry_price = 0
            entry_time = None

        # Open position if signal is generated and no position is currently open
        if position == 0 and signal != 0:
            position = signal
            entry_price = price
            entry_time = row.name  # Store entry time
            # Calculate position size (Assuming full balance is used)
            position_size = balance / price  # Number of units
            # Deduct entry fee from balance
            entry_fee = price * position_size * fee_rate
            balance -= entry_fee
            # No need to adjust balance further; the position is now open

        # Update equity (balance + unrealized P&L)
        if position != 0:
            unrealized_pnl = (price - entry_price) * position_size * position
            equity = balance + unrealized_pnl
        else:
            equity = balance

        equity_curve.append({'Time': row.name, 'Equity': equity})

    # Close any open positions at the end
    if position != 0:
        exit_price = df.iloc[-1]['close']
        gross_profit = (exit_price - entry_price) * position_size * position
        entry_fee = entry_price * position_size * fee_rate
        exit_fee = exit_price * position_size * fee_rate
        net_profit = gross_profit - exit_fee  # Entry fee was already deducted
        balance += net_profit
        trade_log.append(net_profit)
        trade_history.append({
            'Entry Time': entry_time,
            'Exit Time': df.iloc[-1].name,
            'Position': 'Long' if position == 1 else 'Short',
            'Entry Price': entry_price,
            'Exit Price': exit_price,
            'Position Size': position_size,
            'Gross Profit': gross_profit,
            'Net Profit': net_profit
        })
        position = 0
        position_size = 0
        entry_price = 0
        entry_time = None
        equity = balance

    # Convert trade_history and equity_curve to DataFrames
    trades_df = pd.DataFrame(trade_history)
    equity_df = pd.DataFrame(equity_curve).set_index('Time')

    # Calculate performance metrics
    total_return = (balance - initial_balance) / initial_balance * 100
    profit_factor = calculate_profit_factor(trade_log)
    max_drawdown = calculate_max_drawdown(equity_df['Equity'])
    sharpe_ratio = calculate_sharpe_ratio(trade_log)

    results = {
        'Final Balance': balance,
        'Total Return (%)': total_return,
        'Profit Factor': profit_factor,
        'Number of Trades': len(trade_log),
        'Max Drawdown (%)': max_drawdown,
        'Sharpe Ratio': sharpe_ratio,
        'Trade History': trades_df,
        'Equity Curve': equity_df
    }

    return results

def calculate_profit_factor(trade_log):
    gains = sum(p for p in trade_log if p > 0)
    losses = abs(sum(p for p in trade_log if p < 0))
    if losses == 0:
        return float('inf')
    else:
        return gains / losses

def calculate_max_drawdown(equity_series):
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    max_drawdown = drawdown.min() * 100  # Convert to percentage
    return abs(max_drawdown)

def calculate_sharpe_ratio(trade_log, risk_free_rate=0):
    returns = np.array(trade_log)
    if returns.std() == 0:
        return 0
    sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std()
    return sharpe_ratio * np.sqrt(252)  # Assuming daily returns
