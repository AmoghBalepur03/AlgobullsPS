import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fetch historical data
def fetch_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

# Calculate ATR
def calculate_atr(df, period):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

# Chandelier Exit calculation
def chandelier_exit(df, atr_period=22, atr_multiplier=3.0, use_close=True, show_labels=True, highlight_state=True, await_bar_confirmation=True):
    df = df.copy()
    df['ATR'] = calculate_atr(df, atr_period) * atr_multiplier

    if use_close:
        df['LongStop'] = df['Close'].rolling(window=atr_period).max() - df['ATR']
        df['ShortStop'] = df['Close'].rolling(window=atr_period).min() + df['ATR']
    else:
        df['LongStop'] = df['High'].rolling(window=atr_period).max() - df['ATR']
        df['ShortStop'] = df['Low'].rolling(window=atr_period).min() + df['ATR']
    
    df['LongStop'] = np.where(df['Close'].shift() > df['LongStop'].shift(), 
                              np.maximum(df['LongStop'], df['LongStop'].shift()), 
                              df['LongStop'])
    
    df['ShortStop'] = np.where(df['Close'].shift() < df['ShortStop'].shift(), 
                               np.minimum(df['ShortStop'], df['ShortStop'].shift()), 
                               df['ShortStop'])
    
    df['Direction'] = np.where(df['Close'] > df['ShortStop'].shift(), 1, 
                               np.where(df['Close'] < df['LongStop'].shift(), -1, np.nan))
    df['Direction'].ffill(inplace=True)
    
    df['BuySignal'] = (df['Direction'] == 1) & (df['Direction'].shift() == -1)
    df['SellSignal'] = (df['Direction'] == -1) & (df['Direction'].shift() == 1)
    
    if await_bar_confirmation:
        df['BuySignal'] &= df['Close'] > df['Close'].shift()
        df['SellSignal'] &= df['Close'] < df['Close'].shift()
    
    return df

# Main function to plot data
def plot_chandelier_exit(df, atr_period=22, atr_multiplier=3.0):
    df = chandelier_exit(df, atr_period, atr_multiplier)
    
    plt.figure(figsize=(14, 7))
    
    # Plot closing price
    plt.plot(df.index, df['Close'], label='Close Price', color='black', alpha=0.5)
    
    # Plot Long Stop
    plt.plot(df.index, df['LongStop'], label='Long Stop', color='green', linestyle='--', alpha=0.7)
    
    # Plot Short Stop
    plt.plot(df.index, df['ShortStop'], label='Short Stop', color='red', linestyle='--', alpha=0.7)
    
    # Plot Buy Signals
    plt.plot(df.index[df['BuySignal']], df['Close'][df['BuySignal']], '^', markersize=10, color='green', label='Buy Signal', alpha=1)
    
    # Plot Sell Signals
    plt.plot(df.index[df['SellSignal']], df['Close'][df['SellSignal']], 'v', markersize=10, color='red', label='Sell Signal', alpha=1)
    
    plt.title('Chandelier Exit Indicator for AAPL')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# Fetch data for the past year
start_date = '2023-07-20'
end_date = '2024-07-20'
ticker = 'AAPL'
df = fetch_data(ticker, start_date, end_date)

# Plot the Chandelier Exit indicator
plot_chandelier_exit(df)
