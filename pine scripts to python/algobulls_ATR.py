import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define different moving average functions
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def rma(series, length):
    alpha = 1 / length
    return series.ewm(alpha=alpha, adjust=False).mean()

def sma(series, length):
    return series.rolling(window=length).mean()

def wma(series, length):
    weights = np.arange(1, length + 1)
    return series.rolling(window=length).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

def true_range(high, low, close):
    return pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)

def average_true_range(data, length=14, smoothing='RMA', atr_type='Regular', position_size=1):
    high = data['High']
    low = data['Low']
    close = data['Close']

    tr = true_range(high, low, close)
    
    if smoothing == 'EMA':
        atr = ema(tr, length)
    elif smoothing == 'RMA':
        atr = rma(tr, length)
    elif smoothing == 'SMA':
        atr = sma(tr, length)
    else:
        atr = wma(tr, length)

    if atr_type == 'Regular':
        atr_value = atr
    elif atr_type == 'Percentage':
        atr_value = (atr / close) * 100
    elif atr_type == 'Ticks':
        atr_value = atr / data['TickSize']
    else:  # Currency
        atr_value = round(atr * data['PointValue'] * position_size * 100) / 100

    return atr_value

# Download stock data
ticker = 'AAPL'
data = yf.download(ticker, start='2022-01-01', end='2023-01-01')
data['TickSize'] = 0.01
data['PointValue'] = 1

# Calculate ATR with different smoothing techniques
atr_rma = average_true_range(data, length=14, smoothing='RMA')
atr_ema = average_true_range(data, length=14, smoothing='EMA')
atr_sma = average_true_range(data, length=14, smoothing='SMA')
atr_wma = average_true_range(data, length=14, smoothing='WMA')

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(data.index, atr_rma, label='ATR (RMA)')
plt.plot(data.index, atr_ema, label='ATR (EMA)')
plt.plot(data.index, atr_sma, label='ATR (SMA)')
plt.plot(data.index, atr_wma, label='ATR (WMA)')
plt.title('Average True Range (ATR) with Different Smoothing Techniques')
plt.xlabel('Date')
plt.ylabel('ATR')
plt.legend()
plt.show()

# Download Apple's stock data for the year 2022 using yfinance.
# Calculate ATR using RMA, EMA, SMA, and WMA smoothing techniques.
# Plot the ATR values for each smoothing technique using matplotlib
