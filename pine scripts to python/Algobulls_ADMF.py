import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Function to calculate different types of moving averages
def f_ma(type, src, length, volume=None):
    if type == "SMA":
        return src.rolling(window=length).mean()
    elif type == "EMA":
        return src.ewm(span=length, adjust=False).mean()
    elif type == "DEMA":
        e = src.ewm(span=length, adjust=False).mean()
        return 2 * e - e.ewm(span=length, adjust=False).mean()
    elif type == "TEMA":
        e = src.ewm(span=length, adjust=False).mean()
        e2 = e.ewm(span=length, adjust=False).mean()
        e3 = e2.ewm(span=length, adjust=False).mean()
        return 3 * (e - e2) + e3
    elif type == "WMA":
        weights = np.arange(1, length + 1)
        return src.rolling(window=length).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    elif type == "VWMA" and volume is not None:
        return (src * volume).rolling(window=length).sum() / volume.rolling(window=length).sum()
    elif type == "SMMA":
        smma = src.rolling(window=length).mean()
        for i in range(length, len(src)):
            smma[i] = (smma[i-1] * (length - 1) + src[i]) / length
        return smma
    elif type == "HMA":
        half_length = int(length / 2)
        sqrt_length = int(np.sqrt(length))
        weights = np.arange(1, length + 1)
        wma_half = src.rolling(window=half_length).apply(lambda prices: np.dot(prices, weights[:half_length]) / weights[:half_length].sum(), raw=True)
        wma_full = src.rolling(window=length).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
        return (2 * wma_half - wma_full).rolling(window=sqrt_length).apply(lambda prices: np.dot(prices, weights[:sqrt_length]) / weights[:sqrt_length].sum(), raw=True)
    return src

# Fetch Apple stock data for the past year
df = yf.download('AAPL', start='2023-07-20', end='2024-07-20')

# Parameters
len_period = 9
price_enable = True
AD_weight = 0.0
ma1Type = "EMA"
ma1Length = 2
ma2Type = "EMA"
ma2Length = 2
normPeriod = 100

# Calculations
df['HLC3'] = (df['High'] + df['Low'] + df['Close']) / 3
df['AD_Ratio'] = df['Close'].diff() / (df[['High', 'Low', 'Close']].apply(lambda x: max(x) - min(x), axis=1))
df['AD_Ratio'] = (1 - AD_weight) * df['AD_Ratio'] + np.sign(df['AD_Ratio']) * AD_weight

vol = df['Volume'] * df['HLC3'] if price_enable else df['Volume']
df['ADMF'] = vol.rolling(window=len_period).mean()

ma1 = f_ma(ma1Type, df['ADMF'], ma1Length)
ma2 = f_ma(ma2Type, ma1, ma2Length)

hist = ma1 - ma2

histMax = hist.rolling(window=normPeriod).max()
histMin = hist.rolling(window=normPeriod).min()
histRange = histMax - histMin
histGrad = 100 * hist / histRange

# Plotting
plt.figure(figsize=(14, 7))

plt.plot(df.index, ma1, label='A/D Money Flow smoothed', color='blue')
plt.plot(df.index, ma2, label='Signal', color='orange')
plt.fill_between(df.index, ma1, ma2, where=(ma1 > ma2), facecolor='green', alpha=0.5)
plt.fill_between(df.index, ma1, ma2, where=(ma1 <= ma2), facecolor='red', alpha=0.5)
plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)

plt.title('Accumulation/Distribution Money Flow v1.1 for AAPL')
plt.legend()
plt.show()
