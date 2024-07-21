import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fetch Apple stock data for the past year
ticker = 'AAPL'
data = yf.download(ticker, period='1y', interval='1d')

# Helper functions for ADX calculation
def calculate_tr(high, low, close):
    trh = high - low
    trc1 = abs(high - close.shift(1))
    trlc1 = abs(low - close.shift(1))
    tr = pd.DataFrame({'trh': trh, 'trc1': trc1, 'trlc1': trlc1}).max(axis=1)
    return tr

def calculate_dm(high, low):
    upMove = high.diff()
    downMove = low.shift(1) - low
    plusDM = np.where((upMove > downMove) & (upMove > 0), upMove, 0)
    minusDM = np.where((downMove > upMove) & (downMove > 0), downMove, 0)
    return plusDM, minusDM

def calculate_di(plusDM, minusDM, tr14):
    plusDI = (plusDM / tr14) * 100
    minusDI = (minusDM / tr14) * 100
    return plusDI, minusDI

def calculate_adx(plusDI, minusDI, adxlen):
    DX = 100 * (abs(plusDI - minusDI) / (plusDI + minusDI)).rolling(window=adxlen).mean()
    ADX = DX.rolling(window=adxlen).mean()
    return ADX

def calculate_ma(series, length):
    return series.rolling(window=length).mean()

# Inputs
dmlen = 13
adxlen = 13
eadxlen = 13

# Extract relevant columns
high = data['High']
low = data['Low']
close = data['Close']

# True Range Calculation
tr = calculate_tr(high, low, close)

# +DM and -DM Calculation
plusDM, minusDM = calculate_dm(high, low)

# +DI and -DI Calculation
tr14 = tr.rolling(window=dmlen).mean()
plusDI, minusDI = calculate_di(plusDM, minusDM, tr14)

# ADX Calculation
ADX = calculate_adx(plusDI, minusDI, adxlen)

# Enhanced ADX Calculations
spdi = calculate_ma(plusDI, eadxlen)
smdi = calculate_ma(minusDI, eadxlen)

# Plotting
plt.figure(figsize=(14, 10))

# Plot ADX and DI values
plt.subplot(3, 1, 1)
plt.plot(plusDI, label='+DI', color='green')
plt.plot(minusDI, label='-DI', color='red')
plt.plot(ADX, label='ADX', color='blue')
plt.title('Directional Indicators and ADX')
plt.legend()

# Plot Enhanced ADX values
plt.subplot(3, 1, 2)
plt.plot(spdi, label='+DI MA', color='lime')
plt.plot(smdi, label='-DI MA', color='maroon')
plt.plot(plusDI, label='+DI', color='green', alpha=0.5)
plt.plot(minusDI, label='-DI', color='red', alpha=0.5)
plt.plot(ADX, label='ADX', color='blue', alpha=0.5)
plt.title('Enhanced ADX')
plt.legend()

# Plot buy/sell conditions
plt.subplot(3, 1, 3)
plt.plot(data.index, close, label='Close Price', color='black')
plt.title('Apple Stock Price')
plt.legend()

plt.tight_layout()
plt.show()
