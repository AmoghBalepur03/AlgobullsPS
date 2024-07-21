import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

def calculate_envelope(data, window, percent, exponential=False):
    if exponential:
        basis = calculate_ema(data, window)
    else:
        basis = calculate_sma(data, window)
    
    k = percent / 100.0
    upper = basis * (1 + k)
    lower = basis * (1 - k)
    
    return basis, upper, lower

# Download Apple stock data for the past year
ticker = 'AAPL'
df = yf.download(ticker, period='1y', interval='1d')
df.reset_index(inplace=True)  # Ensure 'Date' is a column

# Parameters
len_env = 20
percent_env = 10.0
len_ma = 20

# Calculate indicators
close = df['Close']
basis_env, upper_env, lower_env = calculate_envelope(close, len_env, percent_env, exponential=False)
ma = calculate_sma(close, len_ma)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(df['Date'], close, label='Close', color='black')
plt.plot(df['Date'], basis_env, label='Envelope Basis', color='#FF6D00')
plt.plot(df['Date'], upper_env, label='Envelope Upper', color='#2962FF')
plt.plot(df['Date'], lower_env, label='Envelope Lower', color='#2962FF')
plt.fill_between(df['Date'], lower_env, upper_env, color='blue', alpha=0.3)
plt.plot(df['Date'], ma, label='Moving Average', color='blue', linewidth=2)

plt.title('Envelope and Moving Average on Apple Stock')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()
