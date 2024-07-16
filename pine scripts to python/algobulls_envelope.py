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

# Load your data into a pandas DataFrame
# Assuming 'data.csv' has 'date' and 'close' columns
df = pd.read_csv('data.csv', parse_dates=['date'])
df.set_index('date', inplace=True)
close = df['close']

# Parameters
len_env = 20
percent_env = 10.0
len_ma = 20

# Calculations
basis_env, upper_env, lower_env = calculate_envelope(close, len_env, percent_env, exponential=False)
ma = calculate_sma(close, len_ma)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(df.index, close, label='Close', color='black')
plt.plot(df.index, basis_env, label='Envelope Basis', color='#FF6D00')
plt.plot(df.index, upper_env, label='Envelope Upper', color='#2962FF')
plt.plot(df.index, lower_env, label='Envelope Lower', color='#2962FF')
plt.fill_between(df.index, lower_env, upper_env, color='blue', alpha=0.3)
plt.plot(df.index, ma, label='Moving Average', color='blue', linewidth=2)

plt.title('Envelope and Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
