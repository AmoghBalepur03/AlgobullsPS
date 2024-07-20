import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate RSI
def calculate_rsi(df, rsi_length=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_length).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate price movement liquidity
def calculate_price_movement_liquidity(df):
    return df['Volume'] / np.abs(df['Close'] - df['Open'])

# Function to calculate EMA
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

# Download stock data
ticker = 'AAPL'
df = yf.download(ticker, start='2022-01-01', end='2023-01-01')

# Calculate RSI
df['RSI'] = calculate_rsi(df, rsi_length=14)

# Calculate price movement liquidity
df['PriceMovementLiquidity'] = calculate_price_movement_liquidity(df)

# Calculate liquidity boundary for outliers
outlier_threshold = 10
df['LiquidityBoundary'] = ema(df['PriceMovementLiquidity'], outlier_threshold) + df['PriceMovementLiquidity'].rolling(window=outlier_threshold).std()

# Identify outliers
df['Outlier'] = df['PriceMovementLiquidity'] > df['LiquidityBoundary']

# Initialize array to store frequency of RSI values (from 0 to 100)
rsi_frequency = np.zeros(101)

# Store RSI value on outliers and update frequency array
for rsi_value in df[df['Outlier']]['RSI']:
    if not np.isnan(rsi_value):
        limited_rsi = min(max(int(round(rsi_value)), 0), 100)
        rsi_frequency[limited_rsi] += 1

# Calculate statistical summary
most_frequent_rsi = np.argmax(rsi_frequency)
least_frequent_rsi = np.where(rsi_frequency > 0, rsi_frequency, np.inf).argmin()
mean_rsi = np.sum(np.arange(101) * rsi_frequency) / np.sum(rsi_frequency)
stddev_rsi = np.sqrt(np.sum((np.arange(101) ** 2) * rsi_frequency) / np.sum(rsi_frequency) - mean_rsi ** 2)
lower_interval = mean_rsi - stddev_rsi
upper_interval = mean_rsi + stddev_rsi

# Print statistical summary
print(f"Most Frequent RSI: {most_frequent_rsi} ({rsi_frequency[most_frequent_rsi]} times)")
print(f"Least Frequent RSI: {least_frequent_rsi} ({rsi_frequency[least_frequent_rsi]} times)")
print(f"Standard Deviation: {stddev_rsi:.2f}")
print(f"68% Confidence Interval: {lower_interval:.0f} - {upper_interval:.0f}")

# Plot RSI with statistical lines
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['RSI'], label='RSI', color='#7E57C2')
plt.axhline(50, color='grey', linestyle='--', linewidth=1)
plt.axhline(upper_interval, color='#787B86', linestyle='-', linewidth=1, label=f'68% Upper Interval: {upper_interval:.0f}')
plt.axhline(lower_interval, color='#787B86', linestyle='-', linewidth=1, label=f'68% Lower Interval: {lower_interval:.0f}')
plt.axhline(most_frequent_rsi, color='#787B86', linestyle='-', linewidth=1, label=f'Most Frequent RSI: {most_frequent_rsi}')
plt.fill_between(df.index, 70, 100, color='green', alpha=0.1)
plt.fill_between(df.index, 0, 30, color='red', alpha=0.1)
plt.legend()
plt.title('RSI Analysis with Statistical Summary')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.show()

# Download Apple's stock data for the year 2022 using yfinance.
# Calculate RSI and upper 68% and lower 68% and most frequent .
# Plot the RSI values using matplotlib
