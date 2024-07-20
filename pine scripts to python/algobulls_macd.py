import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta

# Load stock data
ticker = 'MSFT'  # Microsoft as an example; replace with your desired stock
data = yf.download(ticker, start='2023-01-01', end='2024-01-01')

# Define parameters
fast_length = 12
slow_length = 26
signal_smoothing = 9
neutral_zone_threshold = 0.05

# Resampling function to simulate request.security
def resample_data(data, timeframe):
    return data.resample(timeframe).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})

# Define timeframes
timeframes = ['2D', '1D', '4H', '1H', '30T', '15T']

# Calculate MACD for different timeframes
macd_dfs = []
for tf in timeframes:
    resampled_data = resample_data(data, tf)
    macd = ta.trend.MACD(resampled_data['Close'], window_slow=slow_length, window_fast=fast_length, window_sign=signal_smoothing)
    resampled_data['macd_line'] = macd.macd()
    resampled_data['signal_line'] = macd.macd_signal()
    resampled_data['macd_diff'] = resampled_data['macd_line'] - resampled_data['signal_line']
    resampled_data['is_neutral'] = resampled_data['macd_diff'].abs() < neutral_zone_threshold
    macd_dfs.append(resampled_data)

# Combine MACD lines from all timeframes
combined_macd = sum([df['macd_line'].reindex(macd_dfs[0].index, method='ffill') for df in macd_dfs]) / len(timeframes)
combined_signal = sum([df['signal_line'].reindex(macd_dfs[0].index, method='ffill') for df in macd_dfs]) / len(timeframes)

# Calculate histogram
hist = combined_macd - combined_signal

# Determine position
position = np.where((combined_macd > combined_signal) & (~macd_dfs[0]['is_neutral'].reindex(macd_dfs[0].index, method='ffill')), 'Long', 
                    np.where((combined_macd < combined_signal) & (~macd_dfs[0]['is_neutral'].reindex(macd_dfs[0].index, method='ffill')), 'Short', 'Neutral'))

# Plot the results
plt.figure(figsize=(14, 8))

# Plot Combined MACD and Signal Line
plt.plot(combined_macd.index, combined_macd, label='Combined MACD Line', color='aqua', linewidth=1.5)
plt.plot(combined_signal.index, combined_signal, label='Combined Signal Line', color='fuchsia', linewidth=1.5)

# Plot MACD Histogram
plt.bar(hist.index, hist, color=np.where(hist >= 0, 'teal', 'purple'), alpha=0.5, label='MACD Histogram')

# Add horizontal lines for neutral zone and threshold
plt.axhline(0, color='grey', linestyle='--', linewidth=1, label='Zero Line')
plt.axhline(neutral_zone_threshold, color='lightgreen', linestyle='--', linewidth=1, label='Neutral Zone Threshold')
plt.axhline(-neutral_zone_threshold, color='lightcoral', linestyle='--', linewidth=1, label='Negative Neutral Zone Threshold')

# Adding titles and labels
plt.title(f'Multi Timeframe MACD for {ticker}', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('MACD Value', fontsize=14)
plt.legend()
plt.grid(True)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show plot
plt.tight_layout()
plt.show()

# Print table summary
summary_table = pd.DataFrame({
    'Timeframe': timeframes,
    'MACD Line': [df['macd_line'].dropna().iloc[-1] for df in macd_dfs],
    'Signal Line': [df['signal_line'].dropna().iloc[-1] for df in macd_dfs],
    'Position': position[-1]
})
print(summary_table)
