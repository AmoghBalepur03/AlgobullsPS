import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define the linear regression function
def linreg(series, length, offset=0):
    """
    Calculate the linear regression (least squares moving average) of a series.
    
    Args:
    series (pd.Series): The data series to calculate the linear regression on.
    length (int): The length of the linear regression window.
    offset (int): The offset for the linear regression calculation.
    
    Returns:
    pd.Series: The linear regression values.
    """
    # Create the index array for the linear regression calculation
    idx = np.arange(length)
    
    # Function to apply linear regression on a rolling window
    def linear_regression(y):
        if len(y) == length:
            x = idx - np.mean(idx)
            y = y - np.mean(y)
            slope = np.dot(x, y) / np.dot(x, x)
            intercept = np.mean(series) - slope * np.mean(idx)
            return slope * (length - 1 + offset) + intercept
        else:
            return np.nan
    
    # Apply the linear regression function on a rolling window
    return series.rolling(window=length).apply(linear_regression, raw=False)

# Define the ZLSMA calculation function
def calculate_zlsma(data, length=32, offset=0, source='Close'):
    """
    Calculate the Zero Lag Least Squares Moving Average (ZLSMA).
    
    Args:
    data (pd.DataFrame): The input data containing price series.
    length (int): The length of the linear regression window.
    offset (int): The offset for the linear regression calculation.
    source (str): The source column for the price data.
    
    Returns:
    pd.Series: The ZLSMA values.
    """
    # Get the source data
    src = data[source]
    
    # Calculate the first linear regression
    lsma = linreg(src, length, offset)
    
    # Calculate the second linear regression
    lsma2 = linreg(lsma, length, offset)
    
    # Calculate the zero lag
    eq = lsma - lsma2
    
    # Calculate the ZLSMA
    zlsma = lsma + eq
    
    return zlsma

# Download stock data
ticker = 'AAPL'
data = yf.download(ticker, start='2022-01-01', end='2023-01-01')

# Calculate ZLSMA
zlsma = calculate_zlsma(data, length=32, offset=0, source='Close')

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Close Price')
plt.plot(zlsma, label='ZLSMA')
plt.title('Zero Lag Least Squares Moving Average (ZLSMA)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Download Apple's stock data for the year 2022 using yfinance.
# Calculate the ZLSMA with a specified length and offset.
# Plot the Close Price and ZLSMA values using matplotlib.
