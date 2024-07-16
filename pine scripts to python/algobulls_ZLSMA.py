import pandas as pd
import numpy as np

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
    # Create the index array for the linear regression calculationZ
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
    src = data[source]
    lsma = linreg(src, length, offset)
    lsma2 = linreg(lsma, length, offset)
    eq = lsma - lsma2
    zlsma = lsma + eq
    return zlsma

# Example usage
data = pd.DataFrame({
    'Close': [105, 110, 112, 115, 118, 120, 122, 123, 125, 128, 130, 132, 133, 135, 137]
})

zlsma = calculate_zlsma(data, length=32, offset=0, source='Close')
print(zlsma)



