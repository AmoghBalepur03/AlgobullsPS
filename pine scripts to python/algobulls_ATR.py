import pandas as pd
import numpy as np

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

# Example usage
# data = pd.DataFrame({
#     'High': [110, 112, 115, 117, 120],
#     'Low': [100, 105, 108, 111, 113],
#     'Close': [105, 110, 112, 115, 118],
#     'TickSize': 0.01,
#     'PointValue': 1
# })

# atr = average_true_range(data, length=14, smoothing='RMA', atr_type='Regular', position_size=1)
# print(atr)



