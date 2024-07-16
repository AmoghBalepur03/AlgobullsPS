import pandas as pd
import numpy as np

def calculate_atr(df, period):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def chandelier_exit(df, atr_period=22, atr_multiplier=3.0, use_close=True, show_labels=True, highlight_state=True, await_bar_confirmation=True):
    df = df.copy()
    df['ATR'] = calculate_atr(df, atr_period) * atr_multiplier

    if use_close:
        df['LongStop'] = df['Close'].rolling(window=atr_period).max() - df['ATR']
        df['ShortStop'] = df['Close'].rolling(window=atr_period).min() + df['ATR']
    else:
        df['LongStop'] = df['High'].rolling(window=atr_period).max() - df['ATR']
        df['ShortStop'] = df['Low'].rolling(window=atr_period).min() + df['ATR']
    
    df['LongStop'] = np.where(df['Close'].shift() > df['LongStop'].shift(), 
                              np.maximum(df['LongStop'], df['LongStop'].shift()), 
                              df['LongStop'])
    
    df['ShortStop'] = np.where(df['Close'].shift() < df['ShortStop'].shift(), 
                               np.minimum(df['ShortStop'], df['ShortStop'].shift()), 
                               df['ShortStop'])
    
    df['Direction'] = np.where(df['Close'] > df['ShortStop'].shift(), 1, 
                               np.where(df['Close'] < df['LongStop'].shift(), -1, np.nan))
    df['Direction'].ffill(inplace=True)
    
    df['BuySignal'] = (df['Direction'] == 1) & (df['Direction'].shift() == -1)
    df['SellSignal'] = (df['Direction'] == -1) & (df['Direction'].shift() == 1)
    
    if await_bar_confirmation:
        df['BuySignal'] &= df['Close'] > df['Close'].shift()
        df['SellSignal'] &= df['Close'] < df['Close'].shift()
    
    return df
