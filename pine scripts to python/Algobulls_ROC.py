import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class T3Indicator:
    def __init__(self, df, lengthT3=21, factorT3=0.7, modeT3="DOUBLE", roc_filter=1):
        self.df = df
        self.lengthT3 = lengthT3
        self.factorT3 = factorT3
        self.modeT3 = modeT3
        self.roc_filter = roc_filter
        
        # Calculate indicators
        self.df['T3'] = self.calculate_t3(self.df['Close'])
        self.df['ROC'] = self.calculate_roc()
        self.df['ROC_Color'], self.df['ROC_Line'] = self.calculate_roc_color()
        signals = self.generate_trade_signals()
        self.df['Long'] = signals['Long']
        self.df['Short'] = signals['Short']
        self.df['Trend'] = signals['Trend']

    def ema(self, prices, length):
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=length, adjust=False).mean()

    def t3(self, prices, length, factor):
        """Calculate T3 Indicator"""
        ema1 = self.ema(prices, length)
        ema2 = self.ema(ema1, length)
        ema3 = self.ema(ema2, length)
        ema4 = self.ema(ema3, length)
        ema5 = self.ema(ema4, length)
        ema6 = self.ema(ema5, length)
        return (1 + factor) * ema1 - factor * ema2

    def calculate_t3(self, close_prices):
        t3 = self.t3(close_prices, self.lengthT3, self.factorT3)
        if self.modeT3 == "DOUBLE":
            t3 = self.t3(t3, self.lengthT3, self.factorT3)
        elif self.modeT3 == "TRIPLE":
            t3 = self.t3(t3, self.lengthT3, self.factorT3)
            t3 = self.t3(t3, self.lengthT3, self.factorT3)
        return t3

    def calculate_roc(self):
        """Calculate Rate of Change"""
        roc = self.df['Close'].pct_change(periods=self.lengthT3) * 100
        return roc

    def calculate_roc_color(self):
        roc_min = self.df['ROC'].min() / 3
        roc_max = self.df['ROC'].max() / 3
        roc_norm_5 = 10 * (self.df['ROC'] - roc_min) / (roc_max - roc_min) - 5
        roc_norm_10 = 20 * (self.df['ROC'] - roc_min) / (roc_max - roc_min) - 10
        roc_percent = 200 * (self.df['ROC'] - roc_min) / (roc_max - roc_min) - 100
        roc_color = np.where(self.df['ROC'] > 0, np.minimum(np.ceil(roc_norm_10), 10), np.maximum(np.floor(roc_norm_10), -10))
        roc_line = np.minimum(np.ceil(np.abs(roc_norm_5)), 5)
        return roc_color, roc_line

    def generate_trade_signals(self):
        signals = {
            'Long': [],
            'Short': [],
            'Trend': []
        }
        trend = 0
        for idx, row in self.df.iterrows():
            long = False
            short = False
            filtered = False

            if trend > 0 and row['ROC_Line'] <= self.roc_filter:
                trend = 2
                filtered = True
            elif trend < 0 and row['ROC_Line'] <= self.roc_filter:
                trend = -2
                filtered = True

            if row['Close'] > row['T3'] and row['ROC'] > 0 and trend < 1 and not filtered:
                long = True
                trend = 1
            elif row['Close'] < row['T3'] and row['ROC'] < 0 and trend > -1 and not filtered:
                short = True
                trend = -1

            signals['Long'].append(long)
            signals['Short'].append(short)
            signals['Trend'].append(trend)

        return pd.DataFrame(signals, index=self.df.index)

# Download Apple stock data
df = yf.download('AAPL', start='2023-07-20', end='2024-07-20')

# Initialize the T3Indicator with the downloaded data
t3_indicator = T3Indicator(df)

# Plotting
plt.figure(figsize=(14, 10))

# Plot Close Price
plt.subplot(3, 1, 1)
plt.plot(df.index, df['Close'], label='Close Price', color='blue')
plt.title('Apple Stock Price and T3 Indicator')
plt.legend(loc='upper left')

# Plot T3
plt.subplot(3, 1, 2)
plt.plot(df.index, t3_indicator.df['T3'], label='T3 Indicator', color='orange')
plt.title('T3 Indicator')
plt.legend(loc='upper left')

# Plot ROC and T3 Line Thickness
plt.subplot(3, 1, 3)
plt.plot(df.index, t3_indicator.df['ROC'], label='ROC', color='green')
plt.title('Rate of Change (ROC)')
plt.legend(loc='upper left')

# Display the plots
plt.tight_layout()
plt.show()
