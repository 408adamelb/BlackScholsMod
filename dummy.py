import yfinance as yf
import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime



# Set stock ticker and expiry date
ticker = input("Enter stock ticker: ")
expiry = input("Enter expiry date: ")


# Get stock data for the past year
start_date = pd.to_datetime('today') - pd.DateOffset(years=1)
end_date = pd.to_datetime('today')
stock_data = yf.download(ticker, start=start_date, end=end_date)

# Calculate historical volatility
returns = stock_data['Adj Close'].pct_change().dropna()
volatility = returns.std() * np.sqrt(252)

# Get option chain for stock ticker and expiry date
option_chain = yf.Ticker(ticker).option_chain(expiry)
print(option_chain.puts.T)
