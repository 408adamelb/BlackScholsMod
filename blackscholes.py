import yfinance as yf
import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras import layers



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
print(option_chain)

# Filter option chain for puts and calls
puts = option_chain.puts.sort_values(by='lastPrice')
calls = option_chain.calls.sort_values(by='lastPrice')

# Filter for conservative trader
conservative_puts = puts[puts["inTheMoney"] == True].head(5)
conservative_calls = calls[calls["inTheMoney"] == True].tail(5)
r = -0.0430

# Calculate time to expiration in years
time_to_expiry = (pd.to_datetime(expiry) - datetime.now()).days / 365


features = puts[['strike', 'lastPrice', 'impliedVolatility','openInterest', 'volume']].values
target = puts['lastPrice'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss=tf.compat.v1.losses.mean_squared_error)

# Train the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
loss = model.evaluate(X_test_scaled, y_test)
print(f'Mean Squared Error on Test Set: {loss}')

option_chain.puts['predicted_price'] = np.nan

# Loop through each row in the option_chain DataFrame
for index, row in option_chain.puts.iterrows():
    # Create a sample input for the current row
    sample_input = np.array([[row['strike'], row['lastPrice'], row['impliedVolatility'], row['openInterest'], row['volume']]])

    # Standardize the sample input using the scaler
    sample_input_scaled = scaler.transform(sample_input)

    # Make predictions and store the result in the 'predicted_price' column
    option_chain.puts.at[index, 'predicted_price'] = model.predict(sample_input_scaled)[0, 0]

# Display the option_chain DataFrame with the predicted prices
print(option_chain.puts[['strike', 'lastTradeDate', 'lastPrice', 'ask', 'impliedVolatility', 'predicted_price']].to_string(index=False))


# Set risk-free interest rate
print(yf.Ticker(ticker).info.keys())
lastClose = yf.Ticker(ticker).info['regularMarketPreviousClose']
def black_scholes_call(S, X, T, r, sigma):
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S, X, T, r, sigma):
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = X * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def implied_volatility(option_price, S, X, T, r, option_type, initial_guess=0.2):
    def black_scholes_iv(sigma):
        if option_type == 'call':
            return black_scholes_call(S, X, T, r, sigma) - option_price
        elif option_type == 'put':
            return black_scholes_put(S, X, T, r, sigma) - option_price
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

    # Use fsolve to find the implied volatility
    implied_volatility = fsolve(black_scholes_iv, initial_guess)[0]
    return implied_volatility


# Define Black-Scholes formula
def binomial_american_put(S,X,T,r,sigma,n,option_type = 'put'):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Initialize stock prices at maturity
    stock_prices = np.zeros((n + 1, n + 1))
    stock_prices[0, 0] = S
    for j in range(1, n + 1):
        stock_prices[j, 0] = stock_prices[j - 1, 0] * u
        for i in range(1, j + 1):
            stock_prices[j, i] = stock_prices[j - 1, i - 1] * d

    # Calculate option values
    option_values = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        if option_type == 'call':
            option_values[n, i] = max(0, stock_prices[n, i] - X)
        elif option_type == 'put':
            option_values[n, i] = max(0, X - stock_prices[n, i])
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

    # Backward induction
    for j in range(n - 1, -1, -1):
        for i in range(j + 1):
            if option_type == 'call':
                option_values[j, i] = max(stock_prices[j, i] - X, np.exp(-r * dt) * (p * option_values[j + 1, i + 1] + (1 - p) * option_values[j + 1, i]))
            elif option_type == 'put':
                option_values[j, i] = max(X - stock_prices[j, i], np.exp(-r * dt) * (p * option_values[j + 1, i + 1] + (1 - p) * option_values[j + 1, i]))

    return option_values[0, 0]
    





def binomial_american_call(S, X, T, r, sigma, n, option_type='call'):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Initialize stock prices at maturity
    stock_prices = np.zeros((n + 1, n + 1))
    stock_prices[0, 0] = S
    for j in range(1, n + 1):
        stock_prices[j, 0] = stock_prices[j - 1, 0] * u
        for i in range(1, j + 1):
            stock_prices[j, i] = stock_prices[j - 1, i - 1] * d

    # Calculate option values
    option_values = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        if option_type == 'call':
            option_values[n, i] = max(0, stock_prices[n, i] - X)
        elif option_type == 'put':
            option_values[n, i] = max(0, X - stock_prices[n, i])
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

    # Backward induction
    for j in range(n - 1, -1, -1):
        for i in range(j + 1):
            if option_type == 'call':
                option_values[j, i] = max(stock_prices[j, i] - X, np.exp(-r * dt) * (p * option_values[j + 1, i + 1] + (1 - p) * option_values[j + 1, i]))
            elif option_type == 'put':
                option_values[j, i] = max(X - stock_prices[j, i], np.exp(-r * dt) * (p * option_values[j + 1, i + 1] + (1 - p) * option_values[j + 1, i]))

    return option_values[0, 0]
    



# Define a function to estimate implied volatility using the Black-Scholes formula

# Extract the implied volatility column


# Apply the implied volatility function to each option
conservative_puts['impliedVolatility'] = conservative_puts.apply(
    lambda row: implied_volatility(
        row['lastPrice'], lastClose,
        row['strike'], time_to_expiry, r, 'put', initial_guess=0.2
    ),
    axis=1
)
conservative_calls['impliedVolatility'] = conservative_calls.apply(
    lambda row: implied_volatility(
        row['lastPrice'], lastClose,
        row['strike'], time_to_expiry, r, 'call', initial_guess=0.2
    ),
    axis=1
)

# Continue with the rest of your code...
# Continue with the rest of your code...

# Estimate fair price for each option using Black-Scholes formula
conservative_puts['fair_price'] = conservative_puts.apply(lambda row: binomial_american_put(
    row['lastPrice'], row['strike'], time_to_expiry, r, row['impliedVolatility'], 100,'put'
), axis=1)

conservative_calls['fair_price'] = conservative_calls.apply(lambda row: binomial_american_call(
    row['lastPrice'], row['strike'], time_to_expiry, r, row['impliedVolatility'],100, 'call'
), axis=1)

# Calculate expected return for each option
conservative_puts['expected_return'] = (conservative_puts['fair_price'] - conservative_puts['lastPrice']) / \
                                       conservative_puts['lastPrice']

conservative_calls['expected_return'] = (conservative_calls['fair_price'] - conservative_calls['lastPrice']) / \
                                        conservative_calls['lastPrice']

# Rank options by expected return and suggest top 3 put and call
suggested_puts = conservative_puts.sort_values('expected_return', ascending=False).head(3)
suggested_calls = conservative_calls.sort_values('expected_return', ascending=False).head(3)

# Print suggested options and stock price
market_price = lastClose
print()
print("Suggested puts:")
print(suggested_puts[['strike', 'lastTradeDate', 'lastPrice', 'ask', 'impliedVolatility', 'fair_price',
                      'expected_return']].to_string(index=False))
print("\nStock Price:")
print(f"The current price of {ticker} is ${market_price:.2f}")

print("\nSuggested calls:")
print(suggested_calls[['strike', 'lastTradeDate', 'lastPrice', 'ask', 'impliedVolatility', 'fair_price',
                       'expected_return']].to_string(index=False))

# Plot stock price and options

fig, ax = plt.subplots()

# Plot stock price
stock_data['Adj Close'].plot(ax=ax, label=ticker)

# Plot put options
for i, row in suggested_puts.iterrows():
    ax.axhline(y=row['strike'], color='r', linestyle='-')

# Plot call options
for i, row in suggested_calls.iterrows():
    ax.axhline(y=row['strike'], color='g', linestyle='-')

plt.legend()
plt.title(f'{ticker} Options for {expiry}')
plt.xlabel('Date')
plt.ylabel('Price')

plt.savefig("plot.png")
