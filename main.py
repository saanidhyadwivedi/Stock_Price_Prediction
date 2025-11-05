# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import yfinance as yf
from datetime import date

# Data Acquisition
today = date.today()
data = yf.download('AAPL', start='2020-01-01', end=today.strftime("%Y-%m-%d"))

#structuring the data for analysis and display graph using numpy and pandas
data = data.reset_index()
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].dt.strftime('%m/%d/%Y')
data = data.set_index('Date')


# Print the data
print(data)


# Print today's closing price
print(f'Today\'s Closing Price: {data.iloc[-1:]["Close"][0]}')

# print last 10 days closing price including today's closing price
print(f'Last 10 Days Closing Price: {data.iloc[-10:]["Close"][0:]}')


# Preprocess Data
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction and Evaluation for historical data
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')


# Predict the closing price for tomorrow
latest_data = data.iloc[-1:][['Open', 'High', 'Low', 'Volume']]  # Data from the most recent day
predicted_price_tomorrow = model.predict(latest_data)
print(f'Predicted Closing Price for Tomorrow: {predicted_price_tomorrow[0]}')
