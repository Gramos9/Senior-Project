import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import os

# Dictionary mapping stock tickers to full names
stock_ticker_to_name = {
    'AMZN': 'Amazon',
    'TSM': 'Taiwan Semiconductor Manufacturing Company',
    'INTC': 'Intel Corporation',
    'MSFT': 'Microsoft Corporation',
    'PSEC': 'Prospect Capital Corporation',
    'AMD': 'Advanced Micro Devices',
    'NVDA': 'NVIDIA Corporation',
    'HRZN': 'Horizon Technology Finance Corporation',
    'META': 'Meta Platforms Inc.',
    'GUSH': 'Direxion Daily S&P Oil & Gas Exp. & Prod. Bull 2X Shares',
    'AAPL': 'Apple Inc.'
}

# Function to get the company logo path
def get_logo_path(stock_ticker):
    logo_folder = "logos"
    logo_file = f"{stock_ticker}_logo.png"
    logo_path = os.path.join(logo_folder, logo_file)
    return logo_path

@st.cache  # Caches the function output to avoid repeated computation
def predict_stock_prices(csv_file_path, n_days, stock_ticker):
    # Load the Data
    data = pd.read_csv(csv_file_path)
    data = data.set_index('Date')

    # Create a Simple Moving Average (SMA) column
    data['SMA'] = data['Close'].rolling(window=5).mean()

    # Drop NA values (because of the moving average)
    data.dropna(inplace=True)

    # Scaler for 'Close' column
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaled = close_scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Columns to be used for prediction
    columns_to_scale = ['Close', 'Volume', 'SMA']
    data_to_scale = data[columns_to_scale].values

    # Scaling data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_to_scale)

    # Create dataset with 60 timesteps and 1 output for the 'Close' price
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i - 60:i])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], len(columns_to_scale)))

    # Split the data (80% training, 20% testing)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build the LSTM Model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], len(columns_to_scale))))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the Model
    model.fit(X_train, y_train, epochs=100, batch_size=32)

    # Evaluate the Model
    loss = model.evaluate(X_test, y_test)
    print('Model Loss on Test Data:', loss)

    # Predict future prices
    predictions = []

    # Initialize last_known_data with the last 60 days of X_test
    last_known_data = X_test[-1]

    for i in range(n_days):
        predicted_price = model.predict(last_known_data.reshape(1, 60, len(columns_to_scale)))
        predictions.append(predicted_price[0, 0])

        # Prepare the new data row with predicted price, 0 volume, and 0 SMA (placeholder values)
        new_row = np.array([[predicted_price[0, 0], 0, 0]])

        # Drop the oldest day and append the new data row
        last_known_data = np.vstack((last_known_data[1:], new_row))

    # Inverse transform to get the real prices using close_scaler
    predicted_prices = close_scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Actual closing prices for visualization
    actual_last_days = data['Close'].values[-n_days:]

    # Plotting
    dates = pd.to_datetime(data.index[-n_days:])  # Last n_days dates for actual data
    future_dates = [dates[-1] + pd.DateOffset(days=i) for i in range(1, n_days+1)]  # Predicted future dates

    return dates, actual_last_days, future_dates, predicted_prices, stock_ticker_to_name.get(stock_ticker, stock_ticker)

# Streamlit app
st.title('Stock Price Prediction App')

# Get list of CSV files in the 'CSV_files' directory
csv_files = [file for file in os.listdir('CSV_files') if file.endswith('.csv')]

# Sidebar to select a CSV file and number of prediction days
selected_csv = st.sidebar.selectbox('Select CSV file', csv_files)
selected_ticker = selected_csv.split('.')[0]  # Extract stock ticker from file name
n_days = st.sidebar.slider('Select Number of Prediction Days', 1, 10, 5)

csv_file_path = os.path.join('CSV_files', selected_csv)

# Run prediction
dates, actual_last_days, future_dates, predicted_prices, company_name = predict_stock_prices(csv_file_path, n_days, selected_ticker)

# Display company logo, name, and description
logo_path = get_logo_path(selected_ticker)
st.image(logo_path, width=100)

# Add company name and short description
st.header(company_name)
# st.write(stock_ticker_to_name[selected_ticker])  # Short description

# Plotting
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(dates, actual_last_days, label='Actual Prices', marker='o')
ax.plot(future_dates, predicted_prices, label='Predicted Prices', marker='x', color='red')
#ax.set_title(f'{company_name} Stock Prices Prediction')
ax.set_xlabel('Date')
ax.set_ylabel('Stock Price')
ax.legend()
ax.grid(True)

# Show the plot
st.pyplot(fig)
