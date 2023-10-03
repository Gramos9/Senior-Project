import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Load the data
data = pd.read_csv('/Users/christianrobertson/Desktop/Senior project CSv/AMZN.csv')
closing_prices = data['Close'].values
closing_prices = closing_prices.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(closing_prices)


# Define function to create data structure for LSTM
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# Split data into train and test sets
training_size = int(len(scaled_data) * 0.65)
test_size = len(scaled_data) - training_size
train_data, test_data = scaled_data[0:training_size, :], scaled_data[training_size:len(scaled_data), :1]

# Create datasets for training and testing
time_step = 2
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape data to be [samples, time steps, features] (required for LSTM)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
lstm = Sequential()
lstm.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm.add(LSTM(50))
lstm.add(Dense(1))
lstm.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
lstm.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# Make predictions
train_predict = lstm.predict(X_train)
test_predict = lstm.predict(X_test)

# Transform predictions back to original scale
train_predict = scaler.inverse_transform(train_predict)
y_train = scaler.inverse_transform([y_train])
test_predict = scaler.inverse_transform(test_predict)
y_test = scaler.inverse_transform([y_test])

# Calculate RMSE
train_RMSE = np.sqrt(mean_squared_error(y_train[0], train_predict[:, 0]))
test_RMSE = np.sqrt(mean_squared_error(y_test[0], test_predict[:, 0]))
print(f"Train RMSE: {train_RMSE}")
print(f"Test RMSE: {test_RMSE}")

# Forecast future prices
forecast_days = 10
input_data = X[-1]  # Take the last sequence from your data
forecasted_prices = []

for _ in range(forecast_days):
    prediction = lstm.predict(input_data.reshape(1, time_step, 1))
    forecasted_prices.append(prediction[0][0])

    # Shift the sequence left by one position and append the new prediction
    input_data = np.roll(input_data, -1, axis=0)
    input_data[-1] = prediction

# Transform forecasted prices back to original scale
forecasted_prices = scaler.inverse_transform([forecasted_prices])[0]

# Plot actual vs predicted prices
plt.figure(figsize=(14, 5))
plt.plot(closing_prices, label='Actual Prices')
plt.plot(np.arange(len(closing_prices) - len(test_predict), len(closing_prices)), test_predict, color='red',
         label='Predicted Prices')
plt.plot(np.arange(len(closing_prices), len(closing_prices) + len(forecasted_prices)), forecasted_prices, color='green',
         label='Forecasted Prices')
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('AMZN Stock Price Prediction')
plt.legend()
plt.show()