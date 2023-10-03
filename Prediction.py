# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the data
data = pd.read_csv('/Users/christianrobertson/Desktop/Senior project CSv/AMZN.csv')

# Data preprocessing (assuming 'Close' column contains the stock prices)
data['Close'] = MinMaxScaler().fit_transform(data['Close'].values.reshape(-1, 1))

# Define a function to split data into sequences for LSTM
def lstm_split(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

# Splitting the data (assuming using past 10 days to predict the next day)
n_steps = 10
X, y = lstm_split(data['Close'].values, n_steps)

# Train-test split
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape data to fit LSTM input shape
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define the LSTM model
lstm = Sequential()
lstm.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=True))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
lstm.fit(X_train, y_train, epochs=100)

# Predict using the model
predicted_stock_prices = lstm.predict(X_test)

# Evaluate the model
loss = lstm.evaluate(X_test, y_test)
print(f"Model Loss on Test Data: {loss}")