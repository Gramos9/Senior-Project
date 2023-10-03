import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Load the Data
data = pd.read_csv('/Users/christianrobertson/Desktop/Senior project CSv/AMZN.csv')
data = data.set_index('Date')  # Assuming Date is one of the columns

# 2. Preprocess the Data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
X, y = [], []

for i in range(1, len(data)):
    X.append(scaled_data[i-1:i])
    y.append(scaled_data[i, 0])  # Assuming 'Close' price is the first column after scaling

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 3. Build the LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train[0].shape)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# 4. Train the Model
model.fit(np.array(X_train), np.array(y_train), epochs=100, batch_size=32)

# 5. Evaluate the Model
loss = model.evaluate(np.array(X_test), np.array(y_test))
print('Model Loss on Test Data:', loss)