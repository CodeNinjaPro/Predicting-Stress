# Import libraries 
import pandas as pd
import os
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# read data from previous steps
df = pd.read_csv('save/preprocessed_data.csv')
# we take partipant 2 
data = df[(df['id'] == 2)]

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM

# create a datetime index
data['datetime'] = pd.to_datetime(data['datetime'], unit='s')
data = data.set_index('datetime')

# split data into train and test sets
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# define ARIMA model
model = ARIMA(train['y'], order=(1, 1, 1))

# fit ARIMA model
model_fit = model.fit()

# make predictions
predictions_arima = model_fit.predict(start=len(train), end=len(train) + len(test)-1, typ='levels')

# calculate root mean squared error for ARIMA
rmse_arima = np.sqrt(mean_squared_error(test['y'], predictions_arima))
print('ARIMA RMSE:', rmse_arima)

# define LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(50, activation='relu', input_shape=(1, 7)))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer='adam', loss='mse')

# reshape data for LSTM model
train_X, train_y = train[['X', 'Y', 'Z', 'EDA', 'TEMP', 'HR', 'respr']].values, train['y'].values
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X, test_y = test[['X', 'Y', 'Z', 'EDA', 'TEMP', 'HR', 'respr']].values, test['y'].values
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# fit LSTM model
model_lstm.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# make predictions
predictions_lstm = model_lstm.predict(test_X)

# reshape predictions for comparison
predictions_lstm = predictions_lstm.reshape(predictions_lstm.shape[0], 1)

# calculate root mean squared error for LSTM
rmse_lstm = np.sqrt(mean_squared_error(test_y, predictions_lstm))
print('LSTM RMSE:', rmse_lstm)
