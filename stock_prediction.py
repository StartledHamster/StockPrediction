# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following (best in a virtual env):
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import os
import plotly.graph_objects as go
import mplfinance as mpf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer



import yfinance as yf
import re
#------------------------------------------------------------------------------
# Load Data
## TO DO:
# 1) Check if data has been saved before. 
# If so, load the saved data
# If not, save the data into a directory
#------------------------------------------------------------------------------
# DATA_SOURCE = "yahoo"


train_data = 0
test_data = 0
COMPANY = 'CBA.AX'
PRICE_VALUE = "Close"

scaled_data = 0
scaler = 0

train_start = '2020-01-01'     # Start date to read
train_end = '2023-08-01'       # End date to read
test_start = '2023-08-02'
test_end = '2024-07-02'

def load_process_dataset(company = COMPANY, start_date = '2020-01-01', end_date = '2024-07-02', split = '2023-08-01', save_path = './data/', scale = True):
    """
    loads data from csv / yf, processes and splits training and testing values

    :param company: company the stock data is from
    :param start_date: starting date of the data
    :param end_date: ending date of the data
    :param split: can be in format of date (YYYY-MM-DD) or ratio (0.#), splits training from testing data
    :param save_path: directory of saved local csv files
    :param scale: true or false, scales data between 0 and 1

    """
    global train_data
    global test_data
    global scaled_data
    global scaler

    #Ensure save directory exists, if not make one
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #file format for full_data = savepath/company_startdate_enddate.csv
    file_path = os.path.join(save_path, f"{company}_{start_date}_{end_date}.csv")

    # If a file for the current parameter data exists, load from the local file
    if os.path.exists(file_path):
        print(f"Loading data from {file_path}")
        full_data = pd.read_csv(file_path, index_col="Date", parse_dates=True)
    else:
        #if no local file exists, download from yf, and create local file for future reference
        print(f"Downloading data for {company} from {start_date} to {end_date}")
            
        # Get the data for the stock AAPL
        full_data = yf.download(company,start_date,end_date)

        full_data.to_csv(file_path)
        print(f"Data saved to {file_path}")


    #This line is used to deal with NaNs; drops all missing values from dataset
    full_data.dropna(inplace=True)

    #------------------------------------------------------------------------------
    # Prepare Data
    ## To do:
    # 1) Check if data has been prepared before. 
    # If so, load the saved data
    # If not, save the data into a directory
    # 2) Use a different price value eg. mid-point of Open & Close
    # 3) Change the Prediction days
    #------------------------------------------------------------------------------
    if (scale):
        scaler = MinMaxScaler(feature_range=(0, 1)) 
    # Note that, by default, feature_range=(0, 1). Thus, if you want a different 
    # feature_range (min,max) then you'll need to specify it here
        scaled_data = scaler.fit_transform(full_data[PRICE_VALUE].values.reshape(-1, 1)) 
    # Flatten and normalise the data
    # First, we reshape a 1D array(n) to 2D array(n,1)
    # We have to do that because sklearn.preprocessing.fit_transform()
    # requires a 2D array
    # Here n == len(scaled_data)
    # Then, we scale the whole array to the range (0,1)
    # The parameter -1 allows (np.)reshape to figure out the array size n automatically 
    # values.reshape(-1, 1) 
    # https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape'
    # When reshaping an array, the new shape must contain the same number of elements 
    # as the old shape, meaning the products of the two shapes' dimensions must be equal. 
    # When using a -1, the dimension corresponding to the -1 will be the product of 
    # the dimensions of the original array divided by the product of the dimensions 
    # given to reshape so as to maintain the same number of elements.


    #Check split type
    regexDate = r"^\d{4}-\d{2}-\d{2}$"
    regexRatio = r"^0\.\d+$"
    if (re.search(regexDate, split)):    #Check split type = Date YYYY-MM-DD
        split_index = full_data.index.get_loc(pd.to_datetime(split))  #set index of split to the location of the split date.
        train_data = full_data[:split_index] #values preceeding split are for training and those after are for testing
        test_data = full_data[split_index:]
    elif(re.search(regexRatio, split)):  #Check split type = Ratio 0.#
        split_index = int(len(full_data) * split)  #set index of split to a ratio of the data set
        train_data = full_data[:split_index]#values preceeding split are for training and those after are for testing
        test_data = full_data[split_index:]









def plot_candlestick_chart(data, n=1):
    """
    Creates candlestick chart with mplfinance.

    :param data: Gets stock data with Open, High, Low, Close price values.
    :param n: Number of trading days each candlestick represents (n >= 1).

    """
    #ensure n days >= 1
    days = n
    if (days<1):
        days = 1

    #resamples the data using Pandas.DataFrame.Resamples, with the parameter n days. 
    data = data.resample(f'{days}D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    
    #Uses mplfinance to plot the data as a candlestick chart
    mpf.plot(data, type='candle', style='charles', title=f'\n \n Candlestick Chart for {COMPANY},\n Trading Days/Candle = {days}',
             ylabel=f'{COMPANY} Price', volume=False)
    



def plot_boxplot_chart(data, n=5, price_column='Close'):
    """
    Plots a boxplot chart of stock data for a moving window of n consecutive trading days.

    :param data: Gets stock data, includes price columns.
    :param n: Window size; days per window
    :param price_column: The column name for the price data plotting (Open/Close/High/Low etc prices, default = Close).
    """

    # Creates list of moving window data; dataframe.iloc is used for index locating the data at i to i+n days through the length of the data
    window_data = [data[price_column].iloc[i:i + n] for i in range(len(data) - n + 1)]

    # Plot the boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(window_data)
    plt.title(f'Boxplot Chart for {COMPANY} Moving Window Size = {n} Days')
    plt.xlabel('Window Number')
    plt.ylabel(f'{price_column} Price')
    plt.show()






load_process_dataset()

# data = web.DataReader(COMPANY, DATA_SOURCE, TRAIN_START, TRAIN_END) # Read data using yahoo

plot_candlestick_chart(train_data, 10)


plot_boxplot_chart(train_data, n=5, price_column='Close')



# Number of days to look back to base the prediction
PREDICTION_DAYS = 60 # Original

# To store the training data
x_train = []
y_train = []

scaled_data = scaled_data[:,0] # Turn the 2D array back to a 1D array
# Prepare the data
for x in range(PREDICTION_DAYS, len(scaled_data)):
    x_train.append(scaled_data[x-PREDICTION_DAYS:x])
    y_train.append(scaled_data[x])

# Convert them into an array
x_train, y_train = np.array(x_train), np.array(y_train)
# Now, x_train is a 2D array(p,q) where p = len(scaled_data) - PREDICTION_DAYS
# and q = PREDICTION_DAYS; while y_train is a 1D array(p)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# We now reshape x_train into a 3D array(p, q, 1); Note that x_train 
# is an array of p inputs with each input being a 2D array 

#------------------------------------------------------------------------------
# Build the Model
## TO DO:
# 1) Check if data has been built before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Change the model to increase accuracy?
#------------------------------------------------------------------------------
model = Sequential() # Basic neural network
# See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
# for some useful examples

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# This is our first hidden layer which also spcifies an input layer. 
# That's why we specify the input shape for this layer; 
# i.e. the format of each training example
# The above would be equivalent to the following two lines of code:
# model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
# model.add(LSTM(units=50, return_sequences=True))
# For som eadvances explanation of return_sequences:
# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
# https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
# As explained there, for a stacked LSTM, you must set return_sequences=True 
# when stacking LSTM layers so that the next LSTM layer has a 
# three-dimensional sequence input. 

# Finally, units specifies the number of nodes in this layer.
# This is one of the parameters you want to play with to see what number
# of units will give you better prediction quality (for your problem)

model.add(Dropout(0.2))
# The Dropout layer randomly sets input units to 0 with a frequency of 
# rate (= 0.2 above) at each step during training time, which helps 
# prevent overfitting (one of the major problems of ML). 

model.add(LSTM(units=50, return_sequences=True))
# More on Stacked LSTM:
# https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1)) 
# Prediction of the next closing value of the stock price

# We compile the model by specify the parameters for the model
# See lecture Week 6 (COS30018)
model.compile(optimizer='adam', loss='mean_squared_error')
# The optimizer and loss are two important parameters when building an 
# ANN model. Choosing a different optimizer/loss can affect the prediction
# quality significantly. You should try other settings to learn; e.g.
    
# optimizer='rmsprop'/'sgd'/'adadelta'/...
# loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

# Now we are going to train this model with our training data 
# (x_train, y_train)
model.fit(x_train, y_train, epochs=25, batch_size=32)
# Other parameters to consider: How many rounds(epochs) are we going to 
# train our model? Typically, the more the better, but be careful about
# overfitting!
# What about batch_size? Well, again, please refer to 
# Lecture Week 6 (COS30018): If you update your model for each and every 
# input sample, then there are potentially 2 issues: 1. If you training 
# data is very big (billions of input samples) then it will take VERY long;
# 2. Each and every input can immediately makes changes to your model
# (a souce of overfitting). Thus, we do this in batches: We'll look at
# the aggreated errors/losses from a batch of, say, 32 input samples
# and update our model based on this aggregated loss.

# TO DO:
# Save the model and reload it
# Sometimes, it takes a lot of effort to train your model (again, look at
# a training data with billions of input samples). Thus, after spending so 
# much computing power to train your model, you may want to save it so that
# in the future, when you want to make the prediction, you only need to load
# your pre-trained model and run it on the new input for which the prediction
# need to be made.

#------------------------------------------------------------------------------
# Test the model accuracy on existing data
#------------------------------------------------------------------------------
# Load the test data
# TEST_START = '2023-08-02'
# TEST_END = '2024-07-02'

# test_data = web.DataReader(COMPANY, DATA_SOURCE, TEST_START, TEST_END)

##########################################test_data = yf.download(COMPANY,TEST_START,TEST_END)


# The above bug is the reason for the following line of code
# test_data = test_data[1:]

actual_prices = test_data[PRICE_VALUE].values

total_dataset = pd.concat((train_data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
# We need to do the above because to predict the closing price of the fisrt
# PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the 
# data from the training period

model_inputs = model_inputs.reshape(-1, 1)
# TO DO: Explain the above line

model_inputs = scaler.transform(model_inputs)
# We again normalize our closing price data to fit them into the range (0,1)
# using the same scaler used above 
# However, there may be a problem: scaler was computed on the basis of
# the Max/Min of the stock price for the period [TRAIN_START, TRAIN_END],
# but there may be a lower/higher price during the test period 
# [TEST_START, TEST_END]. That can lead to out-of-bound values (negative and
# greater than one)
# We'll call this ISSUE #2

# TO DO: Generally, there is a better way to process the data so that we 
# can use part of it for training and the rest for testing. You need to 
# implement such a way

#------------------------------------------------------------------------------
# Make predictions on test data
#------------------------------------------------------------------------------
x_test = []
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# TO DO: Explain the above 5 lines

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
# Clearly, as we transform our data into the normalized range (0,1),
# we now need to reverse this transformation 
#------------------------------------------------------------------------------
# Plot the test predictions
## To do:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
#------------------------------------------------------------------------------

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------


real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

# A few concluding remarks here:
# 1. The predictor is quite bad, especially if you look at the next day 
# prediction, it missed the actual price by about 10%-13%
# Can you find the reason?
# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# gives a much better prediction. Even though on the surface, it didn't seem 
# to be a big difference (both use Stacked LSTM)
# Again, can you explain it?
# A more advanced and quite different technique use CNN to analyse the images
# of the stock price changes to detect some patterns with the trend of
# the stock price:
# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
# Can you combine these different techniques for a better prediction??
