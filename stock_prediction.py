import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import os
import plotly.graph_objects as go
import mplfinance as mpf
import pmdarima as pm


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer, GRU, SimpleRNN, Dense, Dropout



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


#plot_candlestick_chart(train_data, 10)


#plot_boxplot_chart(train_data, n=5, price_column='Close')



def create_dl_model(layer_type='LSTM', num_layers=3, units=50, dropout=0.2, input_shape=(60, 1), output_size=1):
    """
    Creates a deep learning model with provided parameters
    
    :param layer_type: The type of layer (LSTM, GRU, RNN).
    :param num_layers: Number of layers in the model.
    :param units: Number of units/neurons per layer.
    :param dropout: Dropout rate to prevent overfitting.
    :param input_shape: Shape of the input data.
    :param output_size: Number of steps ahead to predict.
    :return: Compiled deep learning model.
    """
    model = Sequential()
    layer_map = {
        'LSTM': LSTM,
        'GRU': GRU,
        'RNN': SimpleRNN
    }

    Layer = layer_map[layer_type]

    # Add input layer
    model.add(Layer(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))

    # Add middle layer stack
    for _ in range(1, num_layers - 1):
        model.add(Layer(units=units, return_sequences=True))
        model.add(Dropout(dropout))

    # Add the final recurrent layer without return_sequences
    model.add(Layer(units=units))
    model.add(Dropout(dropout))

    # Add output layer with size based on the number of steps ahead
    model.add(Dense(units=output_size))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model





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


# Define model parameters
input_shape = (x_train.shape[1], 1)  # As your input is (p, q, 1) where p = num_samples and q = PREDICTION_DAYS



#------------------------------------------------------
#Experimenting Models
#------------------------------------------------------
# different DL networks (e.g., LSTM, RNN, GRU,etc.) and with different 
# hyperparameter configurations (e.g. different numbers of layers and
# layer sizes, number of epochs, batch sizes, etc.)



# Create LSTM model 
#model = create_dl_model(layer_type='LSTM', num_layers=3, units=50, dropout=0.2, input_shape=input_shape)
# Train model
#model.fit(x_train, y_train, epochs=25, batch_size=32)
#print(f"Finished training LSTM model\n")

# Create GRU model 
#model = create_dl_model(layer_type='GRU', num_layers=3, units=50, dropout=0.3, input_shape=input_shape)
# Train model
#model.fit(x_train, y_train, epochs=25, batch_size=32)
#print(f"Finished training GRU model\n")

# Create RNN model 
#model = create_dl_model(layer_type='RNN', num_layers=4, units=100, dropout=0.2, input_shape=input_shape)
# Train model
#model.fit(x_train, y_train, epochs=25, batch_size=32)
#print(f"Finished training RNN model\n")



def create_multistep_sequences(data, window_size, steps_ahead):
    """
    Creates input output pairs for multistep predictions 
    Input sequences contain data of past window size days
    Output sequences contain data of next steps ahead
    :param data: gets stock data
    :param window_size: window size for past/input prices.
    :param steps_ahead: prediction days (k)
    :return: input and output sequence arrays.
    """
    X, y = [], []
    for i in range(len(data) - window_size - steps_ahead):
        #append input sequences of window size
        X.append(data[i:i+window_size])
        #append ouput sequences
        y.append(data[i+window_size:i+window_size+steps_ahead])
    return np.array(X), np.array(y)



def create_multivariate_sequences(data, window_size):
    """
    Creates input output pairs for multivariate prediction.
    Input sequences contain the multivariate (open, high, low, close etc) data of the past window size days
    Output sequence contains closing price of the next day.

    :param data: multivariate data with columns ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    :param window_size: the number of past days to use as input 
    :return: X (input sequences), y (output target - closing price).
    """
    X, y = [], []
    
    # get multivariate features and close price from data
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    target_column = 'Close'
    
    # prepare input sequences and target (closing price)
    for i in range(window_size, len(data)):
        # input sequence: all features for the last window size days
        X.append(data[feature_columns].iloc[i - window_size:i].values)
        # target sequence: closing price for the next day
        y.append(data[target_column].iloc[i])
    
    return np.array(X), np.array(y)



def create_multistep_multivariate_sequences(data, window_size, steps_ahead):
    """
    Creates input output pairs for multistep, multivariate prediction.
    Input sequences contain the multivariate (open, high, low, close, etc.) data of past window size days.
    Output sequences contain closing prices for the next steps ahead days.

    :param data: multivariate data with columns ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    :param window_size: the number of past days to use as input 
    :param steps_ahead: number of future days to predict (k)
    :return: X (input sequences), y (target - future closing prices).
    """
    X, y = [], []

    # multivariate feature columns and the target column (Close price)
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    target_column = 'Close'

    # prepare input sequences and output target (closing prices for steps ahead days)
    for i in range(len(data) - window_size - steps_ahead):
        # input sequence: multivariate data for the past window size days
        X.append(data[feature_columns].iloc[i:i + window_size].values)

        # target sequence: closing prices for the next steps ahead days
        y.append(data[target_column].iloc[i + window_size:i + window_size + steps_ahead].values)
    
    return np.array(X), np.array(y)



def ensemble_modeling():
    # Train ARIMA model
    arima_model = pm.auto_arima(train_data['Close'], 
                                seasonal=False,  # True = SARIMA
                                stepwise=True, 
                                suppress_warnings=True)
    # Fit ARIMA model on training data
    arima_model.fit(train_data['Close'])
    # Get ARIMA predictions
    arima_predictions = arima_model.predict(n_periods=len(test_data))



    # Train LSTM model  
    window_size = 60  # past days to look at
    steps_ahead = 5  # predict 5 days ahead
    
    # Scale data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_data['Close'].values.reshape(-1, 1))
    scaled_test = scaler.transform(test_data['Close'].values.reshape(-1, 1))
    # Prepare LSTM model data (input-output sequences) with multistep function
    X_train, y_train = create_multistep_sequences(scaled_train, window_size, steps_ahead)
    # Create LSTM model
    input_shape = (window_size, 1)  # (time_steps, features)
    lstm_model = create_dl_model(input_shape=input_shape, output_size=steps_ahead)
    # Train LSTM model
    lstm_model.fit(X_train, y_train, epochs=20, batch_size=32)

    # LSTM predictions
    X_test, _ = create_multistep_sequences(scaled_test, window_size, steps_ahead)
    lstm_predictions = lstm_model.predict(X_test)
    lstm_predictions = scaler.inverse_transform(lstm_predictions)

    # Ensemble Prediction = average
    ensemble_predictions = (arima_predictions[:len(lstm_predictions)] + lstm_predictions[:, 0]) / 2

    print("ARIMA Predictions: ", arima_predictions[:len(lstm_predictions)])
    print("LSTM Predictions: ", lstm_predictions[:, 0])
    print("Ensemble Predictions: ", ensemble_predictions)


ensemble_modeling()

