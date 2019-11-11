#Code for setting up the data so that we can start to build the model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import alpha_vantage
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Lasso
from pandas import Series
import time
import csv 
import json
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from ta import *
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from IPython.display import display
import io
plt.rcParams['figure.figsize'] = (18, 12)
file='SP500hist_data.csv'

data=pd.DataFrame()
data=pd.read_csv('SP500hist_data.csv',header=0,encoding = 'unicode_escape')


#take out the comma in string objects and convert them to a floating point #
data['Price'] = data["Price"].apply(lambda x: float(x.split()[0].replace(',', '')))
data['Open'] = data["Open"].apply(lambda x: float(x.split()[0].replace(',', '')))
data['High'] = data["High"].apply(lambda x: float(x.split()[0].replace(',', '')))
data['Low'] = data["Low"].apply(lambda x: float(x.split()[0].replace(',', '')))
data['Change %'] = data["Change %"].apply(lambda x: float(x.split()[0].replace('%', '')))

# getting the data to where we can work with it
stock_price=data['Price']
opening_price=data["Open"]
daily_high=data['High']
daily_low=data['Low']
closing_price=data['close']
adjusted_close=data['adjusted_close']
daily_volume=data['volume']
daily_change_percent=data['Change %']
rsi=data['RSI']
sma=data['SMA']
ema=data['EMA']
mac_d=data['MACD']
macd_hist=data['MACD_Hist']
macd_signal=data['MACD_Signal']
slow_D=data['SlowD']
slow_K=data['SlowK']
wma=data['WMA']
real_upper_band=data['Real Upper Band']
real_lower_band=data['Real Lower Band']
real_middle_band=data['Real Middle Band']
chaikin_AD=data['Chaikin A/D']
obv=data['OBV']
mom=data['MOM']
willr=data['WILLR']
adx=data['ADX']
cci=data['CCI']
aroon_up=data['Aroon Up']
aroon_down=data['Aroon Down']
 #I made this list just case we need to use it later to build model   &&   #  maybe take out daily change percentage
feature_list = ['Price','Open','High','Low', 'close','adjusted_close', 'volume','Change %','RSI','SMA','EMA',\
    'MACD','MACD_Hist','MACD_Signal','SlowD','SlowK','WMA','Real Upper Band','Real Lower Band',\
        'Real Middle Band','Chaikin A/D','OBV','MOM','WILLR','ADX','CCI','Aroon Up','Aroon Down']


Xs = data.drop(['adjusted_close'], axis=1)
y = data['adjusted_close'].values.reshape(-1,1)

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv = 5)

lasso_regressor.fit(Xs, y)

print("The best parameter is:", lasso_regressor.best_params_)
print("The best score is:", lasso_regressor.best_score_)

# start preprocessing
# NEED to change this to the data where it is not normalized

# Separate the train and test data 80/20
X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=.2, random_state=10)
#Implement lasso regression
lassoReg = Lasso(alpha=0.3, normalize=True)
lassoReg.fit(X_train,y_train)
pred_cv = lassoReg.predict(X_test)
# calculating mse
mse = np.mean((pred_cv - y_test)**2)
print("MSE for lasso Regression:",mse)
print("Score for Lasso Regression:",lassoReg.score(X_test,y_test))
coeffs = lassoReg.coef_       
print("Lasso Coefficients:",coeffs)
coeffs = lassoReg.sparse_coef_  
print("Lasso Sparse coefficients:",coeffs)
coeffs = lassoReg.intercept_    
print("Lasso Intercept:",coeffs)
# Normalizes data for use when need be

def get_normalized_data(data,feature_list,scaler=None):
    
    # Initialize a scaler, then apply it to the features
    if scaler=='MinMaxScaler()':
        scaler = MinMaxScaler()
        data[feature_list] = scaler.fit_transform(data[feature_list])
    if scaler=='StandardScaler()':
        scaler=StandardScaler()
        data[feature_list] = scaler.fit_transform(data[feature_list])

    return data


def price(x):
    """
    formats data for the @plot_basic
    """
    return '$%1.2f' % x

def plot_basic(data, title='S&P500', y_label='Price USD', x_label='Trading Days'):
    
    fig, ax = plt.subplots()
    ax.plot(data['Day'], data['adjusted_close'],'#0A7388')
    
    ax.format_ydata = price
    ax.set_title(title)

    # Add labels
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    plt.show()

# plot_basic(data)



def scale_range(x, input_range, target_range):
    
    range = [np.amin(x), np.amax(x)]
    x_std = (x - input_range[0]) / (1.0*(input_range[1] - input_range[0]))
    x_scaled = x_std * (1.0*(target_range[1] - target_range[0])) + target_range[0]
    return x_scaled, range



def train_test_split_linear_regression(data):
    
    # Create numpy arrays for features and targets
    feature = []
    label = []

    # Convert dataframe columns to numpy arrays for scikit learn
    # fix this later
    for index, row in data.iterrows():
        # print([np.array(row['close'])])
        feature.append([row['Day']])
        label.append([(row['adjusted_close'])])
        

    # Regularize the feature and target arrays and store min/max of input data for rescaling later
    feature_bounds = [min(feature), max(feature)]
    feature_bounds = [feature_bounds[0][0], feature_bounds[1][0]]
    label_bounds = [min(label), max(label)]
    label_bounds = [label_bounds[0][0], label_bounds[1][0]]

    feature_scaled, feature_range = scale_range(np.array(feature), input_range=feature_bounds, target_range=[-1.0, 1.0])
    label_scaled, label_range = scale_range(np.array(label), input_range=label_bounds, target_range=[-1.0, 1.0])

    # Define Test/Train Split 80/20
    split = .2
    split = int(math.floor(len(data['Day']) * split))

    # Set up training and test sets
    X_train = feature_scaled[:-split]
    X_test = feature_scaled[-split:]

    y_train = label_scaled[:-split]
    y_test = label_scaled[-split:]

    return X_train, X_test, y_train, y_test, label_range

stocks_normalized=get_normalized_data(data,feature_list,'MinMaxScaler()')
stocks_normalized.to_csv('stocks_preprocessed.csv',index= False)

normalized_stocks=pd.read_csv('stocks_preprocessed.csv')
normalized_stocks.index = np.arange(1, len(normalized_stocks)+1)
# print(normalized_stocks)
X_train, X_test, y_train, y_test, label_range= train_test_split_linear_regression(normalized_stocks)
# print("label range:",label_range)
# print("x_train", X_train.shape)
# print("y_train", y_train.shape)
# print("x_test", X_test.shape)
# print("y_test", y_test.shape)




def build_model(X, y):
    
    X = np.reshape(X, (X.shape[0], 1))
    y = np.reshape(y, (y.shape[0], 1))
    linear_mod = linear_model.LinearRegression()  # defining the linear regression model
    
    linear_mod.fit(X, y)  # fitting the data points in the model

    return linear_mod
def predict_prices(model, x, label_range):
    
    x = np.reshape(x, (x.shape[0], 1))
    predicted_price = model.predict(x)
    predictions_rescaled, re_range = scale_range(predicted_price, input_range=[-1.0, 1.0], target_range=label_range)

    return predictions_rescaled.flatten()
    
#
print(data.head())

# # print(normalized_stocks)
# # print("Open   --- mean :", np.mean(normalized_stocks['Open']),  "  \t Std: ", np.std(normalized_stocks['Open']),  "  \t Max: ", np.max(normalized_stocks['Open']),  "  \t Min: ", np.min(normalized_stocks['Open']))
# # print("Close  --- mean :", np.mean(normalized_stocks['close']), "  \t Std: ", np.std(normalized_stocks['close']), "  \t Max: ", np.max(normalized_stocks['close']), "  \t Min: ", np.min(normalized_stocks['close']))
# # print("Volume --- mean :", np.mean(normalized_stocks['volume']),"  \t Std: ", np.std(normalized_stocks['volume']),"  \t Max: ", np.max(normalized_stocks['volume']),"  \t Min: ", np.min(normalized_stocks['volume']))

# # linear regression model



def plot_prediction(actual, prediction, title='S&P500 vs. Prediction', y_label='Price USD', x_label='Trading Days'):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Add labels
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    # Plot actual and predicted close values

    plt.plot(actual, '#00FF00', label='Adjusted Close')
    plt.plot(prediction, '#0000FF', label='Predicted Close')

    # Set title
    ax.set_title(title)
    ax.legend(loc='upper left')
    plt.show()
#Building the model for simple linear regression
# model=build_model(X_train,y_train)
# simple_predictions=predict_prices(model,X_test, label_range)
# plot_prediction(y_test,simple_predictions)


normalized_stocks=normalized_stocks.drop('Day',1)
# display(normalized_stocks.head())

def train_test_split_lstm(stocks, prediction_time=1, test_data_size=400, unroll_length=50):
    
    # training data
    test_data_cut = test_data_size + unroll_length + 1

    x_train = stocks[0:-prediction_time - test_data_cut].as_matrix()
    y_train = stocks[prediction_time:-test_data_cut]['adjusted_close'].as_matrix()

    # test data
    x_test = stocks[0 - test_data_cut:-prediction_time].as_matrix()
    y_test = stocks[prediction_time - test_data_cut:]['adjusted_close'].as_matrix()

    return x_train, x_test, y_train, y_test


def unroll(data, sequence_length=24):
    
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    return np.asarray(result)
X_train, X_test,y_train, y_test = train_test_split_lstm(normalized_stocks, 5)

unroll_length = 50
X_train = unroll(X_train, unroll_length)
X_test = unroll(X_test, unroll_length)
y_train = y_train[-X_train.shape[0]:]
y_test = y_test[-X_test.shape[0]:]
print("x_train", X_train.shape)
print("y_train", y_train.shape)
print("x_test", X_test.shape)
print("y_test", y_test.shape)
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
def build_basic_model(input_dim, output_dim, return_sequences):
    
    model = Sequential()
    model.add(LSTM(
        input_shape=(None, input_dim),
        units=output_dim,
        return_sequences=return_sequences))

    model.add(LSTM(
        100,
        return_sequences=False))

    model.add(Dense(
        units=1))
    model.add(Activation('linear'))

    return model
model=build_basic_model(input_dim = X_train.shape[-1],output_dim = unroll_length, return_sequences=True)

# Compile the model
start = time.time()
model.compile(loss='mean_squared_error', optimizer='adam')
# print('compilation time : ', time.time() - start)

model.fit(X_train,y_train,epochs=5,validation_split=.05)

predictions = model.predict(X_test)
def plot_lstm_prediction(actual, prediction, title='S&P500', y_label='Price USD', x_label='Trading Days'):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Add labels
    plt.ylabel(y_label)
    plt.xlabel(x_label)

    # Plot actual and predicted close values

    plt.plot(actual, '#00FF00', label='Adjusted Close')
    plt.plot(prediction, '#0000FF', label='Predicted Close')

    # Set title
    ax.set_title(title)
    ax.legend(loc='upper left')


    plt.show()
plot_lstm_prediction(y_test,predictions)

trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.8f MSE (%.8f RMSE)' % (trainScore, math.sqrt(trainScore)))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore)))

def build_improved_model(input_dim, output_dim, return_sequences):
    
    model = Sequential()
    model.add(LSTM(
        input_shape=(None, input_dim),
        units=output_dim,
        return_sequences=return_sequences))

    model.add(Dropout(0.2))
# .4 is best so far

    model.add(LSTM(
        128,
        return_sequences=False))

    # model.add(Dropout(0.2))

    model.add(Dense(
        units=1))
    model.add(Activation('linear'))

    return model


# Set up hyperparameters
batch_size = 20

epochs = 5

# # build improved lstm model
# model = build_improved_model( X_train.shape[-1],output_dim = unroll_length, return_sequences=True)
# start = time.time()
# #final_model.compile(loss='mean_squared_error', optimizer='adam')
# model.compile(loss='mean_squared_error', optimizer='adam')
# print('compilation time : ', time.time() - start)
# model.fit(X_train, y_train, batch_size=batch_size,epochs=epochs,verbose=2,validation_split=0.05)
# predictions = model.predict(X_test, batch_size=batch_size)
# plot_lstm_prediction(y_test,predictions)
# trainScore = model.evaluate(X_train, y_train, verbose=0)
# print('Train Score: %.8f MSE (%.8f RMSE)' % (trainScore, math.sqrt(trainScore)))

# testScore = model.evaluate(X_test, y_test, verbose=0)
# print('Test Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore)))

def plotPCA(normalized_stocks):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(normalized_stocks)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['Principal Component 1', 'Principal Component 2'])
    finalDf = pd.concat([principalDf, data[['adjusted_close']]], axis = 1)

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = data['adjusted_close']
    colors = ['r', 'g', 'b']
    for target in targets:
        indicesToKeep = finalDf['adjusted_close'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'Principal Component 1']
                   , finalDf.loc[indicesToKeep, 'Principal Component 2']
                   , s = 40)
    print(pca.explained_variance_ratio_)
    ax.grid()
    plt.show()
# plotPCA(normalized_stocks)



