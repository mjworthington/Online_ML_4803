import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
#import alpha_vantage
import random
from pprint import pprint
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
#from keras.layers.core import Dense, Activation, Dropout
#from keras.layers.recurrent import LSTM
#from keras.models import Sequential
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Lasso
from pandas import Series
import time
import csv 
import json
import sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
#from alpha_vantage.timeseries import TimeSeries
#from alpha_vantage.techindicators import TechIndicators
from sklearn.preprocessing import StandardScaler
#from yellowbrick.features.pca import PCADecomposition
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
#from ta import *
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from IPython.display import display
import io
plt.rcParams['figure.figsize'] = (18, 12)


data1=pd.DataFrame()
data1=pd.read_csv('NASDAQ_FEATURES.csv',header=0,encoding = 'unicode_escape',index_col='timestamp')
data1=data1.reset_index(drop=True)
data2=pd.DataFrame()
data2=pd.read_csv('NASDAQ_FEATURES.csv',header=0,encoding = 'unicode_escape',index_col='timestamp')
data2=data2.reset_index(drop=True)
data3=pd.DataFrame()
data3=pd.read_csv('NASDAQ_FEATURES.csv',header=0,encoding = 'unicode_escape',index_col='timestamp')
data3=data3.reset_index(drop=True)
data4=pd.DataFrame()
data4=pd.read_csv('NASDAQ_FEATURES.csv',header=0,encoding = 'unicode_escape',index_col='timestamp')
data4=data4.reset_index(drop=True)
data5=pd.DataFrame()
data5=pd.read_csv('NASDAQ_FEATURES.csv',header=0,encoding = 'unicode_escape',index_col='timestamp')
data5=data5.reset_index(drop=True)

high_diff1=data1['high']-data1['open']
data1['high_diff']=high_diff1
# low difference
low_diff1=data1['open']-data1['low']
data1['low_diff']=low_diff1
# Daily Difference
daily_diff1=data1['close']-data1['open']
data1['daily_diff']=daily_diff1
# Data2 setup adding high diff, low diff & daily diff
# High Difference 
high_diff2=data2['high']-data2['open']
data2['high_diff']=high_diff2
# low difference
low_diff2=data2['open']-data2['low']
data2['low_diff']=low_diff2
# Daily Difference
daily_diff2=data2['close']-data2['open']
data2['daily_diff']=daily_diff2
# Data3 setup adding high diff, low diff & daily diff
# High Difference 
high_diff3=data3['high']-data3['open']
data3['high_diff']=high_diff3
# low difference
low_diff3=data3['open']-data3['low']
data3['low_diff']=low_diff3
# Daily Difference
daily_diff=data3['close']-data3['open']
data3['daily_diff']=daily_diff
# Data4 setup adding high diff, low diff & daily diff
# High Difference 
high_diff4=data4['high']-data4['open']
data4['high_diff']=high_diff4
# low difference
low_diff4=data4['open']-data4['low']
data4['low_diff']=low_diff4
# Daily Difference
daily_diff=data4['close']-data4['open']
data4['daily_diff']=daily_diff
# Data5 setup adding high diff, low diff & daily diff
# High Difference 
high_diff5=data5['high']-data5['open']
data5['high_diff']=high_diff5
# low difference
low_diff5=data5['open']-data5['low']
data5['low_diff']=low_diff5
# Daily Difference
daily_diff=data5['close']-data5['open']
data5['daily_diff']=daily_diff
data5['change'] = data5['daily_diff'] >= 0
data5['change'] = data5['change'] * 1


#print(data5.head(2))
 #I made this list just case we need to use it later to build model   &&   #  maybe take out daily change percentage
feature_list = ['open','high','low', 'close', 'volume','RSI','SMA','EMA',\
    'MACD','MACD_Hist','MACD_Signal','SlowD','SlowK','WMA','Real Upper Band','Real Lower Band',\
        'Real Middle Band','KAMA','DEMA','ADX','Aroon Up','Aroon Down','daily_diff','high_diff','low_diff']


#begin here
days = [data5.Day.unique()]
reality = [data5.change.unique()]


print(reality)

#data5['Model1Prediction'] = 

    
