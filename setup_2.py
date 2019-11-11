#Code for setting up the data so that we can start to build the model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import alpha_vantage
import csv 
import json
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from ta import *
import time

file='SP500hist_data.csv'
json_file='SP500hist_data.json'
data=pd.DataFrame()
data=pd.read_csv('SP500hist_data.csv',header=0,encoding = 'unicode_escape')
data=data.drop(["Date"],axis=1)
# data=data.drop(["Chaikin A/D"],axis=1)
print(data)


data['Price'] = data["Price"].apply(lambda x: float(x.split()[0].replace(',', '')))
data['Open'] = data["Open"].apply(lambda x: float(x.split()[0].replace(',', '')))
data['High'] = data["High"].apply(lambda x: float(x.split()[0].replace(',', '')))
data['Low'] = data["Low"].apply(lambda x: float(x.split()[0].replace(',', '')))
data['Change %'] = data["Change %"].apply(lambda x: float(x.split()[0].replace('%', '')))
print(data)
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
                                                                         # maybe take out daily change percentage
feature_list=[opening_price,daily_high,daily_low,closing_price,adjusted_close,daily_volume,daily_change_percent,\
rsi,sma,ema,mac_d,macd_hist,macd_signal,slow_D,slow_K,wma,real_upper_band,real_lower_band,real_middle_band,chaikin_AD,obv,mom,willr,\
    adx,cci,aroon_up,aroon_down]
# target_raw = (SP500['Adj Close'].shift(-1)/SP500['Adj Close'])-1
target_raw = (adjusted_close.shift(-1)/adjusted_close)-1
# new_data=pd.DataFrame()
# new_data=data.drop(["¥éËDate"],axis=1)


# print(target_raw.head())

print("Our new dataset has {}".format(data.shape[0]), "rows and {}".format(data.shape[1]), "columns")
pca = PCA()
classifier = DecisionTreeRegressor()
X_transformed = pca.fit_transform(data)
classifier=classifier.fit(X_transformed, data)
answer=classifier.feature_importances_
print(answer)