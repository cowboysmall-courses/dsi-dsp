#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 13:28:45 2024

@author: jerry


Global market indices of interest:

    NSEI:  Nifty 50 
    DJI:   Dow Jones Index
    IXIC:  Nasdaq
    HSI:   Hang Seng
    N225:  Nikkei 225
    GDAXI: Dax
    VIX:   Volatility Index

"""



# %% 0 -import required libraries
import pandas as pd
import yfinance as yf

from functools import reduce


# %% 0 -list of indices
indices = ['NSEI', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']


# %% 1 - function to retrieve data
def retrieve_data(index, start_date = '2017-12-1', end_date = '2024-1-31'):
    data = yf.download('^{}'.format(index), start_date, end_date)
    data['Daily Returns'] = data.Close.pct_change() * 100
    data.columns = ['{}_{}'.format(index, '_'.join(c.upper() for c in column.split())) for column in data.columns]
    return data


# %% 1 - function to read index data
# def read_file(index):
#     data = pd.read_csv("../data/raw/{}.csv".format(index))
#     # data.set_index('Date', inplace = True)
#     return data


# %% 2 - function to merge data
def merge_files(left, right):
    return pd.merge(left, right, left_index = True, right_index = True, how = 'outer')
    # return pd.merge(left, right, how = 'outer')


# %% 3 - read data, merge data, and process the data
data = [retrieve_data(index) for index in indices]
# data = list(reversed(sorted(data, key = lambda x: x.shape[0])))
# data

merged1 = reduce(merge_files, data)
merged2 = pd.concat(data, axis = 1)

merged1 = merged1['2018-01-02':'2023-12-29']
merged2 = merged2['2018-01-02':'2023-12-29']

merged1.head()
merged2.head()

merged1.info()
merged2.info()

# compared = merged1.compare(merged2)
# compared.head()
# compared.info()

