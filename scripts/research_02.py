#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 22:16:27 2024

@author: jerry
"""



# %% 0 -import required libraries
import yfinance as yf


# %% 0 -list of indices
indices = ['NSEI', 'GSPC', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']


# %% 1 - function to retrieve data
def save_data(index, start_date = '2017-12-1', end_date = '2024-1-31'):
    data = yf.download('^{}'.format(index), start_date, end_date)
    data['Daily Returns'] = data.Close.pct_change() * 100
    data.columns = ['{}_{}'.format(index, '_'.join(c.upper() for c in column.split())) for column in data.columns]
    data.to_csv("../data/raw/{}.csv".format(index))


# %% 2 - retrieve data for indices
for index in indices:
    save_data(index)
