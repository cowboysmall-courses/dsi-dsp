#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:00:17 2024

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

from functools import reduce


# %% 0 -list of indices
indices = ['NSEI', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']


# %% 1 - function to read index data
def read_file(index):
    data = pd.read_csv("../data/raw/{}.csv".format(index))
    data.set_index('Date', inplace = True)
    return data


# %% 2 - function to merge data
def merge_files(left, right):
    return pd.merge(left, right, left_index = True, right_index = True, how = 'outer')


# %% 3 - read data, merge data, and process the data
merged = reduce(merge_files, [read_file(index) for index in indices])
merged.fillna(method = 'ffill', inplace = True)

merged['MONTH']   = pd.PeriodIndex(merged.index, freq = 'M')
merged['QUARTER'] = pd.PeriodIndex(merged.index, freq = 'Q')
merged['YEAR']    = pd.PeriodIndex(merged.index, freq = 'Y')

merged['2018-01-02':'2023-12-29'].to_csv("../data/processed/master_data.csv")
