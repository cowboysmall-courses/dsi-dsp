#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 11:42:07 2024

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



# %% 0 - import required libraries
import pandas as pd


# %% 0 - list of indices
indices = ['NSEI', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']


# %% 1 - function to read index data
def read_file(index):
    data = pd.read_csv("./data/raw/{}.csv".format(index), index_col = 'Date')
    data.index = pd.to_datetime(data.index)
    return data


# %% 2 - function to merge data
def merge_files(files):
    merged = pd.concat(files, axis = 1)

    merged.fillna(method = 'ffill', inplace = True)

    merged['MONTH']   = pd.PeriodIndex(merged.index, freq = 'M')
    merged['QUARTER'] = pd.PeriodIndex(merged.index, freq = 'Q')
    merged['YEAR']    = pd.PeriodIndex(merged.index, freq = 'Y')

    return merged['2018-01-02':'2023-12-29']


# %% 3 - read data, merge data, and save the data
merge_files([read_file(index) for index in indices]).to_csv("./data/processed/master_data.csv")
