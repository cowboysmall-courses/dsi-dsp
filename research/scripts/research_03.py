#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 17:09:29 2024

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
import statsmodels.api as sm

from scipy import stats


# %% 0 - list of indices
indices = ['NSEI', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']


# %% 1 - function to read index data
def read_file(index):
    data = pd.read_csv("./data/raw/{}.csv".format(index), index_col = 'Date')
    data.index = pd.to_datetime(data.index)
    return data


# %% 2 - function to test for normality
def check_normality(index, data):
    column = '{}_DAILY_RETURNS'.format(index)

    print()
    print('\t Index {}'.format(index))
    print('\tColumn {}'.format(column))

    result = stats.shapiro(data[column].dropna())
    if result[1] < 0.05:
        print('\t     Shapiro-Wilks Test:         reject null hypothesis - with p-value = {}'.format(result[1]))
    else:
        print('\t     Shapiro-Wilks Test: fail to reject null hypothesis - with p-value = {}'.format(result[1]))

    result = sm.stats.diagnostic.lilliefors(data[column].dropna())
    if result[1] < 0.05:
        print('\tKolmogorov-Smirnov Test:         reject null hypothesis - with p-value = {}'.format(result[1]))
    else:
        print('\tKolmogorov-Smirnov Test: fail to reject null hypothesis - with p-value = {}'.format(result[1]))

    print()


# %% 3 - test for normality of data
for index in indices:
    check_normality(index, read_file(index))
