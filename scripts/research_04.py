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

merged = merged['2018-01-02':'2023-12-29']

merged.head()
#                NSEI_OPEN     NSEI_HIGH      NSEI_LOW  ...    MONTH  QUARTER  YEAR
# Date                                                  ...                        
# 2018-01-02  10477.549805  10495.200195  10404.650391  ...  2018-01   2018Q1  2018
# 2018-01-03  10482.650391  10503.599609  10429.549805  ...  2018-01   2018Q1  2018
# 2018-01-04  10469.400391  10513.000000  10441.450195  ...  2018-01   2018Q1  2018
# 2018-01-05  10534.250000  10566.099609  10520.099609  ...  2018-01   2018Q1  2018
# 2018-01-08  10591.700195  10631.200195  10588.549805  ...  2018-01   2018Q1  2018

# [5 rows x 52 columns]

merged.info()
# Index: 1563 entries, 2018-01-02 to 2023-12-29
# Data columns (total 52 columns):
#  #   Column               Non-Null Count  Dtype        
# ---  ------               --------------  -----        
#  0   NSEI_OPEN            1563 non-null   float64      
#  1   NSEI_HIGH            1563 non-null   float64      
#  2   NSEI_LOW             1563 non-null   float64      
#  3   NSEI_CLOSE           1563 non-null   float64      
#  4   NSEI_ADJ_CLOSE       1563 non-null   float64      
#  5   NSEI_VOLUME          1563 non-null   float64      
#  6   NSEI_DAILY_RETURNS   1563 non-null   float64      
#  7   DJI_OPEN             1563 non-null   float64      
#  8   DJI_HIGH             1563 non-null   float64      
#  9   DJI_LOW              1563 non-null   float64      
#  10  DJI_CLOSE            1563 non-null   float64      
#  11  DJI_ADJ_CLOSE        1563 non-null   float64      
#  12  DJI_VOLUME           1563 non-null   float64      
#  13  DJI_DAILY_RETURNS    1563 non-null   float64      
#  14  IXIC_OPEN            1563 non-null   float64      
#  15  IXIC_HIGH            1563 non-null   float64      
#  16  IXIC_LOW             1563 non-null   float64      
#  17  IXIC_CLOSE           1563 non-null   float64      
#  18  IXIC_ADJ_CLOSE       1563 non-null   float64      
#  19  IXIC_VOLUME          1563 non-null   float64      
#  20  IXIC_DAILY_RETURNS   1563 non-null   float64      
#  21  HSI_OPEN             1563 non-null   float64      
#  22  HSI_HIGH             1563 non-null   float64      
#  23  HSI_LOW              1563 non-null   float64      
#  24  HSI_CLOSE            1563 non-null   float64      
#  25  HSI_ADJ_CLOSE        1563 non-null   float64      
#  26  HSI_VOLUME           1563 non-null   float64      
#  27  HSI_DAILY_RETURNS    1563 non-null   float64      
#  28  N225_OPEN            1563 non-null   float64      
#  29  N225_HIGH            1563 non-null   float64      
#  30  N225_LOW             1563 non-null   float64      
#  31  N225_CLOSE           1563 non-null   float64      
#  32  N225_ADJ_CLOSE       1563 non-null   float64      
#  33  N225_VOLUME          1563 non-null   float64      
#  34  N225_DAILY_RETURNS   1563 non-null   float64      
#  35  GDAXI_OPEN           1563 non-null   float64      
#  36  GDAXI_HIGH           1563 non-null   float64      
#  37  GDAXI_LOW            1563 non-null   float64      
#  38  GDAXI_CLOSE          1563 non-null   float64      
#  39  GDAXI_ADJ_CLOSE      1563 non-null   float64      
#  40  GDAXI_VOLUME         1563 non-null   float64      
#  41  GDAXI_DAILY_RETURNS  1563 non-null   float64      
#  42  VIX_OPEN             1563 non-null   float64      
#  43  VIX_HIGH             1563 non-null   float64      
#  44  VIX_LOW              1563 non-null   float64      
#  45  VIX_CLOSE            1563 non-null   float64      
#  46  VIX_ADJ_CLOSE        1563 non-null   float64      
#  47  VIX_VOLUME           1563 non-null   float64      
#  48  VIX_DAILY_RETURNS    1563 non-null   float64      
#  49  MONTH                1563 non-null   period[M]    
#  50  QUARTER              1563 non-null   period[Q-DEC]
#  51  YEAR                 1563 non-null   period[A-DEC]
# dtypes: float64(49), period[A-DEC](1), period[M](1), period[Q-DEC](1)
# memory usage: 647.2+ KB

merged.to_csv("../data/processed/master_data.csv")
