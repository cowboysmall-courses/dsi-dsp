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



# %% 0 - import required libraries
import pandas as pd


# %% 0 - list of indices
indices = ['NSEI', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']


# %% 1 - function to read master data
def read_master():
    data = pd.read_csv("../data/processed/master_data.csv", index_col = 'Date')

    data.index = pd.to_datetime(data.index)

    data['MONTH']   = pd.PeriodIndex(data['MONTH'],   freq = 'M')
    data['QUARTER'] = pd.PeriodIndex(data['QUARTER'], freq = 'Q')
    data['YEAR']    = pd.PeriodIndex(data['YEAR'],    freq = 'Y')

    return data


# %% 2 - read master data
data = read_master()

data.head()
#                NSEI_OPEN     NSEI_HIGH      NSEI_LOW  ...    MONTH  QUARTER  YEAR
# Date                                                  ...
# 2018-01-02  10477.549805  10495.200195  10404.650391  ...  2018-01   2018Q1  2018
# 2018-01-03  10482.650391  10503.599609  10429.549805  ...  2018-01   2018Q1  2018
# 2018-01-04  10469.400391  10513.000000  10441.450195  ...  2018-01   2018Q1  2018
# 2018-01-05  10534.250000  10566.099609  10520.099609  ...  2018-01   2018Q1  2018
# 2018-01-08  10591.700195  10631.200195  10588.549805  ...  2018-01   2018Q1  2018

# [5 rows x 52 columns]

data.info()
# DatetimeIndex: 1563 entries, 2018-01-02 to 2023-12-29
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
# memory usage: 647.2 KB

data.sum(numeric_only = True)
# NSEI_OPEN              2.227940e+07
# NSEI_HIGH              2.238517e+07
# NSEI_LOW               2.213283e+07
# NSEI_CLOSE             2.226385e+07
# NSEI_ADJ_CLOSE         2.226385e+07
# NSEI_VOLUME            6.144286e+08
# NSEI_DAILY_RETURNS     9.394397e+01
# DJI_OPEN               4.673150e+07
# DJI_HIGH               4.700724e+07
# DJI_LOW                4.643958e+07
# DJI_CLOSE              4.673657e+07
# DJI_ADJ_CLOSE          4.673657e+07
# DJI_VOLUME             5.388821e+11
# DJI_DAILY_RETURNS      6.018719e+01
# IXIC_OPEN              1.696537e+07
# IXIC_HIGH              1.709140e+07
# IXIC_LOW               1.682828e+07
# IXIC_CLOSE             1.696834e+07
# IXIC_ADJ_CLOSE         1.696834e+07
# IXIC_VOLUME            6.096828e+12
# IXIC_DAILY_RETURNS     1.072585e+02
# HSI_OPEN               3.866393e+07
# HSI_HIGH               3.891499e+07
# HSI_LOW                3.834805e+07
# HSI_CLOSE              3.864110e+07
# HSI_ADJ_CLOSE          3.864110e+07
# HSI_VOLUME             3.310837e+12
# HSI_DAILY_RETURNS     -2.015246e+01
# N225_OPEN              3.997675e+07
# N225_HIGH              4.018083e+07
# N225_LOW               3.975022e+07
# N225_CLOSE             3.997446e+07
# N225_ADJ_CLOSE         3.997446e+07
# N225_VOLUME            1.176986e+11
# N225_DAILY_RETURNS     3.346293e+01
# GDAXI_OPEN             2.122368e+07
# GDAXI_HIGH             2.135211e+07
# GDAXI_LOW              2.108441e+07
# GDAXI_CLOSE            2.122356e+07
# GDAXI_ADJ_CLOSE        2.122356e+07
# GDAXI_VOLUME           1.294603e+11
# GDAXI_DAILY_RETURNS    4.937069e+01
# VIX_OPEN               3.244029e+04
# VIX_HIGH               3.435547e+04
# VIX_LOW                3.057491e+04
# VIX_CLOSE              3.210560e+04
# VIX_ADJ_CLOSE          3.210560e+04
# VIX_VOLUME             0.000000e+00
# VIX_DAILY_RETURNS      4.201235e+02
