#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 20:54:21 2024

@author: jerry
"""



# %% 0 -import required libraries
import numpy as np 
import pandas as pd 
import yfinance as yf


# %% 1 - retrieve a list of indices
indices = pd.read_html('https://finance.yahoo.com/world-indices/')




# %% 2 - look t the head of the retrieved data
indices[0].head()
#   Symbol                          Name  ...  52 Week Range  Day Chart
# 0  ^GSPC                       S&P 500  ...            NaN        NaN
# 1   ^DJI  Dow Jones Industrial Average  ...            NaN        NaN
# 2  ^IXIC              NASDAQ Composite  ...            NaN        NaN
# 3   ^NYA           NYSE COMPOSITE (DJ)  ...            NaN        NaN
# 4   ^XAX     NYSE AMEX COMPOSITE INDEX  ...            NaN        NaN



# %% 3 - look at the shape of the retrieved data
indices[0].shape
# (36, 9)



# %% 3 - 
indices[0].to_excel("../data/raw/indices.xlsx")



# %% 4 - 
sorted(indices[0].Symbol)



# %% 5 - 
nsei = yf.Ticker('^NSEI')



# %% 6 - 
data = nsei.history(period = '1d', start = '2018-1-1', end = '2023-12-31')



# %% 7 - 
data.head()



# %% 8 - 
data.to_excel("../data/raw/nsei.xlsx")




