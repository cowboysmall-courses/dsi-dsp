#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 20:54:21 2024

@author: jerry
"""



# %% 0 -import required libraries
import pandas as pd 



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



# %% 3 - look at the number of rows in the retrieved data
indices[0].shape[0]
# 36



# %% 3 - save index information to file
indices[0].to_csv("../data/raw/indices.csv")



# %% 4 - print a comma separated list of index names
print(', '.join(index for index in sorted(indices[0].Symbol)))
