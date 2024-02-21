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


# %% 2 - save index information to file
indices[0].to_csv("../data/indices.csv")
