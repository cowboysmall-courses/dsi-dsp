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



# %% 0 - import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf


# %% 0 - list of indices
indices = ['NSEI', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']


# %% 1 - function to read index data
def read_file(index):
    data = pd.read_csv("./data/raw/{}.csv".format(index), index_col = 'Date')
    data.index = pd.to_datetime(data.index)
    return data


# %% 2 - 
plt.figure(figsize = (8, 6))
plt.style.use('ggplot')
sns.set_style("darkgrid")
sns.set_context("paper")


# %% 3 - 
for index in indices:
    data      = read_file(index)

    data['M'] = pd.PeriodIndex(data.index, freq = 'M')
    data['Q'] = pd.PeriodIndex(data.index, freq = 'Q')
    data['Y'] = pd.PeriodIndex(data.index, freq = 'Y')

    data      = data['2018-01-02':'2023-12-29']

    returns   = '{}_DAILY_RETURNS'.format(index)
    close     = '{}_CLOSE'.format(index)

    dmonth    = data.groupby('M')[[close]].sum()
    dquarter  = data.groupby('Q')[[close]].sum()


    plt.scatter(data.index, data[returns])
    plt.title('Plot: {}'.format(index))
    plt.xlabel('Date')
    plt.ylabel('Daily Returns')
    plt.savefig("./images/indices/scatter/{}.png".format(returns))
    plt.clf()


    sns.boxplot(x = 'Y', y = returns, data = data)
    plt.title('Box Plot: {}'.format(index))
    plt.xlabel('Year')
    plt.ylabel('Daily Returns')
    plt.savefig("./images/indices/boxplot/{}.png".format(returns))
    plt.clf()


    fig  = sm.qqplot(data[returns], line = '45', fit = True)
    fig.savefig("./images/indices/qqplots/{}.png".format(returns))
    fig.clf()
    plt.clf()



    sns.histplot(data[returns], kde = True)
    plt.title('Histogram: {}'.format(index))
    plt.xlabel('Daily Returns')
    plt.ylabel('Frequency')
    plt.savefig("./images/indices/histogram/{}.png".format(returns))
    plt.clf()



    sns.lineplot(x = data.index, y = data[close])
    plt.title('Line Plot: {}'.format(index))
    plt.xlabel('Day')
    plt.ylabel('Closing Price')
    plt.savefig("./images/indices/lineplots/{}_DAY.png".format(close))
    plt.clf()

    sns.lineplot(x = dmonth.index.month, y = dmonth[close])
    plt.title('Line Plot: {}'.format(index))
    plt.xlabel('Month')
    plt.ylabel('Closing Price')
    plt.savefig("./images/indices/lineplots/{}_MONTH.png".format(close))
    plt.clf()

    sns.lineplot(x = dquarter.index.quarter, y = dquarter[close])
    plt.title('Line Plot: {}'.format(index))
    plt.xlabel('Quarter')
    plt.ylabel('Closing Price')
    plt.savefig("./images/indices/lineplots/{}_QUARTER.png".format(close))
    plt.clf()



    fig = sm.tsa.seasonal_decompose(dmonth[close].values, period = 12).plot()
    fig.savefig("./images/indices/seasonal/{}_month.png".format(close))
    fig.clf()
    plt.clf()

    fig = sm.tsa.seasonal_decompose(dquarter[close].values, period = 4).plot()
    fig.savefig("./images/indices/seasonal/{}_quarter.png".format(close))
    fig.clf()
    plt.clf()


    fig = plot_acf(data[close].values)
    fig.savefig("./images/indices/correlogram/{}.png".format(close))
    fig.clf()
    plt.clf()
    



# %%
