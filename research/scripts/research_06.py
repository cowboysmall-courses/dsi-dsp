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
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf


# %% 0 - list of indices
indices = ['NSEI', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']


# %% 1 -
plt.figure(figsize = (8, 6))
plt.style.use("ggplot")

sns.set_style("darkgrid")
sns.set_context("paper")


# %% 2 - various functions
def read_file(index):
    data = pd.read_csv("./data/raw/{}.csv".format(index), index_col = 'Date')
    data.index = pd.to_datetime(data.index)
    return data


def scatter_plot(data, column, column_name, index):
    plt.clf()
    plt.scatter(data.index, data[column])
    plt.title('Scatter Plot: {}'.format(index))
    plt.xlabel('Date')
    plt.ylabel(column_name)
    plt.savefig("./images/indices/scatter/{}.png".format(column))
    plt.close()


def box_plot(data, column, column_name, index):
    plt.clf()
    sns.boxplot(x = 'Y', y = returns, data = data)
    plt.title('Box Plot: {}'.format(index))
    plt.xlabel('Year')
    plt.ylabel(column_name)
    plt.savefig("./images/indices/boxplot/{}.png".format(column))
    plt.close()


def qq_plot(data, column):
    plt.clf()
    fig = sm.qqplot(data[column], line = '45', fit = True)
    fig.savefig("./images/indices/qqplots/{}.png".format(column))
    plt.close()


def histogram(data, column, column_name, index):
    plt.clf()
    sns.histplot(data[column], kde = True)
    plt.title('Histogram: {}'.format(index))
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.savefig("./images/indices/histogram/{}.png".format(column))
    plt.close()


def line_plot(x_vals, data, column, column_name, interval, interval_name, index):
    plt.clf()
    sns.lineplot(x = x_vals, y = data[close])
    plt.title('Line Plot: {}'.format(index))
    plt.xlabel(interval_name)
    plt.ylabel(column_name)
    plt.savefig("./images/indices/lineplots/{}_{}.png".format(column, interval))
    plt.close()


def seasonal_plot(data, column, p_name, p_value):
    plt.clf()
    fig = sm.tsa.seasonal_decompose(data[column].values, period = p_value).plot()
    fig.savefig("./images/indices/seasonal/{}_{}.png".format(column, p_name))
    plt.close()


def correlogram(data, column):
    plt.clf()
    fig = plot_acf(data[column].values)
    fig.savefig("./images/indices/correlogram/{}.png".format(column))
    plt.close()


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


    # plots on daily returns
    qq_plot(data, returns)
    scatter_plot(data, returns, 'Daily Returns', index)
    box_plot(data, returns, 'Daily Returns', index)
    histogram(data, returns, 'Daily Returns', index)


    # plots on closing prices
    correlogram(data, close)

    seasonal_plot(dmonth, close, 'M', 12)
    seasonal_plot(dquarter, close, 'Q', 4)

    line_plot(data.index, data, close, 'Closing Price', 'D', 'Day', index)
    line_plot(dmonth.index.month, dmonth, close, 'Closing Price', 'M', 'Month', index)
    line_plot(dquarter.index.quarter, dquarter, close, 'Closing Price', 'Q', 'Quarter', index)


# %%
