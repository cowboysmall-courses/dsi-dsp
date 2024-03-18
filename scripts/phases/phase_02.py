
"""

Global market indices of interest:

    NSEI:  Nifty 50
    DJI:   Dow Jones Index
    IXIC:  Nasdaq
    HSI:   Hang Seng
    N225:  Nikkei 225
    GDAXI: Dax
    VIX:   Volatility Index

"""



# %% 1 - import required libraries
import pandas as pd
import numpy as np

from gsma import INDICES, COLUMNS

from gsma.data.file import read_master_file
from gsma.plots import plt, sns




# %% 2 - read master data
master = read_master_file()

CONDITIONS = [(master.index <= '2020-01-30'), ('2022-05-05' <= master.index)]
CHOICES    = ['PRE_COVID', 'POST_COVID']

master['PANDEMIC'] = np.select(CONDITIONS, CHOICES, 'COVID')
master['PANDEMIC'] = pd.Categorical(master['PANDEMIC'], categories = ['PRE_COVID', 'COVID', 'POST_COVID'], ordered = True)

master["NSEI_OPEN_DIR"] = np.where(master["NSEI_OPEN"] > master["NSEI_CLOSE"].shift(), 1, 0)




# %% 3 - global indices 5 years performance analytics
print("\n5 Years Performance Analytics\n")

master_5 = master['2018-01-02':'2022-12-30']
for index, column in zip(INDICES[:-1], COLUMNS[:-1]):
    table = master_5.groupby("YEAR", observed = False)[column].agg(['count', 'mean', 'std', 'var'])
    print(f"\n{index}\n\n{table}\n\n")

    plt.plot_setup()
    sns.sns_setup()
    sns.box_plot(master_5["YEAR"], master_5[column], column, "Daily Returns", "YEAR", "Years", index, "phase_02")

    table = master_5.groupby("YEAR", observed = False)[column].agg(['median'])
    plt.plot_setup()
    sns.sns_setup()
    sns.bar_plot(table.index, table["median"], column, "Median Daily Return", "YEAR", "Year", index, "phase_02")

    table = pd.pivot_table(master_5, values = column, index = ["YEAR"], columns = ["QUARTER"], aggfunc = "mean", observed = False)
    plt.plot_setup()
    sns.sns_setup()
    sns.heat_map(table, column, "MEAN", "YEAR", index, "phase_02")

    table = pd.pivot_table(master_5, values = column, index = ["YEAR"], columns = ["QUARTER"], aggfunc = "median", observed = False)
    plt.plot_setup()
    sns.sns_setup()
    sns.heat_map(table, column, "MEDIAN", "YEAR", index, "phase_02")




# %% 4 - global indices correlation analysis
matrix = master[COLUMNS]['2018-01-02':'2022-12-30'].corr()
plt.plot_setup()
sns.sns_setup()
sns.correlation_matrix(matrix, "DAILY_RETURNS", "Daily Returns", "2018-2022", "phase_02")

matrix = master[COLUMNS]['2023-01-02':'2023-12-29'].corr()
plt.plot_setup()
sns.sns_setup()
sns.correlation_matrix(matrix, "DAILY_RETURNS", "Daily Returns", "2023-2023", "phase_02")




# %% 5 - pre-post covid performance analytics
print("\nPre-Post Covid Performance Analytics\n")

for index, column in zip(INDICES[:-1], COLUMNS[:-1]):
    table = master.groupby("PANDEMIC", observed = False)[column].agg(['count', 'mean', 'std', 'var'])
    print(f"\n{index}\n\n{table}\n\n")

    plt.plot_setup()
    sns.sns_setup()
    sns.box_plot(master["PANDEMIC"], master[column], column, "Daily Returns", "PANDEMIC", "Pandemic", index, "phase_02")

    table = master.groupby("PANDEMIC", observed = False)[column].agg(['median'])
    plt.plot_setup()
    sns.sns_setup()
    sns.bar_plot(table.index, table["median"], column, "Median Daily Return", "PANDEMIC", "Pandemic", index, "phase_02")

    table = pd.pivot_table(master, values = column, index = ["PANDEMIC"], columns = ["QUARTER"], aggfunc = "mean", observed = False)
    plt.plot_setup()
    sns.sns_setup()
    sns.heat_map(table, column, "MEAN", "PANDEMIC", index, "phase_02")

    table = pd.pivot_table(master, values = column, index = ["PANDEMIC"], columns = ["QUARTER"], aggfunc = "median", observed = False)
    plt.plot_setup()
    sns.sns_setup()
    sns.heat_map(table, column, "MEDIAN", "PANDEMIC", index, "phase_02")




# %% 6 - nifty fifty daily movement - pre-modeling
table1 = master.groupby("YEAR", observed = False)[["NSEI_OPEN_DIR"]].sum()
table2 = master.groupby("YEAR", observed = False)[["NSEI_OPEN_DIR"]].count()
table  = ((table1["NSEI_OPEN_DIR"] / table2["NSEI_OPEN_DIR"]) * 100).round(2)

print("\nNifty Fifty Daily Movement\n")
print(f"\n{table}\n")

for index, column in zip(INDICES, COLUMNS):
    plt.plot_setup()
    sns.sns_setup()
    sns.box_plot(master["NSEI_OPEN_DIR"], master[column].shift(), column, "Daily Returns", "NSEI_OPEN_DIR", "NSEI Open Direction", index, "phase_02")
