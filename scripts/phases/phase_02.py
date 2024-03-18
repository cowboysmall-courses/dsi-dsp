
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



# %% 0 - import required libraries
import pandas as pd
import numpy as np


from gsma.data.file import read_master_file
from gsma.plots import plt, sns




# %% 1 - list of indices
indices = ['NSEI', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']




# %% 2 - list of columns
columns = [f"{index}_DAILY_RETURNS" for index in indices]




# %% 3 - read master data
master = read_master_file()




# %% 4 - global indices 5 years performance analytics
master_5 = master['2018-01-02':'2022-12-30']

for index, column in zip(indices[:-1], columns[:-1]):
    table = master_5.groupby("YEAR")[column].agg(['count', 'mean', 'std', 'var'])
    print(f"\n{index}\n\n{table}\n\n")

    plt.plot_setup()
    sns.sns_setup()
    sns.box_plot(master_5["YEAR"], master_5[column], column, "Daily Returns", "YEAR", "Years", index, "phase_02")

    table = master_5.groupby("YEAR")[column].agg(['median'])
    plt.plot_setup()
    sns.sns_setup()
    sns.bar_plot(table.index, table["median"], column, "Median Daily Return", "YEAR", "Year", index, "phase_02")

    table = pd.pivot_table(master_5, values = column, index = ["YEAR"], columns = ["QUARTER"], aggfunc = "mean")
    plt.plot_setup()
    sns.sns_setup()
    sns.heat_map(table, column, "MEAN", "YEAR", index, "phase_02")

    table = pd.pivot_table(master_5, values = column, index = ["YEAR"], columns = ["QUARTER"], aggfunc = "median")
    plt.plot_setup()
    sns.sns_setup()
    sns.heat_map(table, column, "MEDIAN", "YEAR", index, "phase_02")




# %% 5 - global indices correlation analysis
matrix1 = master[columns]['2018-01-02':'2022-12-30'].corr()
plt.plot_setup()
sns.sns_setup()
sns.correlation_matrix(matrix1, "DAILY_RETURNS", "Daily Returns", "2018-2022", "phase_02")

matrix2 = master[columns]['2023-01-02':'2023-12-29'].corr()
plt.plot_setup()
sns.sns_setup()
sns.correlation_matrix(matrix2, "DAILY_RETURNS", "Daily Returns", "2023-2023", "phase_02")




# %% 6 - pre-post covid performance analytics
master['PANDEMIC'] = np.select(
    [
        master.index <= '2020-01-30',
        ('2020-01-31' <= master.index) & (master.index <= '2022-05-04'),
        '2022-05-05' <= master.index
    ],
    ['PRE_COVID', 'COVID', 'POST_COVID']
)

master['PANDEMIC'] = pd.Categorical(master['PANDEMIC'], categories = ['PRE_COVID', 'COVID', 'POST_COVID'], ordered = True)

for index, column in zip(indices[:-1], columns[:-1]):
    table = master.groupby("PANDEMIC")[column].agg(['count', 'mean', 'std', 'var'])
    print(f"\n{index}\n\n{table}\n\n")

    plt.plot_setup()
    sns.sns_setup()
    sns.box_plot(master["PANDEMIC"], master[column], column, "Daily Returns", "PANDEMIC", "Pandemic", index, "phase_02")

    table = master.groupby("PANDEMIC")[column].agg(['median'])
    plt.plot_setup()
    sns.sns_setup()
    sns.bar_plot(table.index, table["median"], column, "Median Daily Return", "PANDEMIC", "Pandemic", index, "phase_02")

    table = pd.pivot_table(master, values = column, index = ["PANDEMIC"], columns = ["QUARTER"], aggfunc = "mean")
    plt.plot_setup()
    sns.sns_setup()
    sns.heat_map(table, column, "MEAN", "PANDEMIC", index, "phase_02")

    table = pd.pivot_table(master, values = column, index = ["PANDEMIC"], columns = ["QUARTER"], aggfunc = "median")
    plt.plot_setup()
    sns.sns_setup()
    sns.heat_map(table, column, "MEDIAN", "PANDEMIC", index, "phase_02")




# %% 7 - nifty fifty daily movement - pre-modeling
master["NSEI_OPEN_DIR"] = np.where(master["NSEI_OPEN"] > master["NSEI_CLOSE"].shift(), 1, 0)

table1 = master.groupby("YEAR")[["NSEI_OPEN_DIR"]].sum()
table2 = master.groupby("YEAR")[["NSEI_OPEN_DIR"]].count()
table  = ((table1["NSEI_OPEN_DIR"] / table2["NSEI_OPEN_DIR"]) * 100).round(2)

print(f"\n{table}\n")

for index in indices:
    returns = f"{index}_DAILY_RETURNS"

    plt.plot_setup()
    sns.sns_setup()
    sns.box_plot(master["NSEI_OPEN_DIR"], master[returns].shift(), returns, "Daily Returns", "NSEI_OPEN_DIR", "NSEI Open Direction", index, "phase_02")
