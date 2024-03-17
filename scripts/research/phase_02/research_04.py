
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

from gsma.data.file import read_master_file
from gsma.plots     import plt, sns


# %% 0 - list of indices
indices = ['NSEI', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']


# %% 1 - 
master = read_master_file()

for index in indices[:-1]:
    master  = master['2018-01-02':'2022-12-30']

    returns = f"{index}_DAILY_RETURNS"

    table1  = pd.pivot_table(master, values = returns, index = ["YEAR"], columns = ["QUARTER"], aggfunc = "mean") 
    plt.plot_setup()
    sns.sns_setup()
    sns.heat_map(table1, returns, "MEAN", "YEAR", index, "phase_02")

    table2  = pd.pivot_table(master, values = returns, index = ["YEAR"], columns = ["QUARTER"], aggfunc = "median") 
    plt.plot_setup()
    sns.sns_setup()
    sns.heat_map(table2, returns, "MEDIAN", "YEAR", index, "phase_02")
