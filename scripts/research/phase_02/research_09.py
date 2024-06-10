
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

from cowboysmall.data.file import read_master_file
from cowboysmall.plots import plt, sns
from cowboysmall.feature import INDICES, COLUMNS



# %% 2 -
master = read_master_file()

CONDITIONS = [(master.index <= '2020-01-30'), ('2022-05-05' <= master.index)]
CHOICES    = ['PRE_COVID', 'POST_COVID']

master['PANDEMIC'] = np.select(CONDITIONS, CHOICES, 'COVID')
master['PANDEMIC'] = pd.Categorical(master['PANDEMIC'], categories = ['PRE_COVID', 'COVID', 'POST_COVID'], ordered = True)

for index, column in zip(INDICES[:-1], COLUMNS[:-1]):
    table = pd.pivot_table(master, values = column, index = ["PANDEMIC"], columns = ["QUARTER"], aggfunc = "mean", observed = False)
    plt.plot_setup()
    sns.sns_setup()
    sns.heat_map(table, column, "MEAN", "PANDEMIC", index, "phase_02")

    table = pd.pivot_table(master, values = column, index = ["PANDEMIC"], columns = ["QUARTER"], aggfunc = "median", observed = False)
    plt.plot_setup()
    sns.sns_setup()
    sns.heat_map(table, column, "MEDIAN", "PANDEMIC", index, "phase_02")
