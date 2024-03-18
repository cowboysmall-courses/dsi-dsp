
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



# %% 2 -
master = read_master_file()

CONDITIONS = [(master.index <= '2020-01-30'), ('2022-05-05' <= master.index)]
CHOICES    = ['PRE_COVID', 'POST_COVID']

master['PANDEMIC'] = np.select(CONDITIONS, CHOICES, 'COVID')
master['PANDEMIC'] = pd.Categorical(master['PANDEMIC'], categories = ['PRE_COVID', 'COVID', 'POST_COVID'], ordered = True)

for index, column in zip(INDICES[:-1], COLUMNS[:-1]):
    plt.plot_setup()
    sns.sns_setup()
    sns.box_plot(master["PANDEMIC"], master[column], column, "Daily Returns", "PANDEMIC", "Pandemic", index, "phase_02")
