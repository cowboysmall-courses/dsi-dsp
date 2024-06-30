
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

from cowboysmall.data.file import read_master_file
from cowboysmall.feature import INDICES, COLUMNS



# %% 2 -
master = read_master_file()



# %% 3 -
CONDITIONS = [(master.index <= '2020-01-30'), ('2022-05-05' <= master.index)]
CHOICES    = ['PRE_COVID', 'POST_COVID']

master['PANDEMIC'] = np.select(CONDITIONS, CHOICES, 'COVID')
master['PANDEMIC'] = pd.Categorical(master['PANDEMIC'], categories = ['PRE_COVID', 'COVID', 'POST_COVID'], ordered = True)



# %% 4 -
for index, column in zip(INDICES[:-1], COLUMNS[:-1]):
    table = master.groupby("PANDEMIC", observed = False)[column].agg(['count', 'mean', 'std', 'var'])
    print(f"\n{index}\n\n{table}\n\n")
