
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
from cowboysmall.data.index import retrieve_data
from cowboysmall.data.file import save_index_file
from cowboysmall.feature import INDICES



# %% 2 - retrieve data for indices
for index in INDICES:
    save_index_file(retrieve_data(index), index)
