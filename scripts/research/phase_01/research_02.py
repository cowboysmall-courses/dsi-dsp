
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
from gsma import INDICES

from gsma.data.index import retrieve_data
from gsma.data.file import save_index_file



# %% 2 - retrieve data for indices
for index in INDICES:
    save_index_file(retrieve_data(index), index)
