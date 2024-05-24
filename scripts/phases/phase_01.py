
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
from cowboysmall import INDICES, COLUMNS

from cowboysmall.data.index import retrieve_data
from cowboysmall.data.file import save_master_file
from cowboysmall.data.master import merge_data

from cowboysmall.si.tests import test_normality




# %% 2 - retrieve data for indices
data = []

for index, column in zip(INDICES, COLUMNS):
    retrieved = retrieve_data(index)
    test_normality(retrieved, column, index)
    data.append(retrieved)




# %% 3 - merge data
save_master_file(merge_data(data))
