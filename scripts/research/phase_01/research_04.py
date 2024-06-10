
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
from cowboysmall.data.file import read_index_file
from cowboysmall.si.tests import test_normality
from cowboysmall.feature import INDICES, COLUMNS



# %% 2 - test for normality of data
for index, column in zip(INDICES[:-1], COLUMNS[:-1]):
    test_normality(read_index_file(index), column, index)
