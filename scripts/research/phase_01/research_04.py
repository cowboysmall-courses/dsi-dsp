
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
from gsma import INDICES, COLUMNS

from gsma.data.file import read_index_file
from gsma.si.tests import test_normality



# %% 2 - test for normality of data
for index, column in zip(INDICES[:-1], COLUMNS[:-1]):
    test_normality(read_index_file(index), column, index)
