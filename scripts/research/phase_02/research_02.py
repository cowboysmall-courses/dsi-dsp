
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

from gsma.data.file import read_master_file



# %% 2 -
master = read_master_file()['2018-01-02':'2022-12-30']

for index, column in zip(INDICES[:-1], COLUMNS[:-1]):
    table = master.groupby("YEAR")[column].agg(['count', 'mean', 'std', 'var'])
    print(f"\n{index}\n\n{table}\n\n")
