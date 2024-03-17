
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
from gsma.data.file import read_master_file


# %% 1 - list of indices
indices = ['NSEI', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']


# %% 2 - 
master = read_master_file()

for index in indices[:-1]:
    master  = master['2018-01-02':'2022-12-30']

    returns = f"{index}_DAILY_RETURNS"

    table   = master.groupby("YEAR")[returns].agg(['count', 'mean', 'std', 'var'])

    print()
    print(index)
    print()
    print(table)
    print()
