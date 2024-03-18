
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
from gsma.data.file import read_master_file



# %% 2 - read master data and sum columns
print(read_master_file().iloc[:, :-3].sum(numeric_only = True))
