
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
from gsma.data.index import retrieve_data
from gsma.data.file  import save_master_file
from gsma.data.master import merge_data

from gsma.si.tests  import test_normality




# %% 1 - list of indices
indices = ['NSEI', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']




# %% 2 - retrieve data for indices
datas = [retrieve_data(index) for index in indices]

for index, data in zip(indices, datas):
    test_normality(data, f"{index}_DAILY_RETURNS", index)




# %% 3 - merge data
save_master_file(merge_data(datas))
