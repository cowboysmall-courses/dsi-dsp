
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
master = read_master_file()[[f"{index}_DAILY_RETURNS" for index in indices[:-1]]]

data1  = master['2018-01-02':'2020-01-30']
data2  = master['2020-01-31':'2022-05-04']
data3  = master['2022-05-05':'2022-12-30']

table1 = data1.agg(['count', 'mean', 'std', 'var'])
table2 = data2.agg(['count', 'mean', 'std', 'var'])
table3 = data3.agg(['count', 'mean', 'std', 'var'])

print()
print(table1)
print()
print(table2)
print()
print(table3)
print()
