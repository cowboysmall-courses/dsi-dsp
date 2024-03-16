
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
from gsma.plots import plt, sns


# %% 0 - list of indices
indices = ['NSEI', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']


# %% 1 - 
master = read_master_file()[[f"{index}_DAILY_RETURNS" for index in indices[:-1]]]

data1  = master['2018-01-02':'2020-01-30']
data2  = master['2020-01-31':'2022-05-04']
data3  = master['2022-05-05':'2022-12-30']

table1 = data1.agg(['median'])
table2 = data2.agg(['median'])
table3 = data3.agg(['median'])

plt.plot_setup()
sns.sns_setup()
sns.simple_bar_plot(table1, "DAILY_RETURNS", "Daily Returns", "PRE_COVID", "phase_02")

plt.plot_setup()
sns.sns_setup()
sns.simple_bar_plot(table2, "DAILY_RETURNS", "Daily Returns", "COVID", "phase_02")

plt.plot_setup()
sns.sns_setup()
sns.simple_bar_plot(table3, "DAILY_RETURNS", "Daily Returns", "POST_COVID", "phase_02")
