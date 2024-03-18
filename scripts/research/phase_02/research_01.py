
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


# %% 1 - list of indices
indices = ['NSEI', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']


# %% 2 - plot daily returns
master = read_master_file()['2018-01-02':'2022-12-30']

for index in indices[:-1]:
    returns = f"{index}_DAILY_RETURNS"

    plt.plot_setup()
    sns.sns_setup()
    sns.box_plot(master["YEAR"], master[returns], returns, "Daily Returns", "YEAR", "Years", index, "phase_02")