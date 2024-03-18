
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
from gsma.plots import plt, sns, sms



# %% 2 - plot daily returns
for index, column in zip(INDICES[:-1], COLUMNS[:-1]):
    data = read_index_file(index, indicators = True)['2018-01-02':'2023-12-29']

    plt.plot_setup()
    sns.sns_setup()
    sms.qq_plot(data, column, "phase_01")

    plt.plot_setup()
    sns.sns_setup()
    sns.histogram(data, column, "Daily Returns", index, "phase_01")
