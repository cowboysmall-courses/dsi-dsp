
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
from cowboysmall.data.file import read_master_file
from cowboysmall.plots import plt, sns
from cowboysmall.feature import INDICES, COLUMNS



# %% 2 -
master = read_master_file()['2018-01-02':'2022-12-30']

for index, column in zip(INDICES[:-1], COLUMNS[:-1]):
    table = master.groupby("YEAR")[column].agg(['median'])
    plt.plot_setup()
    sns.sns_setup()
    sns.bar_plot(table.index, table["median"], column, "Median Daily Return", "YEAR", "Year", index, "phase_02")
