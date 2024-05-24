
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



# %% 2 -
INDICES = ['NSEI', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']
COLUMNS = [f"{index}_DAILY_RETURNS" for index in INDICES]

master = read_master_file()['2018-01-02':'2022-12-30']

for index, column in zip(INDICES[:-1], COLUMNS[:-1]):
    table = master.groupby("YEAR")[column].agg(['median'])
    plt.plot_setup()
    sns.sns_setup()
    sns.bar_plot(table.index, table["median"], column, "Median Daily Return", "YEAR", "Year", index, "phase_02")
