
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
from gsma.data.file import read_index_file
from gsma.plots     import plt, sns


# %% 0 - list of indices
indices = ['NSEI', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']


# %% 1 - 
for index in indices[:-1]:
    data     = read_index_file(index, indicators = True)
    data     = data['2018-01-02':'2022-12-30']

    returns  = f"{index}_DAILY_RETURNS"

    table    = data.groupby("YEAR")[returns].agg(['median'])

    plt.plot_setup()
    sns.sns_setup()
    sns.bar_plot(table.index, table["median"], returns, "Median Daily Return", "YEAR", "Year", index, "phase_02")
