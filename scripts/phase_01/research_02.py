
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
from gsma.plots import plots


# %% 0 - list of indices
indices = ['NSEI', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']


# %% 2 -plot daily returns
plots.plot_setup()

for index in indices:
    data     = read_index_file(index, indicators = True)
    data     = data['2018-01-02':'2023-12-29']

    returns  = f"{index}_DAILY_RETURNS"

    # plots on daily returns
    plots.qq_plot(data, returns)
    plots.box_plot(data["YEAR"], data[returns], returns, "Daily Returns", "YEAR", "Years", index)
    plots.histogram(data, returns, "Daily Returns", index)
