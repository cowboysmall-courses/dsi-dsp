
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
for index in indices[:-1]:
    data     = read_index_file(index, indicators = True)
    data     = data['2018-01-02':'2023-12-29']

    returns  = f"{index}_DAILY_RETURNS"

    plots.plot_setup()
    plots.sns_setup()
    plots.qq_plot(data, returns, "phase_01")

    plots.plot_setup()
    plots.sns_setup()
    plots.histogram(data, returns, "Daily Returns", index, "phase_01")
