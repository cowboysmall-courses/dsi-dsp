
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
import pandas as pd

from gsma.data.file import read_master_file
from gsma.plots     import plots


# %% 0 - list of indices
indices = ['NSEI', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']


# %% 1 - 
master  = read_master_file()[[f"{index}_DAILY_RETURNS" for index in indices[:-1]]]

matrix1 = master['2018-01-02':'2022-12-30'].corr()
matrix2 = master['2023-01-02':'2023-12-29'].corr()

plots.plot_setup()
plots.sns_setup()
plots.correlation_matrix(matrix1, "DAILY_RETURNS", "Daily Returns", "2018-2022", "phase_02")

plots.plot_setup()
plots.sns_setup()
plots.correlation_matrix(matrix2, "DAILY_RETURNS", "Daily Returns", "2023-2023", "phase_02")

