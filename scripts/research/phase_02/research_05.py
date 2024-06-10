
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
from cowboysmall.feature import COLUMNS



# %% 2 -
master = read_master_file()[COLUMNS[:-1]]

matrix = master['2018-01-02':'2022-12-30'].corr()
plt.plot_setup()
sns.sns_setup()
sns.correlation_matrix(matrix, "DAILY_RETURNS", "Daily Returns", "2018-2022", "phase_02")

matrix = master['2023-01-02':'2023-12-29'].corr()
plt.plot_setup()
sns.sns_setup()
sns.correlation_matrix(matrix, "DAILY_RETURNS", "Daily Returns", "2023-2023", "phase_02")
