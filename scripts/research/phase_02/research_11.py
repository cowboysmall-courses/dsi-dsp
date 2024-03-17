
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
import numpy as np

from gsma.data.file import read_master_file
from gsma.plots     import plt, sns


# %% 0 - list of indices
indices = ['NSEI', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']


# %% 1 - 
master = read_master_file()

master["NSEI_OPEN_DIR"] = np.where(master["NSEI_OPEN"] > master["NSEI_CLOSE"].shift(), 1, 0)

for index in indices:
    returns = f"{index}_DAILY_RETURNS"

    plt.plot_setup()
    sns.sns_setup()
    sns.box_plot(master["NSEI_OPEN_DIR"], master[returns].shift(), returns, "Daily Returns", "NSEI_OPEN_DIR", "NSEI Open Direction", index, "phase_02")
