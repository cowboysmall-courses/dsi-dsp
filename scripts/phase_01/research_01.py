
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


# %% 1 - retrieve a list of indices
indices = pd.read_html('https://finance.yahoo.com/world-indices/')


# %% 2 - save index information to file
indices[0].to_csv("./data/indices.csv")
