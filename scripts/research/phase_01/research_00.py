
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


# %% 0 - 
import pandas as pd


# %% 0 - 
sales = pd.DataFrame({"CustomerId": [1, 1, 2, 3, 2, 3], "UnitPrice": [10, 10, 10, 10, 10, 10], "Quantity": [1, 2, 1, 3, 1, 4]})


# %% 0 - 
total_sales = sales.groupby('CustomerId')[['UnitPrice','Quantity']].sum().reset_index()
total_sales
