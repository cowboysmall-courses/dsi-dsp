
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



# close    = f"{index}_CLOSE"

# dmonth   = data.groupby("MONTH")[[close]].sum()
# dquarter = data.groupby("QUARTER")[[close]].sum()

# # plots on closing prices
# plots.correlogram(data, close)

# plots.seasonal_plot(dmonth, close, 12, 'MONTH')
# plots.seasonal_plot(dquarter, close, 4, 'QUARTER')

# plots.line_plot(data.index, data[close], close, 'Closing Price', 'DAY', 'Day', index)
# plots.line_plot(dmonth.index.month, dmonth[close], 'Closing Price', 'MONTH', 'Month', index)
# plots.line_plot(dquarter.index.quarter, dquarter[close], 'Closing Price', 'QUARTER', 'Quarter', index)
