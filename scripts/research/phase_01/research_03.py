
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
from cowboysmall.data.file import read_index_file
from cowboysmall.plots import plt, sns, sms
from cowboysmall.feature import INDICES, COLUMNS



# %% 2 - plot daily returns
plt.plot_setup()
sns.sns_setup()

for index, column in zip(INDICES[:-1], COLUMNS[:-1]):
    data = read_index_file(index, indicators = True)['2018-01-02':'2023-12-29']
    sms.qq_plot(data[column].values)
    sns.histogram(data[column].values, "Daily Returns", 'Frequency', index)
