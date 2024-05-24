
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
import pandas as pd
import numpy as np

from cowboysmall.data.file import read_master_file



# %% 2 -
INDICES = ['NSEI', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']
COLUMNS = [f"{index}_DAILY_RETURNS" for index in INDICES]

master = read_master_file()

CONDITIONS = [(master.index <= '2020-01-30'), ('2022-05-05' <= master.index)]
CHOICES    = ['PRE_COVID', 'POST_COVID']

master['PANDEMIC'] = np.select(CONDITIONS, CHOICES, 'COVID')
master['PANDEMIC'] = pd.Categorical(master['PANDEMIC'], categories = ['PRE_COVID', 'COVID', 'POST_COVID'], ordered = True)

for index, column in zip(INDICES[:-1], COLUMNS[:-1]):
    pre_covid  = master.loc[(master['PANDEMIC'] == 'PRE_COVID'),  [column]]
    post_covid = master.loc[(master['PANDEMIC'] == 'POST_COVID'), [column]]

    mean_pre   = pre_covid.values.mean()
    post_count = np.where(post_covid[column].ge(mean_pre).values == True)[0][0]
    post_date  = post_covid.index[post_covid[column].ge(mean_pre)][0].date()

    print(f"{index.rjust(5)} returned to pre-covid levels (mean {mean_pre: 2.4f}) on {post_date} after {post_count} trading day(s)")
