#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 22:16:27 2024

@author: jerry
"""



# %% 0 -import required libraries
import pandas as pd 
import yfinance as yf
import statsmodels.api as sm

from scipy import stats
from functools import reduce


# %% 1 - function to retrieve data
def get_data(ticker, start_date = '2017-12-1', end_date = '2024-1-31', save_raw = True):
    data = yf.download('^{}'.format(ticker), start_date, end_date)
    data['Daily Returns'] = data.Close.pct_change() * 100

    data.columns = ['{}_{}'.format(ticker, '_'.join(c.upper() for c in column.split())) for column in data.columns]

    if save_raw:
        data.to_csv("../data/raw/{}.csv".format(ticker))
    return data


# %% 2 - NSEI
data_01 = get_data('NSEI')
data_01.head()
#                NSEI_OPEN     NSEI_HIGH  ...  NSEI_VOLUME  NSEI_DAILY_RETURNS
# Date                                    ...                                 
# 2017-12-01  10263.700195  10272.700195  ...       143400                 NaN
# 2017-12-04  10175.049805  10179.200195  ...       148600            0.058786
# 2017-12-05  10118.250000  10147.950195  ...       155500           -0.093802
# 2017-12-06  10088.799805  10104.200195  ...       166100           -0.732838
# 2017-12-07  10063.450195  10182.650391  ...       166200            1.220623

# [5 rows x 7 columns]

stats.shapiro(data_01['NSEI_DAILY_RETURNS'].dropna())
# ShapiroResult(statistic=0.8644470572471619, pvalue=2.743593253737161e-34)

sm.stats.diagnostic.lilliefors(data_01['NSEI_DAILY_RETURNS'].dropna())
# (0.08314006694420253, 0.0009999999999998899)

# as in both cases the p-value < 0.05, we reject the null hypothesis 
# that the sample is drawn from normal population


# %% 3 - GSPC
data_02 = get_data('GSPC')
data_02.head()
#               GSPC_OPEN    GSPC_HIGH  ...  GSPC_VOLUME  GSPC_DAILY_RETURNS
# Date                                  ...                                 
# 2017-12-01  2645.100098  2650.620117  ...   3950930000                 NaN
# 2017-12-04  2657.189941  2665.189941  ...   4025840000           -0.105216
# 2017-12-05  2639.780029  2648.719971  ...   3547570000           -0.373938
# 2017-12-06  2626.239990  2634.409912  ...   3253080000           -0.011411
# 2017-12-07  2628.379883  2640.989990  ...   3297060000            0.293236

# [5 rows x 7 columns]

stats.shapiro(data_02['GSPC_DAILY_RETURNS'].dropna())
# ShapiroResult(statistic=0.8794808387756348, pvalue=4.9357946172692266e-33)

sm.stats.diagnostic.lilliefors(data_02['GSPC_DAILY_RETURNS'].dropna())
# (0.0946955261005249, 0.0009999999999998899)

# as in both cases the p-value < 0.05, we reject the null hypothesis 
# that the sample is drawn from normal population


# %% 4 - DJI
data_03 = get_data('DJI')
data_03.head()
#                 DJI_OPEN      DJI_HIGH  ...  DJI_VOLUME  DJI_DAILY_RETURNS
# Date                                    ...                               
# 2017-12-01  24305.400391  24322.060547  ...   417910000                NaN
# 2017-12-04  24424.109375  24534.039062  ...   424250000           0.241259
# 2017-12-05  24335.009766  24349.740234  ...   371190000          -0.450432
# 2017-12-06  24171.900391  24229.349609  ...   312720000          -0.164307
# 2017-12-07  24116.599609  24262.880859  ...   319060000           0.292327

# [5 rows x 7 columns]

stats.shapiro(data_03['DJI_DAILY_RETURNS'].dropna())
# ShapiroResult(statistic=0.8422792553901672, pvalue=1.1548236514399918e-36)

sm.stats.diagnostic.lilliefors(data_03['DJI_DAILY_RETURNS'].dropna())
# (0.11062368219270749, 0.0009999999999998899)

# as in both cases the p-value < 0.05, we reject the null hypothesis 
# that the sample is drawn from normal population


# %% 5 - IXIC
data_04 = get_data('IXIC')
data_04.head()
#               IXIC_OPEN    IXIC_HIGH  ...  IXIC_VOLUME  IXIC_DAILY_RETURNS
# Date                                  ...                                 
# 2017-12-01  6844.040039  6872.169922  ...   2302140000                 NaN
# 2017-12-04  6897.129883  6899.229980  ...   2427170000           -1.054674
# 2017-12-05  6759.140137  6836.450195  ...   2085760000           -0.194235
# 2017-12-06  6742.069824  6787.419922  ...   1901770000            0.209546
# 2017-12-07  6785.740234  6829.290039  ...   1953740000            0.538045

# [5 rows x 7 columns]

stats.shapiro(data_04['IXIC_DAILY_RETURNS'].dropna())
# ShapiroResult(statistic=0.9358991384506226, pvalue=2.732241438907868e-25)

sm.stats.diagnostic.lilliefors(data_04['IXIC_DAILY_RETURNS'].dropna())
# (0.08214150147067645, 0.0009999999999998899)

# as in both cases the p-value < 0.05, we reject the null hypothesis 
# that the sample is drawn from normal population


# %% 6 - HSI
data_05 = get_data('HSI')
data_05.head()
#                 HSI_OPEN      HSI_HIGH  ...  HSI_VOLUME  HSI_DAILY_RETURNS
# Date                                    ...                               
# 2017-12-01  29261.310547  29345.970703  ...  2086681200                NaN
# 2017-12-04  28951.189453  29343.779297  ...  1624072100           0.220260
# 2017-12-05  28898.580078  29124.240234  ...  1439200300          -1.014056
# 2017-12-06  28850.359375  28929.210938  ...  2605434400          -2.142649
# 2017-12-07  28366.140625  28478.339844  ...  1782453900           0.277730

# [5 rows x 7 columns]

stats.shapiro(data_05['HSI_DAILY_RETURNS'].dropna())
# ShapiroResult(statistic=0.9711626172065735, pvalue=7.557595223594841e-17)

sm.stats.diagnostic.lilliefors(data_05['HSI_DAILY_RETURNS'].dropna())
# (0.04982628106789255, 0.0009999999999998899)

# as in both cases the p-value < 0.05, we reject the null hypothesis 
# that the sample is drawn from normal population


# %% 7 - N225
data_06 = get_data('N225')
data_06.head()
#                N225_OPEN     N225_HIGH  ...  N225_VOLUME  N225_DAILY_RETURNS
# Date                                    ...                                 
# 2017-12-01  22916.929688  22994.310547  ...     89700000                 NaN
# 2017-12-04  22843.529297  22864.330078  ...     68900000           -0.490245
# 2017-12-05  22595.330078  22682.710938  ...     75900000           -0.373359
# 2017-12-06  22525.380859  22528.210938  ...     97300000           -1.968589
# 2017-12-07  22317.150391  22515.240234  ...     79500000            1.447399

# [5 rows x 7 columns]

stats.shapiro(data_06['N225_DAILY_RETURNS'].dropna())
# ShapiroResult(statistic=0.9678009748458862, pvalue=8.65255570712721e-18)

sm.stats.diagnostic.lilliefors(data_06['N225_DAILY_RETURNS'].dropna())
# (0.05388381718318283, 0.0009999999999998899)

# as in both cases the p-value < 0.05, we reject the null hypothesis 
# that the sample is drawn from normal population


# %% 8 - GDAXI
data_07 = get_data('GDAXI')
data_07.head()
#               GDAXI_OPEN    GDAXI_HIGH  ...  GDAXI_VOLUME  GDAXI_DAILY_RETURNS
# Date                                    ...                                   
# 2017-12-01  13044.150391  13064.290039  ...     114375600                  NaN
# 2017-12-04  13038.769531  13117.750000  ...      85813900             1.532167
# 2017-12-05  13056.820312  13094.379883  ...      81417900            -0.076653
# 2017-12-06  12897.429688  13033.750000  ...      84631500            -0.380812
# 2017-12-07  13026.299805  13083.080078  ...      76453600             0.356191

# [5 rows x 7 columns]

stats.shapiro(data_07['GDAXI_DAILY_RETURNS'].dropna())
# ShapiroResult(statistic=0.89853835105896, pvalue=6.147803011156481e-31)

sm.stats.diagnostic.lilliefors(data_07['GDAXI_DAILY_RETURNS'].dropna())
# (0.09171583385249443, 0.0009999999999998899)

# as in both cases the p-value < 0.05, we reject the null hypothesis 
# that the sample is drawn from normal population


# %% 9 - VIX
data_08 = get_data('VIX')
data_08.head()
#             VIX_OPEN  VIX_HIGH  ...  VIX_VOLUME  VIX_DAILY_RETURNS
# Date                            ...                               
# 2017-12-01     11.19     14.58  ...           0                NaN
# 2017-12-04     11.05     11.86  ...           0           2.187227
# 2017-12-05     11.38     11.67  ...           0          -2.996579
# 2017-12-06     11.63     11.68  ...           0          -2.736094
# 2017-12-07     10.90     11.32  ...           0          -7.803998

# [5 rows x 7 columns]

stats.shapiro(data_08['VIX_DAILY_RETURNS'].dropna())
# ShapiroResult(statistic=0.8391106724739075, pvalue=6.112061224258144e-37)

sm.stats.diagnostic.lilliefors(data_08['VIX_DAILY_RETURNS'].dropna())
# (0.10895884465554928, 0.0009999999999998899)

# as in both cases the p-value < 0.05, we reject the null hypothesis 
# that the sample is drawn from normal population


# %% 10 - merge data, forward-fill missing values, and save relevant data
data = [data_01, data_02, data_03, data_04, data_05, data_06, data_07, data_08]

merged = reduce(lambda  l, r: pd.merge(l, r, on = ['Date'], how = 'outer'), data)
merged.fillna(method = 'ffill', inplace = True)

merged['MONTH']   = pd.PeriodIndex(merged.index, freq = 'M')
merged['QUARTER'] = pd.PeriodIndex(merged.index, freq = 'Q')
merged['YEAR']    = pd.PeriodIndex(merged.index, freq = 'Y')

merged = merged['2018-01-01':'2023-12-31']
merged.head()

merged.to_csv("../data/processed/master_data.csv")
