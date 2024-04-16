
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

from statsmodels.formula.api import logit
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

from gsma import COLUMNS
from gsma.data.file import read_master_file




# %% 2 -
master = read_master_file()
master["NSEI_OPEN_DIR"] = np.where(master["NSEI_OPEN"] > master["NSEI_CLOSE"].shift(), 1, 0)



# %% 3 -
data = pd.concat([master[COLUMNS].shift(), master["NSEI_OPEN_DIR"]], axis = 1)[1:]
data.head()



# %% 4 -
train, test = train_test_split(data, test_size = 0.2)



# %% 5 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + DJI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + N225_DAILY_RETURNS + GDAXI_DAILY_RETURNS + VIX_DAILY_RETURNS', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1249
# Model:                          Logit   Df Residuals:                     1241
# Method:                           MLE   Df Model:                            7
# Date:                Tue, 16 Apr 2024   Pseudo R-squ.:                  0.1537
# Time:                        21:40:05   Log-Likelihood:                -653.59
# converged:                       True   LL-Null:                       -772.30
# Covariance Type:            nonrobust   LLR p-value:                 1.314e-47
# =======================================================================================
#                           coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# Intercept               0.9440      0.070     13.437      0.000       0.806       1.082
# NSEI_DAILY_RETURNS     -0.2060      0.079     -2.598      0.009      -0.361      -0.051
# DJI_DAILY_RETURNS       0.1851      0.128      1.445      0.149      -0.066       0.436
# IXIC_DAILY_RETURNS      0.4815      0.093      5.202      0.000       0.300       0.663
# HSI_DAILY_RETURNS      -0.0977      0.057     -1.720      0.085      -0.209       0.014
# N225_DAILY_RETURNS     -0.1383      0.073     -1.892      0.058      -0.281       0.005
# GDAXI_DAILY_RETURNS     0.0137      0.080      0.172      0.864      -0.142       0.170
# VIX_DAILY_RETURNS      -0.0423      0.014     -3.090      0.002      -0.069      -0.015
# =======================================================================================
# """



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                Feature       VIF
# 0   NSEI_DAILY_RETURNS  1.371808
# 1    DJI_DAILY_RETURNS  4.302031
# 2   IXIC_DAILY_RETURNS  3.813379
# 3    HSI_DAILY_RETURNS  1.344366
# 4   N225_DAILY_RETURNS  1.378738
# 5  GDAXI_DAILY_RETURNS  1.977443
# 6    VIX_DAILY_RETURNS  2.172571

