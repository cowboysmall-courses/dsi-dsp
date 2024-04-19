
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
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

from statsmodels.formula.api import logit
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

from gsma import COLUMNS
from gsma.data.file import read_master_file
# from gsma.plots import plt, sns


# %% 2 -
def calculate_rsi(values, window = 14):
    delta_up = values.diff()
    delta_dn = delta_up.copy()

    delta_up[delta_up < 0] = 0
    delta_dn[delta_dn > 0] = 0

    mean_up = delta_up.rolling(window).mean()
    mean_dn = delta_dn.rolling(window).mean().abs()

    return (mean_up / (mean_up + mean_dn)) * 100



# %% 2 -
master = read_master_file()

master["NSEI_OPEN_DIR"] = np.where(master["NSEI_OPEN"] > master["NSEI_CLOSE"].shift(), 1, 0)

master["NSEI_HL_RATIO"] = master["NSEI_HIGH"] / master["NSEI_LOW"]
master["DJI_HL_RATIO"]  = master["DJI_HIGH"] / master["DJI_LOW"]

master["NSEI_RSI"]      = calculate_rsi(master["NSEI_CLOSE"])
master["DJI_RSI"]       = calculate_rsi(master["DJI_CLOSE"])



# %% 3 -
data = pd.concat([master["NSEI_OPEN_DIR"], master[COLUMNS].shift(), master["NSEI_HL_RATIO"].shift(), master["DJI_HL_RATIO"].shift(), master["NSEI_RSI"].shift(), master["DJI_RSI"].shift()], axis = 1)
data.dropna(inplace = True)
data.head()



# %% 4 -
train, test = train_test_split(data, test_size = 0.2)



# %% 5 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + DJI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + N225_DAILY_RETURNS + GDAXI_DAILY_RETURNS + VIX_DAILY_RETURNS + NSEI_HL_RATIO + DJI_HL_RATIO + NSEI_RSI + DJI_RSI', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1238
# Model:                          Logit   Df Residuals:                     1226
# Method:                           MLE   Df Model:                           11
# Date:                Fri, 19 Apr 2024   Pseudo R-squ.:                  0.1561
# Time:                        10:23:47   Log-Likelihood:                -647.59
# converged:                       True   LL-Null:                       -767.42
# Covariance Type:            nonrobust   LLR p-value:                 4.062e-45
# =======================================================================================
#                           coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# Intercept               8.7796     13.330      0.659      0.510     -17.346      34.905
# NSEI_DAILY_RETURNS     -0.2320      0.082     -2.824      0.005      -0.393      -0.071
# DJI_DAILY_RETURNS       0.1522      0.131      1.162      0.245      -0.105       0.409
# IXIC_DAILY_RETURNS      0.3595      0.092      3.889      0.000       0.178       0.541
# HSI_DAILY_RETURNS      -0.0903      0.056     -1.620      0.105      -0.200       0.019
# N225_DAILY_RETURNS     -0.1008      0.070     -1.441      0.150      -0.238       0.036
# GDAXI_DAILY_RETURNS     0.0654      0.079      0.831      0.406      -0.089       0.220
# VIX_DAILY_RETURNS      -0.0553      0.014     -3.850      0.000      -0.083      -0.027
# NSEI_HL_RATIO          10.4696     11.077      0.945      0.345     -11.241      32.181
# DJI_HL_RATIO          -18.5196     11.968     -1.547      0.122     -41.976       4.937
# NSEI_RSI                0.0036      0.004      0.803      0.422      -0.005       0.012
# DJI_RSI                 0.0014      0.005      0.276      0.783      -0.008       0.011
# =======================================================================================
# """



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                 Feature       VIF
# 0    NSEI_DAILY_RETURNS  1.412049
# 1     DJI_DAILY_RETURNS  4.341081
# 2    IXIC_DAILY_RETURNS  3.867615
# 3     HSI_DAILY_RETURNS  1.322373
# 4    N225_DAILY_RETURNS  1.333174
# 5   GDAXI_DAILY_RETURNS  1.780794
# 6     VIX_DAILY_RETURNS  2.131131
# 7         NSEI_HL_RATIO  1.546306
# 8          DJI_HL_RATIO  1.819724
# 9              NSEI_RSI  1.480878
# 10              DJI_RSI  1.537845



# %% 5 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + DJI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + N225_DAILY_RETURNS + GDAXI_DAILY_RETURNS + VIX_DAILY_RETURNS + NSEI_HL_RATIO + DJI_HL_RATIO + NSEI_RSI', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1238
# Model:                          Logit   Df Residuals:                     1227
# Method:                           MLE   Df Model:                           10
# Date:                Fri, 19 Apr 2024   Pseudo R-squ.:                  0.1561
# Time:                        10:27:55   Log-Likelihood:                -647.63
# converged:                       True   LL-Null:                       -767.42
# Covariance Type:            nonrobust   LLR p-value:                 8.359e-46
# =======================================================================================
#                           coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# Intercept               9.6155     12.996      0.740      0.459     -15.856      35.087
# NSEI_DAILY_RETURNS     -0.2334      0.082     -2.842      0.004      -0.394      -0.072
# DJI_DAILY_RETURNS       0.1582      0.129      1.224      0.221      -0.095       0.412
# IXIC_DAILY_RETURNS      0.3585      0.092      3.880      0.000       0.177       0.540
# HSI_DAILY_RETURNS      -0.0893      0.056     -1.606      0.108      -0.198       0.020
# N225_DAILY_RETURNS     -0.1003      0.070     -1.433      0.152      -0.237       0.037
# GDAXI_DAILY_RETURNS     0.0658      0.079      0.837      0.403      -0.088       0.220
# VIX_DAILY_RETURNS      -0.0551      0.014     -3.840      0.000      -0.083      -0.027
# NSEI_HL_RATIO          10.8812     10.984      0.991      0.322     -10.647      32.409
# DJI_HL_RATIO          -19.7085     11.164     -1.765      0.077     -41.589       2.172
# NSEI_RSI                0.0041      0.004      0.980      0.327      -0.004       0.012
# =======================================================================================
# """



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                Feature       VIF
# 0   NSEI_DAILY_RETURNS  1.403068
# 1    DJI_DAILY_RETURNS  4.276810
# 2   IXIC_DAILY_RETURNS  3.863694
# 3    HSI_DAILY_RETURNS  1.319353
# 4   N225_DAILY_RETURNS  1.329866
# 5  GDAXI_DAILY_RETURNS  1.780762
# 6    VIX_DAILY_RETURNS  2.123159
# 7        NSEI_HL_RATIO  1.509887
# 8         DJI_HL_RATIO  1.601013
# 9             NSEI_RSI  1.235125



# %% 5 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + DJI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + N225_DAILY_RETURNS + VIX_DAILY_RETURNS + NSEI_HL_RATIO + DJI_HL_RATIO + NSEI_RSI', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1238
# Model:                          Logit   Df Residuals:                     1228
# Method:                           MLE   Df Model:                            9
# Date:                Fri, 19 Apr 2024   Pseudo R-squ.:                  0.1556
# Time:                        10:30:06   Log-Likelihood:                -647.98
# converged:                       True   LL-Null:                       -767.42
# Covariance Type:            nonrobust   LLR p-value:                 2.206e-46
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept              9.6594     12.957      0.745      0.456     -15.737      35.055
# NSEI_DAILY_RETURNS    -0.2179      0.080     -2.732      0.006      -0.374      -0.062
# DJI_DAILY_RETURNS      0.1876      0.124      1.508      0.131      -0.056       0.431
# IXIC_DAILY_RETURNS     0.3576      0.092      3.873      0.000       0.177       0.539
# HSI_DAILY_RETURNS     -0.0844      0.055     -1.527      0.127      -0.193       0.024
# N225_DAILY_RETURNS    -0.0967      0.070     -1.388      0.165      -0.233       0.040
# VIX_DAILY_RETURNS     -0.0562      0.014     -3.936      0.000      -0.084      -0.028
# NSEI_HL_RATIO         10.3655     10.935      0.948      0.343     -11.066      31.797
# DJI_HL_RATIO         -19.2344     11.187     -1.719      0.086     -41.160       2.692
# NSEI_RSI               0.0040      0.004      0.970      0.332      -0.004       0.012
# ======================================================================================
# """



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.320207
# 1   DJI_DAILY_RETURNS  3.925444
# 2  IXIC_DAILY_RETURNS  3.861987
# 3   HSI_DAILY_RETURNS  1.302397
# 4  N225_DAILY_RETURNS  1.321922
# 5   VIX_DAILY_RETURNS  2.116157
# 6       NSEI_HL_RATIO  1.509864
# 7        DJI_HL_RATIO  1.600458
# 8            NSEI_RSI  1.234592



# %% 5 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + DJI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + N225_DAILY_RETURNS + VIX_DAILY_RETURNS + DJI_HL_RATIO + NSEI_RSI', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1238
# Model:                          Logit   Df Residuals:                     1229
# Method:                           MLE   Df Model:                            8
# Date:                Fri, 19 Apr 2024   Pseudo R-squ.:                  0.1551
# Time:                        10:32:13   Log-Likelihood:                -648.43
# converged:                       True   LL-Null:                       -767.42
# Covariance Type:            nonrobust   LLR p-value:                 6.053e-47
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             16.5456     10.693      1.547      0.122      -4.413      37.504
# NSEI_DAILY_RETURNS    -0.2170      0.079     -2.740      0.006      -0.372      -0.062
# DJI_DAILY_RETURNS      0.1862      0.124      1.504      0.132      -0.056       0.429
# IXIC_DAILY_RETURNS     0.3603      0.092      3.912      0.000       0.180       0.541
# HSI_DAILY_RETURNS     -0.0846      0.055     -1.530      0.126      -0.193       0.024
# N225_DAILY_RETURNS    -0.0955      0.070     -1.372      0.170      -0.232       0.041
# VIX_DAILY_RETURNS     -0.0561      0.014     -3.938      0.000      -0.084      -0.028
# DJI_HL_RATIO         -15.6319     10.510     -1.487      0.137     -36.230       4.966
# NSEI_RSI               0.0032      0.004      0.785      0.432      -0.005       0.011
# ======================================================================================
# """



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.316326
# 1   DJI_DAILY_RETURNS  3.925443
# 2  IXIC_DAILY_RETURNS  3.859033
# 3   HSI_DAILY_RETURNS  1.302385
# 4  N225_DAILY_RETURNS  1.313983
# 5   VIX_DAILY_RETURNS  2.109220
# 6        DJI_HL_RATIO  1.211758
# 7            NSEI_RSI  1.191334



# %% 5 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + DJI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + N225_DAILY_RETURNS + VIX_DAILY_RETURNS + DJI_HL_RATIO', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1238
# Model:                          Logit   Df Residuals:                     1230
# Method:                           MLE   Df Model:                            7
# Date:                Fri, 19 Apr 2024   Pseudo R-squ.:                  0.1547
# Time:                        10:33:34   Log-Likelihood:                -648.74
# converged:                       True   LL-Null:                       -767.42
# Covariance Type:            nonrobust   LLR p-value:                 1.350e-47
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             18.9862     10.206      1.860      0.063      -1.018      38.990
# NSEI_DAILY_RETURNS    -0.2047      0.077     -2.644      0.008      -0.356      -0.053
# DJI_DAILY_RETURNS      0.1877      0.124      1.518      0.129      -0.055       0.430
# IXIC_DAILY_RETURNS     0.3595      0.092      3.905      0.000       0.179       0.540
# HSI_DAILY_RETURNS     -0.0829      0.055     -1.502      0.133      -0.191       0.025
# N225_DAILY_RETURNS    -0.0952      0.070     -1.368      0.171      -0.231       0.041
# VIX_DAILY_RETURNS     -0.0552      0.014     -3.891      0.000      -0.083      -0.027
# DJI_HL_RATIO         -17.8662     10.091     -1.770      0.077     -37.645       1.912
# ======================================================================================
# """



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.269736
# 1   DJI_DAILY_RETURNS  3.925442
# 2  IXIC_DAILY_RETURNS  3.859020
# 3   HSI_DAILY_RETURNS  1.302193
# 4  N225_DAILY_RETURNS  1.313619
# 5   VIX_DAILY_RETURNS  2.093912
# 6        DJI_HL_RATIO  1.083848



# %% 5 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + DJI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + VIX_DAILY_RETURNS + DJI_HL_RATIO', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1238
# Model:                          Logit   Df Residuals:                     1231
# Method:                           MLE   Df Model:                            6
# Date:                Fri, 19 Apr 2024   Pseudo R-squ.:                  0.1534
# Time:                        10:37:27   Log-Likelihood:                -649.68
# converged:                       True   LL-Null:                       -767.42
# Covariance Type:            nonrobust   LLR p-value:                 5.155e-48
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             18.0390     10.125      1.782      0.075      -1.805      37.883
# NSEI_DAILY_RETURNS    -0.2289      0.076     -3.005      0.003      -0.378      -0.080
# DJI_DAILY_RETURNS      0.1599      0.122      1.312      0.190      -0.079       0.399
# IXIC_DAILY_RETURNS     0.3641      0.092      3.963      0.000       0.184       0.544
# HSI_DAILY_RETURNS     -0.1068      0.052     -2.039      0.041      -0.209      -0.004
# VIX_DAILY_RETURNS     -0.0563      0.014     -3.974      0.000      -0.084      -0.029
# DJI_HL_RATIO         -16.9297     10.010     -1.691      0.091     -36.550       2.691
# ======================================================================================
# """



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.229641
# 1   DJI_DAILY_RETURNS  3.826464
# 2  IXIC_DAILY_RETURNS  3.850225
# 3   HSI_DAILY_RETURNS  1.172667
# 4   VIX_DAILY_RETURNS  2.088912
# 5        DJI_HL_RATIO  1.081731



# %% 5 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + VIX_DAILY_RETURNS + DJI_HL_RATIO', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1238
# Model:                          Logit   Df Residuals:                     1232
# Method:                           MLE   Df Model:                            5
# Date:                Fri, 19 Apr 2024   Pseudo R-squ.:                  0.1523
# Time:                        10:38:31   Log-Likelihood:                -650.55
# converged:                       True   LL-Null:                       -767.42
# Covariance Type:            nonrobust   LLR p-value:                 1.694e-48
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             18.7477      9.842      1.905      0.057      -0.542      38.037
# NSEI_DAILY_RETURNS    -0.2153      0.075     -2.868      0.004      -0.363      -0.068
# IXIC_DAILY_RETURNS     0.4345      0.075      5.816      0.000       0.288       0.581
# HSI_DAILY_RETURNS     -0.1082      0.052     -2.070      0.038      -0.211      -0.006
# VIX_DAILY_RETURNS     -0.0622      0.013     -4.671      0.000      -0.088      -0.036
# DJI_HL_RATIO         -17.6271      9.730     -1.812      0.070     -36.697       1.443
# ======================================================================================
# """



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.193657
# 1  IXIC_DAILY_RETURNS  1.943311
# 2   HSI_DAILY_RETURNS  1.172249
# 3   VIX_DAILY_RETURNS  1.963871
# 4        DJI_HL_RATIO  1.081292














# %% 7 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + N225_DAILY_RETURNS + GDAXI_DAILY_RETURNS + VIX_DAILY_RETURNS + NSEI_HL_RATIO + DJI_HL_RATIO', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1249
# Model:                          Logit   Df Residuals:                     1240
# Method:                           MLE   Df Model:                            8
# Date:                Wed, 17 Apr 2024   Pseudo R-squ.:                  0.1533
# Time:                        11:29:28   Log-Likelihood:                -661.87
# converged:                       True   LL-Null:                       -781.69
# Covariance Type:            nonrobust   LLR p-value:                 2.711e-47
# =======================================================================================
#                           coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# Intercept              40.0827     10.309      3.888      0.000      19.878      60.287
# NSEI_DAILY_RETURNS     -0.2518      0.076     -3.320      0.001      -0.400      -0.103
# IXIC_DAILY_RETURNS      0.4403      0.074      5.937      0.000       0.295       0.586
# HSI_DAILY_RETURNS      -0.1375      0.055     -2.510      0.012      -0.245      -0.030
# N225_DAILY_RETURNS     -0.0788      0.066     -1.194      0.232      -0.208       0.051
# GDAXI_DAILY_RETURNS     0.0711      0.071      0.996      0.319      -0.069       0.211
# VIX_DAILY_RETURNS      -0.0534      0.013     -4.155      0.000      -0.079      -0.028
# NSEI_HL_RATIO         -13.6743      9.420     -1.452      0.147     -32.138       4.789
# DJI_HL_RATIO          -25.0859     10.497     -2.390      0.017     -45.659      -4.512
# =======================================================================================
# """



# %% 8 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                Feature       VIF
# 0   NSEI_DAILY_RETURNS  1.369319
# 1   IXIC_DAILY_RETURNS  2.263584
# 2    HSI_DAILY_RETURNS  1.332201
# 3   N225_DAILY_RETURNS  1.332324
# 4  GDAXI_DAILY_RETURNS  1.717464
# 5    VIX_DAILY_RETURNS  2.049962
# 6        NSEI_HL_RATIO  1.524168
# 7         DJI_HL_RATIO  1.592250



# %% 9 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + N225_DAILY_RETURNS + VIX_DAILY_RETURNS + NSEI_HL_RATIO + DJI_HL_RATIO', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1249
# Model:                          Logit   Df Residuals:                     1241
# Method:                           MLE   Df Model:                            7
# Date:                Wed, 17 Apr 2024   Pseudo R-squ.:                  0.1526
# Time:                        11:30:57   Log-Likelihood:                -662.37
# converged:                       True   LL-Null:                       -781.69
# Covariance Type:            nonrobust   LLR p-value:                 7.268e-48
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             39.8380     10.265      3.881      0.000      19.718      59.958
# NSEI_DAILY_RETURNS    -0.2345      0.073     -3.192      0.001      -0.378      -0.090
# IXIC_DAILY_RETURNS     0.4570      0.072      6.341      0.000       0.316       0.598
# HSI_DAILY_RETURNS     -0.1338      0.055     -2.453      0.014      -0.241      -0.027
# N225_DAILY_RETURNS    -0.0674      0.065     -1.042      0.298      -0.194       0.059
# VIX_DAILY_RETURNS     -0.0552      0.013     -4.358      0.000      -0.080      -0.030
# NSEI_HL_RATIO        -14.2323      9.417     -1.511      0.131     -32.689       4.224
# DJI_HL_RATIO         -24.2871     10.473     -2.319      0.020     -44.814      -3.761
# ======================================================================================
# """



# %% 10 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.268557
# 1  IXIC_DAILY_RETURNS  2.079045
# 2   HSI_DAILY_RETURNS  1.321853
# 3  N225_DAILY_RETURNS  1.291874
# 4   VIX_DAILY_RETURNS  2.007826
# 5       NSEI_HL_RATIO  1.519542
# 6        DJI_HL_RATIO  1.589102



# %% 7 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + VIX_DAILY_RETURNS + NSEI_HL_RATIO + DJI_HL_RATIO', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1249
# Model:                          Logit   Df Residuals:                     1242
# Method:                           MLE   Df Model:                            6
# Date:                Wed, 17 Apr 2024   Pseudo R-squ.:                  0.1519
# Time:                        11:32:38   Log-Likelihood:                -662.92
# converged:                       True   LL-Null:                       -781.69
# Covariance Type:            nonrobust   LLR p-value:                 1.876e-48
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             38.4641     10.013      3.841      0.000      18.839      58.089
# NSEI_DAILY_RETURNS    -0.2513      0.072     -3.480      0.001      -0.393      -0.110
# IXIC_DAILY_RETURNS     0.4534      0.072      6.282      0.000       0.312       0.595
# HSI_DAILY_RETURNS     -0.1519      0.052     -2.942      0.003      -0.253      -0.051
# VIX_DAILY_RETURNS     -0.0555      0.013     -4.377      0.000      -0.080      -0.031
# NSEI_HL_RATIO        -13.8623      9.252     -1.498      0.134     -31.997       4.272
# DJI_HL_RATIO         -23.3000     10.439     -2.232      0.026     -43.760      -2.840
# ======================================================================================
# """



# %% 10 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.233650
# 1  IXIC_DAILY_RETURNS  2.066982
# 2   HSI_DAILY_RETURNS  1.179232
# 3   VIX_DAILY_RETURNS  2.004351
# 4       NSEI_HL_RATIO  1.518209
# 5        DJI_HL_RATIO  1.577404



# %% 7 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + VIX_DAILY_RETURNS + DJI_HL_RATIO', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1249
# Model:                          Logit   Df Residuals:                     1243
# Method:                           MLE   Df Model:                            5
# Date:                Wed, 17 Apr 2024   Pseudo R-squ.:                  0.1506
# Time:                        11:33:51   Log-Likelihood:                -663.97
# converged:                       True   LL-Null:                       -781.69
# Covariance Type:            nonrobust   LLR p-value:                 7.327e-49
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             31.5643      9.412      3.354      0.001      13.118      50.010
# NSEI_DAILY_RETURNS    -0.2495      0.073     -3.421      0.001      -0.392      -0.107
# IXIC_DAILY_RETURNS     0.4389      0.072      6.099      0.000       0.298       0.580
# HSI_DAILY_RETURNS     -0.1480      0.052     -2.874      0.004      -0.249      -0.047
# VIX_DAILY_RETURNS     -0.0558      0.013     -4.409      0.000      -0.081      -0.031
# DJI_HL_RATIO         -30.3397      9.301     -3.262      0.001     -48.569     -12.110
# ======================================================================================
# """



# %% 10 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.229487
# 1  IXIC_DAILY_RETURNS  2.030198
# 2   HSI_DAILY_RETURNS  1.178408
# 3   VIX_DAILY_RETURNS  2.004127
# 4        DJI_HL_RATIO  1.072773




# %% 8 - ROC Curve
train['predicted'] = model.predict(train)

# plt.plot_setup()
# sns.sns_setup()

fpr, tpr, thresholds = roc_curve(train['NSEI_OPEN_DIR'], train['predicted'])

plt.figure(figsize = (8, 6))

plt.plot(fpr, tpr, label = 'ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label = 'Random Guess')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc = 'lower right')

plt.show()



# %% 9 - AUC Curve
auc_roc = roc_auc_score(train['NSEI_OPEN_DIR'], train['predicted'])
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.765442960985893



# %% 10 - Optimal Threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold is : {optimal_threshold}')
# Best Threshold is : 0.654



# %% 11 - Classification Report
train['predicted_class'] = (train['predicted'] > optimal_threshold).astype(int)
print(classification_report(train['NSEI_OPEN_DIR'], train['predicted_class']))
#               precision    recall  f1-score   support
# 
#            0       0.62      0.60      0.61       398
#            1       0.82      0.82      0.82       851
# 
#     accuracy                           0.75      1249
#    macro avg       0.72      0.71      0.71      1249
# weighted avg       0.75      0.75      0.75      1249



# %% 11 - 
table = pd.crosstab(np.where(train['predicted'] <= optimal_threshold,  0, 1), train['NSEI_OPEN_DIR'])
table
# NSEI_OPEN_DIR    0    1
# row_0                  
# 0              239  149
# 1              159  702



# %% 11 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.654 is : 82.49

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.654 is : 60.05



# %% 12 - ROC Curve
test['predicted'] = model.predict(test)

# plt.plot_setup()
# sns.sns_setup()

fpr, tpr, thresholds = roc_curve(test['NSEI_OPEN_DIR'], test['predicted'])

plt.figure(figsize = (8, 6))

plt.plot(fpr, tpr, label = 'ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label = 'Random Guess')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc = 'lower right')

plt.show()



# %% 13 - AUC Curve
auc_roc = roc_auc_score(test['NSEI_OPEN_DIR'], test['predicted'])
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7112676056338028



# %% 14 - Classification Report
test['predicted_class'] = (test['predicted'] > optimal_threshold).astype(int)
print(classification_report(test['NSEI_OPEN_DIR'], test['predicted_class']))
#               precision    recall  f1-score   support
# 
#            0       0.54      0.56      0.55       100
#            1       0.79      0.77      0.78       213
# 
#     accuracy                           0.71       313
#    macro avg       0.66      0.67      0.67       313
# weighted avg       0.71      0.71      0.71       313



# %% 11 - 
table = pd.crosstab(np.where(test['predicted'] <= optimal_threshold,  0, 1), test['NSEI_OPEN_DIR'])
table
# NSEI_OPEN_DIR   0    1
# row_0                 
# 0              56   48
# 1              44  165



# %% 11 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.654 is : 77.46

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.654 is : 56.0
