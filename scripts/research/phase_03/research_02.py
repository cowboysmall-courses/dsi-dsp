
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

from statsmodels.api import Logit
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from cowboysmall.data.file import read_master_file
from cowboysmall.feature import COLUMNS, INDICATORS, RATIOS
from cowboysmall.feature.indicators import get_indicators, get_ratios
from cowboysmall.plots import plt, sns



# %% 2 -
ALL_COLS = COLUMNS + RATIOS + INDICATORS



# %% 2 -
master = read_master_file()



# %% 2 -
master["NSEI_OPEN_DIR"] = np.where(master["NSEI_OPEN"] > master["NSEI_CLOSE"].shift(), 1, 0)



# %% 2 -
master = get_ratios(master)
master = get_indicators(master)



# %% 2 -
counts = master['NSEI_OPEN_DIR'].value_counts().reset_index()
counts.columns = ['NSEI_OPEN_DIR', 'Freq']
print(counts)
#    NSEI_OPEN_DIR  Freq
# 0              1  1064
# 1              0   499



# %% 2 -
print((counts["Freq"][0] / (counts["Freq"][0] + counts["Freq"][1])).round(3))
# 0.681



# %% 3 -
data = pd.concat([master["NSEI_OPEN_DIR"].shift(-1), master[ALL_COLS]], axis = 1)
data.dropna(inplace = True)
data.head()



# %% 3 -
X = data[ALL_COLS]
y = data['NSEI_OPEN_DIR']



# %% 3 -
X.insert(loc = 0, column = "Intercept", value = 1)



# %% 4 -
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337)



# %% 4 -
X_dropped = []



# %% 5 -
model = Logit(y_train, X_train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1206
# Method:                           MLE   Df Model:                           13
# Date:                Mon, 10 Jun 2024   Pseudo R-squ.:                  0.1461
# Time:                        22:53:38   Log-Likelihood:                -653.47
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 1.944e-40
# =======================================================================================
#                           coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# Intercept              10.4306     12.655      0.824      0.410     -14.373      35.234
# NSEI_DAILY_RETURNS     -0.1465      0.089     -1.638      0.101      -0.322       0.029
# DJI_DAILY_RETURNS      -0.0461      0.130     -0.354      0.723      -0.301       0.209
# IXIC_DAILY_RETURNS      0.4359      0.094      4.649      0.000       0.252       0.620
# HSI_DAILY_RETURNS      -0.1100      0.056     -1.982      0.047      -0.219      -0.001
# N225_DAILY_RETURNS     -0.1594      0.071     -2.251      0.024      -0.298      -0.021
# GDAXI_DAILY_RETURNS     0.0451      0.077      0.587      0.557      -0.106       0.196
# VIX_DAILY_RETURNS      -0.0403      0.013     -2.984      0.003      -0.067      -0.014
# NSEI_HL_RATIO           9.0971     10.913      0.834      0.404     -12.292      30.486
# DJI_HL_RATIO          -20.2324     12.120     -1.669      0.095     -43.987       3.523
# NSEI_RSI               -0.0196      0.015     -1.334      0.182      -0.048       0.009
# DJI_RSI                 0.0532      0.015      3.477      0.001       0.023       0.083
# NSEI_TSI                0.0137      0.008      1.707      0.088      -0.002       0.029
# DJI_TSI                -0.0298      0.009     -3.365      0.001      -0.047      -0.012
# =======================================================================================
# """

# Drop DJI_DAILY_RETURNS



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                 Feature       VIF
# 0    NSEI_DAILY_RETURNS  1.859291
# 1     DJI_DAILY_RETURNS  4.528768
# 2    IXIC_DAILY_RETURNS  4.006814
# 3     HSI_DAILY_RETURNS  1.369066
# 4    N225_DAILY_RETURNS  1.453563
# 5   GDAXI_DAILY_RETURNS  1.971029
# 6     VIX_DAILY_RETURNS  2.124774
# 7         NSEI_HL_RATIO  1.619396
# 8          DJI_HL_RATIO  2.017921
# 9              NSEI_RSI  7.920968
# 10              DJI_RSI  6.239617
# 11             NSEI_TSI  7.183497
# 12              DJI_TSI  5.739381



# %% 6 -
X_train = X_train.drop("DJI_DAILY_RETURNS", axis = 1)
X_dropped.append("DJI_DAILY_RETURNS")



# %% 7 -
model = Logit(y_train, X_train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1207
# Method:                           MLE   Df Model:                           12
# Date:                Mon, 10 Jun 2024   Pseudo R-squ.:                  0.1460
# Time:                        22:55:47   Log-Likelihood:                -653.53
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 4.659e-41
# =======================================================================================
#                           coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# Intercept              10.1455     12.783      0.794      0.427     -14.908      35.199
# NSEI_DAILY_RETURNS     -0.1483      0.090     -1.656      0.098      -0.324       0.027
# IXIC_DAILY_RETURNS      0.4168      0.077      5.446      0.000       0.267       0.567
# HSI_DAILY_RETURNS      -0.1071      0.055     -1.952      0.051      -0.215       0.000
# N225_DAILY_RETURNS     -0.1613      0.071     -2.280      0.023      -0.300      -0.023
# GDAXI_DAILY_RETURNS     0.0369      0.074      0.501      0.616      -0.107       0.181
# VIX_DAILY_RETURNS      -0.0394      0.013     -2.960      0.003      -0.066      -0.013
# NSEI_HL_RATIO           9.2936     10.951      0.849      0.396     -12.171      30.758
# DJI_HL_RATIO          -20.1028     12.178     -1.651      0.099     -43.972       3.766
# NSEI_RSI               -0.0190      0.015     -1.299      0.194      -0.048       0.010
# DJI_RSI                 0.0517      0.015      3.516      0.000       0.023       0.081
# NSEI_TSI                0.0134      0.008      1.677      0.093      -0.002       0.029
# DJI_TSI                -0.0291      0.009     -3.370      0.001      -0.046      -0.012
# =======================================================================================
# """

# Drop GDAXI_DAILY_RETURNS



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                 Feature       VIF
# 0    NSEI_DAILY_RETURNS  1.853563
# 1    IXIC_DAILY_RETURNS  2.246942
# 2     HSI_DAILY_RETURNS  1.337539
# 3    N225_DAILY_RETURNS  1.439436
# 4   GDAXI_DAILY_RETURNS  1.762197
# 5     VIX_DAILY_RETURNS  2.097088
# 6         NSEI_HL_RATIO  1.619354
# 7          DJI_HL_RATIO  2.017823
# 8              NSEI_RSI  7.845808
# 9               DJI_RSI  5.969188
# 10             NSEI_TSI  7.124829
# 11              DJI_TSI  5.583324



# %% 6 -
X_train = X_train.drop("GDAXI_DAILY_RETURNS", axis = 1)
X_dropped.append("GDAXI_DAILY_RETURNS")



# %% 7 -
model = Logit(y_train, X_train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1208
# Method:                           MLE   Df Model:                           11
# Date:                Mon, 10 Jun 2024   Pseudo R-squ.:                  0.1458
# Time:                        22:57:35   Log-Likelihood:                -653.65
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 1.135e-41
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             10.5044     12.702      0.827      0.408     -14.391      35.400
# NSEI_DAILY_RETURNS    -0.1361      0.086     -1.582      0.114      -0.305       0.033
# IXIC_DAILY_RETURNS     0.4232      0.075      5.621      0.000       0.276       0.571
# HSI_DAILY_RETURNS     -0.1046      0.055     -1.914      0.056      -0.212       0.003
# N225_DAILY_RETURNS    -0.1592      0.070     -2.258      0.024      -0.297      -0.021
# VIX_DAILY_RETURNS     -0.0406      0.013     -3.098      0.002      -0.066      -0.015
# NSEI_HL_RATIO          8.5950     10.838      0.793      0.428     -12.648      29.838
# DJI_HL_RATIO         -19.7622     12.156     -1.626      0.104     -43.587       4.062
# NSEI_RSI              -0.0198      0.015     -1.360      0.174      -0.048       0.009
# DJI_RSI                0.0525      0.015      3.593      0.000       0.024       0.081
# NSEI_TSI               0.0137      0.008      1.720      0.085      -0.002       0.029
# DJI_TSI               -0.0294      0.009     -3.404      0.001      -0.046      -0.012
# ======================================================================================
# """

# Drop NSEI_HL_RATIO



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                Feature       VIF
# 0   NSEI_DAILY_RETURNS  1.697582
# 1   IXIC_DAILY_RETURNS  2.111415
# 2    HSI_DAILY_RETURNS  1.322058
# 3   N225_DAILY_RETURNS  1.426486
# 4    VIX_DAILY_RETURNS  2.043059
# 5        NSEI_HL_RATIO  1.605852
# 6         DJI_HL_RATIO  2.013707
# 7             NSEI_RSI  7.763424
# 8              DJI_RSI  5.921066
# 9             NSEI_TSI  7.093119
# 10             DJI_TSI  5.570420



# %% 6 -
X_train = X_train.drop("NSEI_HL_RATIO", axis = 1)
X_dropped.append("NSEI_HL_RATIO")



# %% 7 -
model = Logit(y_train, X_train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1209
# Method:                           MLE   Df Model:                           10
# Date:                Mon, 10 Jun 2024   Pseudo R-squ.:                  0.1454
# Time:                        22:59:01   Log-Likelihood:                -653.97
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 3.162e-42
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             15.1718     11.370      1.334      0.182      -7.113      37.457
# NSEI_DAILY_RETURNS    -0.1276      0.086     -1.490      0.136      -0.295       0.040
# IXIC_DAILY_RETURNS     0.4239      0.075      5.653      0.000       0.277       0.571
# HSI_DAILY_RETURNS     -0.1064      0.055     -1.946      0.052      -0.214       0.001
# N225_DAILY_RETURNS    -0.1589      0.070     -2.256      0.024      -0.297      -0.021
# VIX_DAILY_RETURNS     -0.0406      0.013     -3.105      0.002      -0.066      -0.015
# DJI_HL_RATIO         -15.7066     11.080     -1.418      0.156     -37.422       6.009
# NSEI_RSI              -0.0220      0.014     -1.548      0.122      -0.050       0.006
# DJI_RSI                0.0534      0.015      3.657      0.000       0.025       0.082
# NSEI_TSI               0.0142      0.008      1.790      0.073      -0.001       0.030
# DJI_TSI               -0.0294      0.009     -3.402      0.001      -0.046      -0.012
# ======================================================================================
# """

# Drop DJI_HL_RATIO



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.677361
# 1  IXIC_DAILY_RETURNS  2.110037
# 2   HSI_DAILY_RETURNS  1.319835
# 3  N225_DAILY_RETURNS  1.425697
# 4   VIX_DAILY_RETURNS  2.042696
# 5        DJI_HL_RATIO  1.515720
# 6            NSEI_RSI  7.476355
# 7             DJI_RSI  5.891491
# 8            NSEI_TSI  7.058562
# 9             DJI_TSI  5.569180



# %% 6 -
X_train = X_train.drop("DJI_HL_RATIO", axis = 1)
X_dropped.append("DJI_HL_RATIO")



# %% 7 -
model = Logit(y_train, X_train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1210
# Method:                           MLE   Df Model:                            9
# Date:                Mon, 10 Jun 2024   Pseudo R-squ.:                  0.1441
# Time:                        23:00:29   Log-Likelihood:                -654.98
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 1.645e-42
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             -0.9075      0.782     -1.160      0.246      -2.441       0.626
# NSEI_DAILY_RETURNS    -0.1267      0.084     -1.503      0.133      -0.292       0.038
# IXIC_DAILY_RETURNS     0.4385      0.075      5.827      0.000       0.291       0.586
# HSI_DAILY_RETURNS     -0.1114      0.055     -2.037      0.042      -0.219      -0.004
# N225_DAILY_RETURNS    -0.1564      0.070     -2.220      0.026      -0.294      -0.018
# VIX_DAILY_RETURNS     -0.0414      0.013     -3.159      0.002      -0.067      -0.016
# NSEI_RSI              -0.0216      0.014     -1.524      0.127      -0.049       0.006
# DJI_RSI                0.0562      0.014      3.902      0.000       0.028       0.084
# NSEI_TSI               0.0149      0.008      1.880      0.060      -0.001       0.030
# DJI_TSI               -0.0285      0.009     -3.313      0.001      -0.045      -0.012
# ======================================================================================
# """

# Drop NSEI_DAILY_RETURNS



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.672722
# 1  IXIC_DAILY_RETURNS  2.102327
# 2   HSI_DAILY_RETURNS  1.317628
# 3  N225_DAILY_RETURNS  1.425582
# 4   VIX_DAILY_RETURNS  2.019710
# 5            NSEI_RSI  7.474704
# 6             DJI_RSI  5.839974
# 7            NSEI_TSI  7.012844
# 8             DJI_TSI  5.492200



# %% 6 -
X_train = X_train.drop("NSEI_DAILY_RETURNS", axis = 1)
X_dropped.append("NSEI_DAILY_RETURNS")



# %% 7 -
model = Logit(y_train, X_train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1211
# Method:                           MLE   Df Model:                            8
# Date:                Mon, 10 Jun 2024   Pseudo R-squ.:                  0.1426
# Time:                        23:01:51   Log-Likelihood:                -656.13
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 9.260e-43
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             -0.5565      0.748     -0.744      0.457      -2.022       0.909
# IXIC_DAILY_RETURNS     0.4321      0.075      5.766      0.000       0.285       0.579
# HSI_DAILY_RETURNS     -0.1266      0.054     -2.362      0.018      -0.232      -0.022
# N225_DAILY_RETURNS    -0.1797      0.068     -2.626      0.009      -0.314      -0.046
# VIX_DAILY_RETURNS     -0.0396      0.013     -3.034      0.002      -0.065      -0.014
# NSEI_RSI              -0.0319      0.012     -2.557      0.011      -0.056      -0.007
# DJI_RSI                0.0593      0.014      4.146      0.000       0.031       0.087
# NSEI_TSI               0.0199      0.007      2.753      0.006       0.006       0.034
# DJI_TSI               -0.0301      0.009     -3.519      0.000      -0.047      -0.013
# ======================================================================================
# """



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  IXIC_DAILY_RETURNS  2.094539
# 1   HSI_DAILY_RETURNS  1.258623
# 2  N225_DAILY_RETURNS  1.364805
# 3   VIX_DAILY_RETURNS  1.994389
# 4            NSEI_RSI  5.771417
# 5             DJI_RSI  5.713214
# 6            NSEI_TSI  5.865700
# 7             DJI_TSI  5.390966

# Drop NSEI_TSI



# %% 6 -
X_train = X_train.drop("NSEI_TSI", axis = 1)
X_dropped.append("NSEI_TSI")



# %% 7 -
model = Logit(y_train, X_train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1212
# Method:                           MLE   Df Model:                            7
# Date:                Mon, 10 Jun 2024   Pseudo R-squ.:                  0.1375
# Time:                        23:04:17   Log-Likelihood:                -659.98
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 6.810e-42
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             -1.3452      0.689     -1.953      0.051      -2.695       0.005
# IXIC_DAILY_RETURNS     0.4555      0.075      6.095      0.000       0.309       0.602
# HSI_DAILY_RETURNS     -0.1387      0.053     -2.610      0.009      -0.243      -0.035
# N225_DAILY_RETURNS    -0.1952      0.068     -2.884      0.004      -0.328      -0.063
# VIX_DAILY_RETURNS     -0.0395      0.013     -3.040      0.002      -0.065      -0.014
# NSEI_RSI              -0.0016      0.006     -0.281      0.778      -0.013       0.010
# DJI_RSI                0.0453      0.013      3.416      0.001       0.019       0.071
# DJI_TSI               -0.0203      0.008     -2.634      0.008      -0.035      -0.005
# ======================================================================================
# """

# Drop NSEI_RSI



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  IXIC_DAILY_RETURNS  2.074798
# 1   HSI_DAILY_RETURNS  1.249875
# 2  N225_DAILY_RETURNS  1.355220
# 3   VIX_DAILY_RETURNS  1.994111
# 4            NSEI_RSI  1.310004
# 5             DJI_RSI  4.986308
# 6             DJI_TSI  4.416925



# %% 6 -
X_train = X_train.drop("NSEI_RSI", axis = 1)
X_dropped.append("NSEI_RSI")



# %% 7 -
model = Logit(y_train, X_train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1213
# Method:                           MLE   Df Model:                            6
# Date:                Mon, 10 Jun 2024   Pseudo R-squ.:                  0.1375
# Time:                        23:05:57   Log-Likelihood:                -660.02
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 1.141e-42
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             -1.4041      0.656     -2.139      0.032      -2.690      -0.118
# IXIC_DAILY_RETURNS     0.4552      0.075      6.093      0.000       0.309       0.602
# HSI_DAILY_RETURNS     -0.1395      0.053     -2.632      0.008      -0.243      -0.036
# N225_DAILY_RETURNS    -0.1960      0.068     -2.897      0.004      -0.329      -0.063
# VIX_DAILY_RETURNS     -0.0397      0.013     -3.054      0.002      -0.065      -0.014
# DJI_RSI                0.0447      0.013      3.415      0.001       0.019       0.070
# DJI_TSI               -0.0205      0.008     -2.660      0.008      -0.036      -0.005
# ======================================================================================
# """



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  IXIC_DAILY_RETURNS  2.073867
# 1   HSI_DAILY_RETURNS  1.244922
# 2  N225_DAILY_RETURNS  1.353286
# 3   VIX_DAILY_RETURNS  1.994009
# 4             DJI_RSI  4.850250
# 5             DJI_TSI  4.379409





# %% 8 - ROC Curve
y_train_pred_prob = model.predict(X_train)

fpr, tpr, thresholds = roc_curve(y_train, y_train_pred_prob)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "Logistic Model")



# %% 9 - Optimal Threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold is : {optimal_threshold}')
# Best Threshold is : 0.684



# %% 10 - AUC Curve
auc_roc = roc_auc_score(y_train, y_train_pred_prob)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7529115595469844



# %% 11 - Classification Report
y_train_pred_class = np.where(y_train_pred_prob <= optimal_threshold,  0, 1)
print(classification_report(y_train, y_train_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.53      0.68      0.60       391
#          1.0       0.83      0.72      0.77       829
# 
#     accuracy                           0.70      1220
#    macro avg       0.68      0.70      0.68      1220
# weighted avg       0.73      0.70      0.71      1220



# %% 11 - 
table = pd.crosstab(y_train_pred_class, y_train)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0              265  234
# 1              126  595



# %% 11 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)

print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Sensitivity for cut-off 0.684 is : 71.77%
# Specificity for cut-off 0.684 is : 67.77%



# %% 12 - 
X_test = X_test.drop(X_dropped, axis = 1)



# %% 12 - ROC Curve
y_test_pred_prob = model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_prob)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "Logistic Model")



# %% 13 - AUC Curve
auc_roc = roc_auc_score(y_test, y_test_pred_prob)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7520816812053925



# %% 14 - Classification Report
y_test_pred_class = np.where(y_test_pred_prob <= optimal_threshold,  0, 1)
print(classification_report(y_test, y_test_pred_class))
#               precision    recall  f1-score   support

#          0.0       0.53      0.65      0.58        97
#          1.0       0.82      0.73      0.77       208

#     accuracy                           0.70       305
#    macro avg       0.67      0.69      0.67       305
# weighted avg       0.72      0.70      0.71       305



# %% 11 - 
table = pd.crosstab(y_test_pred_class, y_test)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0               63   57
# 1               34  151



# %% 11 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)

print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Sensitivity for cut-off 0.684 is : 72.6%
# Specificity for cut-off 0.684 is : 64.95%
