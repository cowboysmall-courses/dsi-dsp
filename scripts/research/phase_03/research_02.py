
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
from cowboysmall.feature.indicators import get_indicators, INDICATORS
from cowboysmall.plots import plt, sns



# %% 2 -
INDICES  = ['NSEI', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']
COLUMNS  = [f"{index}_DAILY_RETURNS" for index in INDICES]
RATIOS   = ["NSEI_HL_RATIO", "DJI_HL_RATIO"]

ALL_COLS = COLUMNS + RATIOS + INDICATORS



# %% 2 -
master = read_master_file()



# %% 2 -
master["NSEI_OPEN_DIR"] = np.where(master["NSEI_OPEN"] > master["NSEI_CLOSE"].shift(), 1, 0)



# %% 2 -
master["NSEI_HL_RATIO"] = master["NSEI_HIGH"] / master["NSEI_LOW"]
master["DJI_HL_RATIO"]  = master["DJI_HIGH"] / master["DJI_LOW"]



# %% 2 -
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
# Model:                          Logit   Df Residuals:                     1193
# Method:                           MLE   Df Model:                           26
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1574
# Time:                        15:33:24   Log-Likelihood:                -644.79
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 1.061e-36
# =======================================================================================
#                           coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# NSEI_DAILY_RETURNS     -0.0988      0.095     -1.039      0.299      -0.285       0.088
# DJI_DAILY_RETURNS      -0.0942      0.132     -0.714      0.475      -0.353       0.164
# IXIC_DAILY_RETURNS      0.4269      0.095      4.492      0.000       0.241       0.613
# HSI_DAILY_RETURNS      -0.1204      0.056     -2.139      0.032      -0.231      -0.010
# N225_DAILY_RETURNS     -0.1727      0.072     -2.408      0.016      -0.313      -0.032
# GDAXI_DAILY_RETURNS     0.0400      0.077      0.518      0.604      -0.111       0.191
# VIX_DAILY_RETURNS      -0.0388      0.014     -2.793      0.005      -0.066      -0.012
# NSEI_HL_RATIO           8.2042     10.513      0.780      0.435     -12.401      28.809
# DJI_HL_RATIO          -10.9730     10.512     -1.044      0.297     -31.577       9.631
# NSEI_RSI               -0.0362      0.022     -1.670      0.095      -0.079       0.006
# DJI_RSI                 0.0869      0.022      3.967      0.000       0.044       0.130
# NSEI_ROC               -0.0238      0.050     -0.477      0.633      -0.122       0.074
# DJI_ROC                 0.0377      0.047      0.804      0.421      -0.054       0.130
# NSEI_AWE                0.0005      0.001      0.535      0.593      -0.001       0.002
# DJI_AWE                -0.0006      0.001     -0.877      0.381      -0.002       0.001
# NSEI_KAM               -0.0006      0.001     -0.819      0.413      -0.002       0.001
# DJI_KAM                 0.0005      0.000      1.286      0.198      -0.000       0.001
# NSEI_TSI                0.0222      0.014      1.559      0.119      -0.006       0.050
# DJI_TSI                -0.0477      0.016     -3.006      0.003      -0.079      -0.017
# NSEI_VPT            -2.837e-06      3e-06     -0.945      0.345   -8.72e-06    3.05e-06
# DJI_VPT              2.369e-10   2.98e-09      0.080      0.937   -5.59e-09    6.07e-09
# NSEI_ULC               -0.0689      0.087     -0.791      0.429      -0.240       0.102
# DJI_ULC                -0.0433      0.088     -0.492      0.623      -0.216       0.129
# NSEI_SMA               -0.0036      0.002     -1.691      0.091      -0.008       0.001
# DJI_SMA                 0.0021      0.001      2.170      0.030       0.000       0.004
# NSEI_EMA                0.0041      0.002      1.896      0.058      -0.000       0.008
# DJI_EMA                -0.0025      0.001     -2.423      0.015      -0.005      -0.000
# =======================================================================================
# """

# Drop DJI_VPT



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                 Feature           VIF
# 0    NSEI_DAILY_RETURNS      2.167613
# 1     DJI_DAILY_RETURNS      4.730156
# 2    IXIC_DAILY_RETURNS      4.045504
# 3     HSI_DAILY_RETURNS      1.394200
# 4    N225_DAILY_RETURNS      1.498701
# 5   GDAXI_DAILY_RETURNS      2.014786
# 6     VIX_DAILY_RETURNS      2.237653
# 7         NSEI_HL_RATIO      2.515305
# 8          DJI_HL_RATIO      2.908842
# 9              NSEI_RSI     16.513897
# 10              DJI_RSI     12.205718
# 11             NSEI_ROC      7.299266
# 12              DJI_ROC      6.761201
# 13             NSEI_AWE     30.430109
# 14              DJI_AWE     32.571359
# 15             NSEI_KAM   1416.550011
# 16              DJI_KAM    407.941128
# 17             NSEI_TSI     22.043991
# 18              DJI_TSI     18.072143
# 19             NSEI_VPT     18.774015
# 20              DJI_VPT      2.935953
# 21             NSEI_ULC     10.157709
# 22              DJI_ULC     11.227513
# 23             NSEI_SMA  10699.059432
# 24              DJI_SMA   3247.844625
# 25             NSEI_EMA  11363.054183
# 26              DJI_EMA   3529.861003



# %% 6 -
X_train = X_train.drop("DJI_VPT", axis = 1)
X_dropped.append("DJI_VPT")



# %% 7 -
model = Logit(y_train, X_train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1193
# Method:                           MLE   Df Model:                           26
# Date:                Sun, 26 May 2024   Pseudo R-squ.:                  0.1582
# Time:                        12:35:52   Log-Likelihood:                -644.17
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 6.072e-37
# =======================================================================================
#                           coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# Intercept              19.3641     17.312      1.119      0.263     -14.566      53.294
# NSEI_DAILY_RETURNS     -0.0961      0.095     -1.014      0.310      -0.282       0.090
# DJI_DAILY_RETURNS      -0.0926      0.131     -0.708      0.479      -0.349       0.164
# IXIC_DAILY_RETURNS      0.4247      0.095      4.480      0.000       0.239       0.611
# HSI_DAILY_RETURNS      -0.1186      0.056     -2.108      0.035      -0.229      -0.008
# N225_DAILY_RETURNS     -0.1718      0.072     -2.394      0.017      -0.312      -0.031
# GDAXI_DAILY_RETURNS     0.0352      0.077      0.454      0.650      -0.117       0.187
# VIX_DAILY_RETURNS      -0.0371      0.014     -2.670      0.008      -0.064      -0.010
# NSEI_HL_RATIO          -0.4257     13.016     -0.033      0.974     -25.936      25.084
# DJI_HL_RATIO          -21.6080     14.173     -1.525      0.127     -49.387       6.171
# NSEI_RSI               -0.0379      0.022     -1.754      0.079      -0.080       0.004
# DJI_RSI                 0.0866      0.022      3.973      0.000       0.044       0.129
# NSEI_ROC               -0.0234      0.049     -0.474      0.635      -0.120       0.073
# DJI_ROC                 0.0329      0.047      0.700      0.484      -0.059       0.125
# NSEI_AWE                0.0005      0.001      0.551      0.581      -0.001       0.002
# DJI_AWE                -0.0005      0.001     -0.788      0.431      -0.002       0.001
# NSEI_KAM               -0.0006      0.001     -0.810      0.418      -0.002       0.001
# DJI_KAM                 0.0005      0.000      1.344      0.179      -0.000       0.001
# NSEI_TSI                0.0229      0.014      1.613      0.107      -0.005       0.051
# DJI_TSI                -0.0479      0.016     -3.050      0.002      -0.079      -0.017
# NSEI_VPT            -3.029e-06   2.92e-06     -1.036      0.300   -8.76e-06     2.7e-06
# NSEI_ULC               -0.0449      0.089     -0.504      0.614      -0.219       0.130
# DJI_ULC                -0.0168      0.086     -0.194      0.846      -0.186       0.153
# NSEI_SMA               -0.0033      0.002     -1.541      0.123      -0.007       0.001
# DJI_SMA                 0.0023      0.001      2.329      0.020       0.000       0.004
# NSEI_EMA                0.0038      0.002      1.733      0.083      -0.000       0.008
# DJI_EMA                -0.0026      0.001     -2.581      0.010      -0.005      -0.001
# =======================================================================================
# """

# Drop NSEI_HL_RATIO



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                 Feature           VIF
# 0    NSEI_DAILY_RETURNS      2.165674
# 1     DJI_DAILY_RETURNS      4.670037
# 2    IXIC_DAILY_RETURNS      4.041491
# 3     HSI_DAILY_RETURNS      1.393412
# 4    N225_DAILY_RETURNS      1.498670
# 5   GDAXI_DAILY_RETURNS      2.012433
# 6     VIX_DAILY_RETURNS      2.236008
# 7         NSEI_HL_RATIO      2.463040
# 8          DJI_HL_RATIO      2.859195
# 9              NSEI_RSI     16.505850
# 10              DJI_RSI     12.081633
# 11             NSEI_ROC      7.190963
# 12              DJI_ROC      6.754017
# 13             NSEI_AWE     30.386650
# 14              DJI_AWE     31.704844
# 15             NSEI_KAM   1371.506724
# 16              DJI_KAM    407.941126
# 17             NSEI_TSI     21.978123
# 18              DJI_TSI     17.752527
# 19             NSEI_VPT     17.541538
# 20             NSEI_ULC     10.073313
# 21              DJI_ULC     10.225403
# 22             NSEI_SMA  10662.528151
# 23              DJI_SMA   3158.409075
# 24             NSEI_EMA  11358.408136
# 25              DJI_EMA   3443.892891



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
# Model:                          Logit   Df Residuals:                     1194
# Method:                           MLE   Df Model:                           25
# Date:                Sun, 26 May 2024   Pseudo R-squ.:                  0.1582
# Time:                        12:38:25   Log-Likelihood:                -644.17
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 1.924e-37
# =======================================================================================
#                           coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# Intercept              19.0286     13.947      1.364      0.172      -8.307      46.364
# NSEI_DAILY_RETURNS     -0.0965      0.094     -1.028      0.304      -0.281       0.088
# DJI_DAILY_RETURNS      -0.0925      0.131     -0.708      0.479      -0.349       0.164
# IXIC_DAILY_RETURNS      0.4246      0.095      4.482      0.000       0.239       0.610
# HSI_DAILY_RETURNS      -0.1185      0.056     -2.108      0.035      -0.229      -0.008
# N225_DAILY_RETURNS     -0.1718      0.072     -2.394      0.017      -0.312      -0.031
# GDAXI_DAILY_RETURNS     0.0355      0.077      0.462      0.644      -0.115       0.186
# VIX_DAILY_RETURNS      -0.0371      0.014     -2.670      0.008      -0.064      -0.010
# DJI_HL_RATIO          -21.6954     13.920     -1.559      0.119     -48.979       5.588
# NSEI_RSI               -0.0378      0.021     -1.772      0.076      -0.080       0.004
# DJI_RSI                 0.0866      0.022      3.991      0.000       0.044       0.129
# NSEI_ROC               -0.0234      0.049     -0.474      0.635      -0.120       0.073
# DJI_ROC                 0.0330      0.047      0.702      0.483      -0.059       0.125
# NSEI_AWE                0.0005      0.001      0.550      0.582      -0.001       0.002
# DJI_AWE                -0.0005      0.001     -0.787      0.431      -0.002       0.001
# NSEI_KAM               -0.0006      0.001     -0.812      0.417      -0.002       0.001
# DJI_KAM                 0.0005      0.000      1.344      0.179      -0.000       0.001
# NSEI_TSI                0.0228      0.014      1.621      0.105      -0.005       0.050
# DJI_TSI                -0.0479      0.016     -3.051      0.002      -0.079      -0.017
# NSEI_VPT            -3.017e-06    2.9e-06     -1.040      0.298   -8.71e-06    2.67e-06
# NSEI_ULC               -0.0458      0.085     -0.539      0.590      -0.212       0.121
# DJI_ULC                -0.0167      0.086     -0.193      0.847      -0.186       0.153
# NSEI_SMA               -0.0033      0.002     -1.546      0.122      -0.007       0.001
# DJI_SMA                 0.0023      0.001      2.331      0.020       0.000       0.004
# NSEI_EMA                0.0038      0.002      1.741      0.082      -0.000       0.008
# DJI_EMA                -0.0026      0.001     -2.584      0.010      -0.005      -0.001
# =======================================================================================
# """

# Drop DJI_ULC



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                 Feature           VIF
# 0    NSEI_DAILY_RETURNS      2.146498
# 1     DJI_DAILY_RETURNS      4.669753
# 2    IXIC_DAILY_RETURNS      4.035063
# 3     HSI_DAILY_RETURNS      1.391716
# 4    N225_DAILY_RETURNS      1.498633
# 5   GDAXI_DAILY_RETURNS      1.983001
# 6     VIX_DAILY_RETURNS      2.235786
# 7          DJI_HL_RATIO      2.771034
# 8              NSEI_RSI     16.146065
# 9               DJI_RSI     11.950258
# 10             NSEI_ROC      7.190865
# 11              DJI_ROC      6.744287
# 12             NSEI_AWE     30.179707
# 13              DJI_AWE     31.625306
# 14             NSEI_KAM   1370.427018
# 15              DJI_KAM    406.927508
# 16             NSEI_TSI     21.726272
# 17              DJI_TSI     17.740249
# 18             NSEI_VPT     17.150123
# 19             NSEI_ULC      9.109499
# 20              DJI_ULC     10.225309
# 21             NSEI_SMA  10583.408341
# 22              DJI_SMA   3148.429227
# 23             NSEI_EMA  11253.907284
# 24              DJI_EMA   3432.175473



# %% 6 -
X_train = X_train.drop("DJI_ULC", axis = 1)
X_dropped.append("DJI_ULC")



# %% 7 -
model = Logit(y_train, X_train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1195
# Method:                           MLE   Df Model:                           24
# Date:                Sun, 26 May 2024   Pseudo R-squ.:                  0.1582
# Time:                        12:39:54   Log-Likelihood:                -644.19
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 6.069e-38
# =======================================================================================
#                           coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# Intercept              20.0767     12.851      1.562      0.118      -5.110      45.263
# NSEI_DAILY_RETURNS     -0.0970      0.094     -1.033      0.302      -0.281       0.087
# DJI_DAILY_RETURNS      -0.0907      0.131     -0.694      0.488      -0.347       0.165
# IXIC_DAILY_RETURNS      0.4246      0.095      4.482      0.000       0.239       0.610
# HSI_DAILY_RETURNS      -0.1185      0.056     -2.108      0.035      -0.229      -0.008
# N225_DAILY_RETURNS     -0.1711      0.072     -2.386      0.017      -0.312      -0.031
# GDAXI_DAILY_RETURNS     0.0354      0.077      0.461      0.645      -0.115       0.186
# VIX_DAILY_RETURNS      -0.0368      0.014     -2.663      0.008      -0.064      -0.010
# DJI_HL_RATIO          -22.7706     12.763     -1.784      0.074     -47.785       2.244
# NSEI_RSI               -0.0380      0.021     -1.782      0.075      -0.080       0.004
# DJI_RSI                 0.0864      0.022      3.986      0.000       0.044       0.129
# NSEI_ROC               -0.0232      0.049     -0.470      0.638      -0.120       0.074
# DJI_ROC                 0.0331      0.047      0.704      0.482      -0.059       0.125
# NSEI_AWE                0.0004      0.001      0.521      0.602      -0.001       0.002
# DJI_AWE                -0.0005      0.001     -0.764      0.445      -0.002       0.001
# NSEI_KAM               -0.0006      0.001     -0.807      0.419      -0.002       0.001
# DJI_KAM                 0.0005      0.000      1.329      0.184      -0.000       0.001
# NSEI_TSI                0.0228      0.014      1.619      0.105      -0.005       0.050
# DJI_TSI                -0.0475      0.016     -3.049      0.002      -0.078      -0.017
# NSEI_VPT            -2.989e-06    2.9e-06     -1.031      0.302   -8.67e-06    2.69e-06
# NSEI_ULC               -0.0539      0.074     -0.733      0.464      -0.198       0.090
# NSEI_SMA               -0.0033      0.002     -1.562      0.118      -0.007       0.001
# DJI_SMA                 0.0023      0.001      2.473      0.013       0.000       0.004
# NSEI_EMA                0.0038      0.002      1.752      0.080      -0.000       0.008
# DJI_EMA                -0.0027      0.001     -2.677      0.007      -0.005      -0.001
# =======================================================================================
# """

# Drop GDAXI_DAILY_RETURNS



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                 Feature           VIF
# 0    NSEI_DAILY_RETURNS      2.145855
# 1     DJI_DAILY_RETURNS      4.656054
# 2    IXIC_DAILY_RETURNS      4.034541
# 3     HSI_DAILY_RETURNS      1.391159
# 4    N225_DAILY_RETURNS      1.492945
# 5   GDAXI_DAILY_RETURNS      1.982988
# 6     VIX_DAILY_RETURNS      2.221244
# 7          DJI_HL_RATIO      2.337460
# 8              NSEI_RSI     16.075638
# 9               DJI_RSI     11.945920
# 10             NSEI_ROC      7.190340
# 11              DJI_ROC      6.744266
# 12             NSEI_AWE     28.605718
# 13              DJI_AWE     28.259517
# 14             NSEI_KAM   1367.864225
# 15              DJI_KAM    401.355806
# 16             NSEI_TSI     21.719575
# 17              DJI_TSI     17.635261
# 18             NSEI_VPT     17.000977
# 19             NSEI_ULC      6.911434
# 20             NSEI_SMA  10497.289001
# 21              DJI_SMA   2871.299313
# 22             NSEI_EMA  11205.875226
# 23              DJI_EMA   3247.173299



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
# Model:                          Logit   Df Residuals:                     1196
# Method:                           MLE   Df Model:                           23
# Date:                Sun, 26 May 2024   Pseudo R-squ.:                  0.1580
# Time:                        12:42:04   Log-Likelihood:                -644.29
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 2.030e-38
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             19.9835     12.880      1.552      0.121      -5.260      45.227
# NSEI_DAILY_RETURNS    -0.0869      0.091     -0.952      0.341      -0.266       0.092
# DJI_DAILY_RETURNS     -0.0727      0.124     -0.585      0.558      -0.316       0.171
# IXIC_DAILY_RETURNS     0.4225      0.095      4.467      0.000       0.237       0.608
# HSI_DAILY_RETURNS     -0.1150      0.056     -2.066      0.039      -0.224      -0.006
# N225_DAILY_RETURNS    -0.1701      0.072     -2.375      0.018      -0.311      -0.030
# VIX_DAILY_RETURNS     -0.0375      0.014     -2.727      0.006      -0.064      -0.011
# DJI_HL_RATIO         -22.6672     12.791     -1.772      0.076     -47.737       2.403
# NSEI_RSI              -0.0382      0.021     -1.794      0.073      -0.080       0.004
# DJI_RSI                0.0867      0.022      3.999      0.000       0.044       0.129
# NSEI_ROC              -0.0240      0.049     -0.488      0.626      -0.121       0.073
# DJI_ROC                0.0330      0.047      0.701      0.483      -0.059       0.125
# NSEI_AWE               0.0004      0.001      0.534      0.594      -0.001       0.002
# DJI_AWE               -0.0005      0.001     -0.768      0.442      -0.002       0.001
# NSEI_KAM              -0.0006      0.001     -0.781      0.435      -0.002       0.001
# DJI_KAM                0.0005      0.000      1.359      0.174      -0.000       0.001
# NSEI_TSI               0.0228      0.014      1.622      0.105      -0.005       0.050
# DJI_TSI               -0.0476      0.016     -3.053      0.002      -0.078      -0.017
# NSEI_VPT           -2.976e-06    2.9e-06     -1.028      0.304   -8.65e-06     2.7e-06
# NSEI_ULC              -0.0540      0.074     -0.734      0.463      -0.198       0.090
# NSEI_SMA              -0.0033      0.002     -1.567      0.117      -0.008       0.001
# DJI_SMA                0.0023      0.001      2.474      0.013       0.000       0.004
# NSEI_EMA               0.0038      0.002      1.748      0.081      -0.000       0.008
# DJI_EMA               -0.0027      0.001     -2.690      0.007      -0.005      -0.001
# ======================================================================================
# """

# Drop NSEI_ROC



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                Feature           VIF
# 0   NSEI_DAILY_RETURNS      2.025756
# 1    DJI_DAILY_RETURNS      4.175378
# 2   IXIC_DAILY_RETURNS      4.028807
# 3    HSI_DAILY_RETURNS      1.360895
# 4   N225_DAILY_RETURNS      1.489149
# 5    VIX_DAILY_RETURNS      2.201342
# 6         DJI_HL_RATIO      2.335883
# 7             NSEI_RSI     16.070100
# 8              DJI_RSI     11.938684
# 9             NSEI_ROC      7.175995
# 10             DJI_ROC      6.743998
# 11            NSEI_AWE     28.577249
# 12             DJI_AWE     28.259399
# 13            NSEI_KAM   1362.895980
# 14             DJI_KAM    399.690905
# 15            NSEI_TSI     21.718155
# 16             DJI_TSI     17.633074
# 17            NSEI_VPT     16.999261
# 18            NSEI_ULC      6.911033
# 19            NSEI_SMA  10496.593037
# 20             DJI_SMA   2870.678903
# 21            NSEI_EMA  11204.125330
# 22             DJI_EMA   3242.651412



# %% 6 -
X_train = X_train.drop("NSEI_ROC", axis = 1)
X_dropped.append("NSEI_ROC")



# %% 7 -
model = Logit(y_train, X_train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1197
# Method:                           MLE   Df Model:                           22
# Date:                Sun, 26 May 2024   Pseudo R-squ.:                  0.1579
# Time:                        12:43:34   Log-Likelihood:                -644.41
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 6.720e-39
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             20.0225     12.889      1.553      0.120      -5.239      45.284
# NSEI_DAILY_RETURNS    -0.0907      0.091     -0.996      0.319      -0.269       0.088
# DJI_DAILY_RETURNS     -0.0706      0.124     -0.571      0.568      -0.313       0.172
# IXIC_DAILY_RETURNS     0.4214      0.095      4.456      0.000       0.236       0.607
# HSI_DAILY_RETURNS     -0.1140      0.056     -2.049      0.040      -0.223      -0.005
# N225_DAILY_RETURNS    -0.1695      0.072     -2.367      0.018      -0.310      -0.029
# VIX_DAILY_RETURNS     -0.0376      0.014     -2.740      0.006      -0.064      -0.011
# DJI_HL_RATIO         -22.6241     12.801     -1.767      0.077     -47.713       2.465
# NSEI_RSI              -0.0418      0.020     -2.089      0.037      -0.081      -0.003
# DJI_RSI                0.0879      0.022      4.078      0.000       0.046       0.130
# DJI_ROC                0.0217      0.041      0.527      0.598      -0.059       0.102
# NSEI_AWE               0.0003      0.001      0.397      0.691      -0.001       0.002
# DJI_AWE               -0.0005      0.001     -0.750      0.453      -0.002       0.001
# NSEI_KAM              -0.0006      0.001     -0.817      0.414      -0.002       0.001
# DJI_KAM                0.0005      0.000      1.335      0.182      -0.000       0.001
# NSEI_TSI               0.0244      0.014      1.792      0.073      -0.002       0.051
# DJI_TSI               -0.0470      0.016     -3.029      0.002      -0.077      -0.017
# NSEI_VPT           -3.064e-06   2.89e-06     -1.061      0.289   -8.72e-06     2.6e-06
# NSEI_ULC              -0.0528      0.074     -0.717      0.473      -0.197       0.091
# NSEI_SMA              -0.0031      0.002     -1.493      0.135      -0.007       0.001
# DJI_SMA                0.0022      0.001      2.423      0.015       0.000       0.004
# NSEI_EMA               0.0036      0.002      1.682      0.092      -0.001       0.008
# DJI_EMA               -0.0026      0.001     -2.647      0.008      -0.005      -0.001
# ======================================================================================
# """

# Drop NSEI_AWE



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                Feature           VIF
# 0   NSEI_DAILY_RETURNS      2.003788
# 1    DJI_DAILY_RETURNS      4.167761
# 2   IXIC_DAILY_RETURNS      4.024949
# 3    HSI_DAILY_RETURNS      1.358481
# 4   N225_DAILY_RETURNS      1.486910
# 5    VIX_DAILY_RETURNS      2.199693
# 6         DJI_HL_RATIO      2.333941
# 7             NSEI_RSI     14.117300
# 8              DJI_RSI     11.746073
# 9              DJI_ROC      4.997394
# 10            NSEI_AWE     25.729795
# 11             DJI_AWE     28.219264
# 12            NSEI_KAM   1351.700126
# 13             DJI_KAM    398.831613
# 14            NSEI_TSI     20.453606
# 15             DJI_TSI     17.556641
# 16            NSEI_VPT     16.922385
# 17            NSEI_ULC      6.906865
# 18            NSEI_SMA   9799.005225
# 19             DJI_SMA   2733.057218
# 20            NSEI_EMA  10698.101117
# 21             DJI_EMA   3087.061282



# %% 6 -
X_train = X_train.drop("NSEI_AWE", axis = 1)
X_dropped.append("NSEI_AWE")



# %% 7 -
model = Logit(y_train, X_train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1198
# Method:                           MLE   Df Model:                           21
# Date:                Sun, 26 May 2024   Pseudo R-squ.:                  0.1578
# Time:                        12:44:48   Log-Likelihood:                -644.49
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 2.096e-39
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             19.7682     12.871      1.536      0.125      -5.458      44.995
# NSEI_DAILY_RETURNS    -0.0930      0.091     -1.022      0.307      -0.271       0.085
# DJI_DAILY_RETURNS     -0.0709      0.124     -0.573      0.567      -0.313       0.171
# IXIC_DAILY_RETURNS     0.4226      0.095      4.470      0.000       0.237       0.608
# HSI_DAILY_RETURNS     -0.1135      0.056     -2.041      0.041      -0.222      -0.005
# N225_DAILY_RETURNS    -0.1684      0.072     -2.355      0.019      -0.309      -0.028
# VIX_DAILY_RETURNS     -0.0376      0.014     -2.739      0.006      -0.064      -0.011
# DJI_HL_RATIO         -22.3723     12.783     -1.750      0.080     -47.427       2.682
# NSEI_RSI              -0.0425      0.020     -2.134      0.033      -0.082      -0.003
# DJI_RSI                0.0893      0.021      4.199      0.000       0.048       0.131
# DJI_ROC                0.0166      0.039      0.425      0.671      -0.060       0.093
# DJI_AWE               -0.0003      0.000     -0.692      0.489      -0.001       0.000
# NSEI_KAM              -0.0006      0.001     -0.770      0.441      -0.002       0.001
# DJI_KAM                0.0005      0.000      1.346      0.178      -0.000       0.001
# NSEI_TSI               0.0278      0.011      2.608      0.009       0.007       0.049
# DJI_TSI               -0.0502      0.013     -3.744      0.000      -0.076      -0.024
# NSEI_VPT           -3.021e-06   2.88e-06     -1.048      0.295   -8.67e-06    2.63e-06
# NSEI_ULC              -0.0556      0.073     -0.758      0.448      -0.199       0.088
# NSEI_SMA              -0.0032      0.002     -1.608      0.108      -0.007       0.001
# DJI_SMA                0.0023      0.001      2.472      0.013       0.000       0.004
# NSEI_EMA               0.0037      0.002      1.760      0.078      -0.000       0.008
# DJI_EMA               -0.0026      0.001     -2.699      0.007      -0.005      -0.001
# ======================================================================================
# """

# Drop DJI_ROC



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                Feature           VIF
# 0   NSEI_DAILY_RETURNS      1.992961
# 1    DJI_DAILY_RETURNS      4.167755
# 2   IXIC_DAILY_RETURNS      4.023171
# 3    HSI_DAILY_RETURNS      1.357458
# 4   N225_DAILY_RETURNS      1.485257
# 5    VIX_DAILY_RETURNS      2.199519
# 6         DJI_HL_RATIO      2.327943
# 7             NSEI_RSI     14.029568
# 8              DJI_RSI     11.385215
# 9              DJI_ROC      4.438404
# 10             DJI_AWE     11.368456
# 11            NSEI_KAM   1326.605587
# 12             DJI_KAM    398.473716
# 13            NSEI_TSI     12.415532
# 14             DJI_TSI     12.732473
# 15            NSEI_VPT     16.918253
# 16            NSEI_ULC      6.837845
# 17            NSEI_SMA   9400.664946
# 18             DJI_SMA   2712.972958
# 19            NSEI_EMA  10480.560270
# 20             DJI_EMA   3061.537341



# %% 6 -
X_train = X_train.drop("DJI_ROC", axis = 1)
X_dropped.append("DJI_ROC")



# %% 7 -
model = Logit(y_train, X_train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1199
# Method:                           MLE   Df Model:                           20
# Date:                Sun, 26 May 2024   Pseudo R-squ.:                  0.1577
# Time:                        12:45:47   Log-Likelihood:                -644.58
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 6.452e-40
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             20.5507     12.736      1.614      0.107      -4.411      45.513
# NSEI_DAILY_RETURNS    -0.0916      0.091     -1.004      0.315      -0.270       0.087
# DJI_DAILY_RETURNS     -0.0659      0.124     -0.533      0.594      -0.308       0.176
# IXIC_DAILY_RETURNS     0.4241      0.094      4.489      0.000       0.239       0.609
# HSI_DAILY_RETURNS     -0.1145      0.056     -2.061      0.039      -0.223      -0.006
# N225_DAILY_RETURNS    -0.1655      0.071     -2.324      0.020      -0.305      -0.026
# VIX_DAILY_RETURNS     -0.0371      0.014     -2.710      0.007      -0.064      -0.010
# DJI_HL_RATIO         -23.1768     12.642     -1.833      0.067     -47.954       1.601
# NSEI_RSI              -0.0432      0.020     -2.169      0.030      -0.082      -0.004
# DJI_RSI                0.0917      0.021      4.464      0.000       0.051       0.132
# DJI_AWE               -0.0002      0.000     -0.547      0.585      -0.001       0.000
# NSEI_KAM              -0.0005      0.001     -0.733      0.464      -0.002       0.001
# DJI_KAM                0.0005      0.000      1.444      0.149      -0.000       0.001
# NSEI_TSI               0.0278      0.011      2.597      0.009       0.007       0.049
# DJI_TSI               -0.0515      0.013     -3.942      0.000      -0.077      -0.026
# NSEI_VPT           -2.918e-06   2.87e-06     -1.015      0.310   -8.55e-06    2.72e-06
# NSEI_ULC              -0.0476      0.071     -0.674      0.500      -0.186       0.091
# NSEI_SMA              -0.0033      0.002     -1.628      0.104      -0.007       0.001
# DJI_SMA                0.0022      0.001      2.442      0.015       0.000       0.004
# NSEI_EMA               0.0037      0.002      1.763      0.078      -0.000       0.008
# DJI_EMA               -0.0026      0.001     -2.665      0.008      -0.004      -0.001
# ======================================================================================
# """

#  Drop DJI_DAILY_RETURNS



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                Feature           VIF
# 0   NSEI_DAILY_RETURNS      1.989023
# 1    DJI_DAILY_RETURNS      4.122573
# 2   IXIC_DAILY_RETURNS      4.017805
# 3    HSI_DAILY_RETURNS      1.353818
# 4   N225_DAILY_RETURNS      1.468820
# 5    VIX_DAILY_RETURNS      2.184374
# 6         DJI_HL_RATIO      2.261062
# 7             NSEI_RSI     13.905453
# 8              DJI_RSI     10.537275
# 9              DJI_AWE      7.534801
# 10            NSEI_KAM   1311.163793
# 11             DJI_KAM    384.118137
# 12            NSEI_TSI     12.414331
# 13             DJI_TSI     11.972998
# 14            NSEI_VPT     16.725477
# 15            NSEI_ULC      6.305868
# 16            NSEI_SMA   9365.952878
# 17             DJI_SMA   2573.947709
# 18            NSEI_EMA  10477.206424
# 19             DJI_EMA   3003.470201



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
# Model:                          Logit   Df Residuals:                     1200
# Method:                           MLE   Df Model:                           19
# Date:                Sun, 26 May 2024   Pseudo R-squ.:                  0.1575
# Time:                        12:47:18   Log-Likelihood:                -644.72
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 2.025e-40
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             20.4602     12.799      1.599      0.110      -4.625      45.545
# NSEI_DAILY_RETURNS    -0.0987      0.091     -1.087      0.277      -0.276       0.079
# IXIC_DAILY_RETURNS     0.3941      0.076      5.217      0.000       0.246       0.542
# HSI_DAILY_RETURNS     -0.1105      0.055     -2.008      0.045      -0.218      -0.003
# N225_DAILY_RETURNS    -0.1691      0.071     -2.377      0.017      -0.309      -0.030
# VIX_DAILY_RETURNS     -0.0358      0.014     -2.645      0.008      -0.062      -0.009
# DJI_HL_RATIO         -23.0114     12.697     -1.812      0.070     -47.898       1.875
# NSEI_RSI              -0.0414      0.020     -2.107      0.035      -0.080      -0.003
# DJI_RSI                0.0881      0.019      4.529      0.000       0.050       0.126
# DJI_AWE               -0.0002      0.000     -0.577      0.564      -0.001       0.000
# NSEI_KAM              -0.0006      0.001     -0.769      0.442      -0.002       0.001
# DJI_KAM                0.0005      0.000      1.430      0.153      -0.000       0.001
# NSEI_TSI               0.0270      0.011      2.548      0.011       0.006       0.048
# DJI_TSI               -0.0496      0.013     -3.936      0.000      -0.074      -0.025
# NSEI_VPT           -3.001e-06   2.87e-06     -1.044      0.297   -8.63e-06    2.63e-06
# NSEI_ULC              -0.0485      0.071     -0.684      0.494      -0.187       0.090
# NSEI_SMA              -0.0032      0.002     -1.581      0.114      -0.007       0.001
# DJI_SMA                0.0021      0.001      2.384      0.017       0.000       0.004
# NSEI_EMA               0.0036      0.002      1.729      0.084      -0.000       0.008
# DJI_EMA               -0.0025      0.001     -2.612      0.009      -0.004      -0.001
# ======================================================================================
# """

# Drop DJI_AWE



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                Feature           VIF
# 0   NSEI_DAILY_RETURNS      1.940575
# 1   IXIC_DAILY_RETURNS      2.128128
# 2    HSI_DAILY_RETURNS      1.335002
# 3   N225_DAILY_RETURNS      1.447881
# 4    VIX_DAILY_RETURNS      2.141760
# 5         DJI_HL_RATIO      2.261061
# 6             NSEI_RSI     13.650721
# 7              DJI_RSI      9.965624
# 8              DJI_AWE      7.519929
# 9             NSEI_KAM   1303.039809
# 10             DJI_KAM    383.881280
# 11            NSEI_TSI     12.225987
# 12             DJI_TSI     11.541376
# 13            NSEI_VPT     16.707013
# 14            NSEI_ULC      6.287075
# 15            NSEI_SMA   9338.809825
# 16             DJI_SMA   2553.938441
# 17            NSEI_EMA  10472.003879
# 18             DJI_EMA   2978.687689



# %% 6 -
X_train = X_train.drop("DJI_AWE", axis = 1)
X_dropped.append("DJI_AWE")



# %% 7 -
model = Logit(y_train, X_train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1201
# Method:                           MLE   Df Model:                           18
# Date:                Sun, 26 May 2024   Pseudo R-squ.:                  0.1573
# Time:                        12:48:39   Log-Likelihood:                -644.89
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 6.347e-41
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             21.0680     12.784      1.648      0.099      -3.989      46.125
# NSEI_DAILY_RETURNS    -0.1010      0.091     -1.112      0.266      -0.279       0.077
# IXIC_DAILY_RETURNS     0.3967      0.075      5.269      0.000       0.249       0.544
# HSI_DAILY_RETURNS     -0.1119      0.055     -2.036      0.042      -0.220      -0.004
# N225_DAILY_RETURNS    -0.1661      0.071     -2.339      0.019      -0.305      -0.027
# VIX_DAILY_RETURNS     -0.0353      0.013     -2.623      0.009      -0.062      -0.009
# DJI_HL_RATIO         -23.4182     12.707     -1.843      0.065     -48.323       1.487
# NSEI_RSI              -0.0417      0.020     -2.123      0.034      -0.080      -0.003
# DJI_RSI                0.0882      0.019      4.533      0.000       0.050       0.126
# NSEI_KAM              -0.0006      0.001     -0.857      0.391      -0.002       0.001
# DJI_KAM                0.0005      0.000      1.578      0.115      -0.000       0.001
# NSEI_TSI               0.0266      0.011      2.514      0.012       0.006       0.047
# DJI_TSI               -0.0528      0.011     -4.659      0.000      -0.075      -0.031
# NSEI_VPT           -2.441e-06   2.71e-06     -0.900      0.368   -7.76e-06    2.87e-06
# NSEI_ULC              -0.0260      0.060     -0.437      0.662      -0.143       0.091
# NSEI_SMA              -0.0029      0.002     -1.485      0.137      -0.007       0.001
# DJI_SMA                0.0021      0.001      2.430      0.015       0.000       0.004
# NSEI_EMA               0.0034      0.002      1.649      0.099      -0.001       0.007
# DJI_EMA               -0.0026      0.001     -2.728      0.006      -0.004      -0.001
# ======================================================================================
# """

# Drop NSEI_ULC



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                Feature           VIF
# 0   NSEI_DAILY_RETURNS      1.939171
# 1   IXIC_DAILY_RETURNS      2.128103
# 2    HSI_DAILY_RETURNS      1.334135
# 3   N225_DAILY_RETURNS      1.440271
# 4    VIX_DAILY_RETURNS      2.137254
# 5         DJI_HL_RATIO      2.260441
# 6             NSEI_RSI     13.648389
# 7              DJI_RSI      9.948546
# 8             NSEI_KAM   1264.894795
# 9              DJI_KAM    364.085677
# 10            NSEI_TSI     12.165152
# 11             DJI_TSI      9.181087
# 12            NSEI_VPT     14.596962
# 13            NSEI_ULC      4.160054
# 14            NSEI_SMA   8666.366430
# 15             DJI_SMA   2539.137878
# 16            NSEI_EMA  10052.398345
# 17             DJI_EMA   2899.513359



# %% 6 -
X_train = X_train.drop("NSEI_ULC", axis = 1)
X_dropped.append("NSEI_ULC")



# %% 7 -
model = Logit(y_train, X_train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1202
# Method:                           MLE   Df Model:                           17
# Date:                Sun, 26 May 2024   Pseudo R-squ.:                  0.1571
# Time:                        12:49:55   Log-Likelihood:                -644.98
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 1.810e-41
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             23.0218     11.967      1.924      0.054      -0.433      46.477
# NSEI_DAILY_RETURNS    -0.1083      0.089     -1.217      0.224      -0.283       0.066
# IXIC_DAILY_RETURNS     0.3970      0.076      5.252      0.000       0.249       0.545
# HSI_DAILY_RETURNS     -0.1117      0.055     -2.032      0.042      -0.219      -0.004
# N225_DAILY_RETURNS    -0.1655      0.071     -2.330      0.020      -0.305      -0.026
# VIX_DAILY_RETURNS     -0.0349      0.013     -2.596      0.009      -0.061      -0.009
# DJI_HL_RATIO         -25.2681     11.970     -2.111      0.035     -48.728      -1.808
# NSEI_RSI              -0.0402      0.019     -2.082      0.037      -0.078      -0.002
# DJI_RSI                0.0885      0.019      4.553      0.000       0.050       0.127
# NSEI_KAM              -0.0006      0.001     -0.835      0.404      -0.002       0.001
# DJI_KAM                0.0005      0.000      1.529      0.126      -0.000       0.001
# NSEI_TSI               0.0270      0.011      2.571      0.010       0.006       0.048
# DJI_TSI               -0.0528      0.011     -4.662      0.000      -0.075      -0.031
# NSEI_VPT           -1.911e-06   2.43e-06     -0.787      0.431   -6.67e-06    2.85e-06
# NSEI_SMA              -0.0027      0.002     -1.421      0.155      -0.006       0.001
# DJI_SMA                0.0022      0.001      2.628      0.009       0.001       0.004
# NSEI_EMA               0.0032      0.002      1.589      0.112      -0.001       0.007
# DJI_EMA               -0.0026      0.001     -2.872      0.004      -0.004      -0.001
# ======================================================================================
# """

# Drop NSEI_VPT



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                Feature          VIF
# 0   NSEI_DAILY_RETURNS     1.871275
# 1   IXIC_DAILY_RETURNS     2.125631
# 2    HSI_DAILY_RETURNS     1.333715
# 3   N225_DAILY_RETURNS     1.437705
# 4    VIX_DAILY_RETURNS     2.136058
# 5         DJI_HL_RATIO     1.943113
# 6             NSEI_RSI    13.389923
# 7              DJI_RSI     9.925318
# 8             NSEI_KAM  1256.613840
# 9              DJI_KAM   351.002725
# 10            NSEI_TSI    11.982679
# 11             DJI_TSI     9.177281
# 12            NSEI_VPT    11.663477
# 13            NSEI_SMA  8255.164533
# 14             DJI_SMA  2370.221637
# 15            NSEI_EMA  9494.686554
# 16             DJI_EMA  2793.152268



# %% 6 -
X_train = X_train.drop("NSEI_VPT", axis = 1)
X_dropped.append("NSEI_VPT")



# %% 7 -
model = Logit(y_train, X_train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1203
# Method:                           MLE   Df Model:                           16
# Date:                Sun, 26 May 2024   Pseudo R-squ.:                  0.1567
# Time:                        12:50:55   Log-Likelihood:                -645.29
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 6.134e-42
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             20.4528     11.503      1.778      0.075      -2.093      42.999
# NSEI_DAILY_RETURNS    -0.1078      0.089     -1.211      0.226      -0.282       0.067
# IXIC_DAILY_RETURNS     0.3954      0.075      5.261      0.000       0.248       0.543
# HSI_DAILY_RETURNS     -0.1121      0.055     -2.040      0.041      -0.220      -0.004
# N225_DAILY_RETURNS    -0.1642      0.071     -2.319      0.020      -0.303      -0.025
# VIX_DAILY_RETURNS     -0.0356      0.013     -2.658      0.008      -0.062      -0.009
# DJI_HL_RATIO         -21.9447     11.191     -1.961      0.050     -43.879      -0.011
# NSEI_RSI              -0.0409      0.019     -2.118      0.034      -0.079      -0.003
# DJI_RSI                0.0888      0.019      4.570      0.000       0.051       0.127
# NSEI_KAM              -0.0006      0.001     -0.829      0.407      -0.002       0.001
# DJI_KAM                0.0006      0.000      1.711      0.087   -8.11e-05       0.001
# NSEI_TSI               0.0270      0.011      2.566      0.010       0.006       0.048
# DJI_TSI               -0.0524      0.011     -4.633      0.000      -0.075      -0.030
# NSEI_SMA              -0.0027      0.002     -1.431      0.152      -0.006       0.001
# DJI_SMA                0.0022      0.001      2.608      0.009       0.001       0.004
# NSEI_EMA               0.0032      0.002      1.580      0.114      -0.001       0.007
# DJI_EMA               -0.0027      0.001     -2.934      0.003      -0.004      -0.001
# ======================================================================================
# """

# Drop NSEI_KAM



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                Feature          VIF
# 0   NSEI_DAILY_RETURNS     1.869024
# 1   IXIC_DAILY_RETURNS     2.125449
# 2    HSI_DAILY_RETURNS     1.333715
# 3   N225_DAILY_RETURNS     1.435077
# 4    VIX_DAILY_RETURNS     2.128350
# 5         DJI_HL_RATIO     1.666902
# 6             NSEI_RSI    13.389883
# 7              DJI_RSI     9.924437
# 8             NSEI_KAM  1256.481689
# 9              DJI_KAM   334.600988
# 10            NSEI_TSI    11.959780
# 11             DJI_TSI     9.128811
# 12            NSEI_SMA  8250.767565
# 13             DJI_SMA  2363.312752
# 14            NSEI_EMA  9482.525464
# 15             DJI_EMA  2785.162456



# %% 6 -
X_train = X_train.drop("NSEI_KAM", axis = 1)
X_dropped.append("NSEI_KAM")



# %% 7 -
model = Logit(y_train, X_train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1204
# Method:                           MLE   Df Model:                           15
# Date:                Sun, 26 May 2024   Pseudo R-squ.:                  0.1563
# Time:                        12:52:05   Log-Likelihood:                -645.64
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 2.080e-42
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             19.9503     11.516      1.732      0.083      -2.621      42.521
# NSEI_DAILY_RETURNS    -0.1073      0.089     -1.204      0.229      -0.282       0.067
# IXIC_DAILY_RETURNS     0.3946      0.075      5.265      0.000       0.248       0.541
# HSI_DAILY_RETURNS     -0.1125      0.055     -2.045      0.041      -0.220      -0.005
# N225_DAILY_RETURNS    -0.1658      0.071     -2.342      0.019      -0.305      -0.027
# VIX_DAILY_RETURNS     -0.0355      0.013     -2.657      0.008      -0.062      -0.009
# DJI_HL_RATIO         -21.3753     11.195     -1.909      0.056     -43.317       0.566
# NSEI_RSI              -0.0414      0.019     -2.147      0.032      -0.079      -0.004
# DJI_RSI                0.0901      0.019      4.640      0.000       0.052       0.128
# DJI_KAM                0.0005      0.000      1.661      0.097   -9.68e-05       0.001
# NSEI_TSI               0.0267      0.010      2.549      0.011       0.006       0.047
# DJI_TSI               -0.0533      0.011     -4.726      0.000      -0.075      -0.031
# NSEI_SMA              -0.0027      0.002     -1.454      0.146      -0.006       0.001
# DJI_SMA                0.0022      0.001      2.599      0.009       0.001       0.004
# NSEI_EMA               0.0026      0.002      1.385      0.166      -0.001       0.006
# DJI_EMA               -0.0027      0.001     -2.910      0.004      -0.004      -0.001
# ======================================================================================
# """

# Drop NSEI_DAILY_RETURNS



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                Feature          VIF
# 0   NSEI_DAILY_RETURNS     1.868692
# 1   IXIC_DAILY_RETURNS     2.123970
# 2    HSI_DAILY_RETURNS     1.333600
# 3   N225_DAILY_RETURNS     1.433121
# 4    VIX_DAILY_RETURNS     2.127977
# 5         DJI_HL_RATIO     1.651439
# 6             NSEI_RSI    13.379599
# 7              DJI_RSI     9.841157
# 8              DJI_KAM   334.096023
# 9             NSEI_TSI    11.951748
# 10             DJI_TSI     9.013121
# 11            NSEI_SMA  8250.767499
# 12             DJI_SMA  2362.687536
# 13            NSEI_EMA  8243.861564
# 14             DJI_EMA  2783.456335



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
# Model:                          Logit   Df Residuals:                     1205
# Method:                           MLE   Df Model:                           14
# Date:                Sun, 26 May 2024   Pseudo R-squ.:                  0.1553
# Time:                        12:53:05   Log-Likelihood:                -646.38
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 9.924e-43
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             19.7219     11.408      1.729      0.084      -2.637      42.081
# IXIC_DAILY_RETURNS     0.3896      0.075      5.219      0.000       0.243       0.536
# HSI_DAILY_RETURNS     -0.1259      0.054     -2.335      0.020      -0.232      -0.020
# N225_DAILY_RETURNS    -0.1826      0.069     -2.635      0.008      -0.319      -0.047
# VIX_DAILY_RETURNS     -0.0339      0.013     -2.548      0.011      -0.060      -0.008
# DJI_HL_RATIO         -20.7693     11.066     -1.877      0.061     -42.458       0.919
# NSEI_RSI              -0.0539      0.016     -3.306      0.001      -0.086      -0.022
# DJI_RSI                0.0939      0.019      4.891      0.000       0.056       0.132
# DJI_KAM                0.0005      0.000      1.639      0.101      -0.000       0.001
# NSEI_TSI               0.0329      0.009      3.571      0.000       0.015       0.051
# DJI_TSI               -0.0553      0.011     -4.942      0.000      -0.077      -0.033
# NSEI_SMA              -0.0034      0.002     -1.895      0.058      -0.007       0.000
# DJI_SMA                0.0023      0.001      2.777      0.005       0.001       0.004
# NSEI_EMA               0.0033      0.002      1.822      0.068      -0.000       0.007
# DJI_EMA               -0.0028      0.001     -3.060      0.002      -0.005      -0.001
# ======================================================================================
# """

# Drop DJI_KAM



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                Feature          VIF
# 0   IXIC_DAILY_RETURNS     2.114061
# 1    HSI_DAILY_RETURNS     1.262880
# 2   N225_DAILY_RETURNS     1.381340
# 3    VIX_DAILY_RETURNS     2.098437
# 4         DJI_HL_RATIO     1.636396
# 5             NSEI_RSI     9.406937
# 6              DJI_RSI     9.454079
# 7              DJI_KAM   333.773271
# 8             NSEI_TSI     9.100848
# 9              DJI_TSI     8.747925
# 10            NSEI_SMA  7424.763354
# 11             DJI_SMA  2299.785000
# 12            NSEI_EMA  7421.334616
# 13             DJI_EMA  2730.611426



# %% 6 -
X_train = X_train.drop("DJI_KAM", axis = 1)
X_dropped.append("DJI_KAM")



# %% 7 -
model = Logit(y_train, X_train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1206
# Method:                           MLE   Df Model:                           13
# Date:                Sun, 26 May 2024   Pseudo R-squ.:                  0.1535
# Time:                        12:54:14   Log-Likelihood:                -647.73
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 8.253e-43
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             16.6057     11.331      1.465      0.143      -5.603      38.814
# IXIC_DAILY_RETURNS     0.3861      0.074      5.223      0.000       0.241       0.531
# HSI_DAILY_RETURNS     -0.1254      0.054     -2.329      0.020      -0.231      -0.020
# N225_DAILY_RETURNS    -0.1777      0.069     -2.565      0.010      -0.313      -0.042
# VIX_DAILY_RETURNS     -0.0365      0.013     -2.763      0.006      -0.062      -0.011
# DJI_HL_RATIO         -17.3973     10.947     -1.589      0.112     -38.854       4.059
# NSEI_RSI              -0.0529      0.016     -3.257      0.001      -0.085      -0.021
# DJI_RSI                0.0908      0.019      4.760      0.000       0.053       0.128
# NSEI_TSI               0.0315      0.009      3.444      0.001       0.014       0.049
# DJI_TSI               -0.0520      0.011     -4.742      0.000      -0.073      -0.030
# NSEI_SMA              -0.0034      0.002     -1.920      0.055      -0.007    7.16e-05
# DJI_SMA                0.0023      0.001      2.695      0.007       0.001       0.004
# NSEI_EMA               0.0033      0.002      1.858      0.063      -0.000       0.007
# DJI_EMA               -0.0022      0.001     -2.627      0.009      -0.004      -0.001
# ======================================================================================
# """

# Drop DJI_HL_RATIO



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                Feature          VIF
# 0   IXIC_DAILY_RETURNS     2.112581
# 1    HSI_DAILY_RETURNS     1.262585
# 2   N225_DAILY_RETURNS     1.377125
# 3    VIX_DAILY_RETURNS     2.071630
# 4         DJI_HL_RATIO     1.559329
# 5             NSEI_RSI     9.390547
# 6              DJI_RSI     9.357348
# 7             NSEI_TSI     9.015876
# 8              DJI_TSI     8.473563
# 9             NSEI_SMA  7424.298144
# 10             DJI_SMA  2290.106099
# 11            NSEI_EMA  7419.839302
# 12             DJI_EMA  2293.768132



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
# Model:                          Logit   Df Residuals:                     1207
# Method:                           MLE   Df Model:                           12
# Date:                Sun, 26 May 2024   Pseudo R-squ.:                  0.1519
# Time:                        12:54:58   Log-Likelihood:                -649.00
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 6.134e-43
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             -1.2691      1.265     -1.004      0.316      -3.748       1.210
# IXIC_DAILY_RETURNS     0.4011      0.075      5.377      0.000       0.255       0.547
# HSI_DAILY_RETURNS     -0.1318      0.054     -2.453      0.014      -0.237      -0.026
# N225_DAILY_RETURNS    -0.1735      0.069     -2.508      0.012      -0.309      -0.038
# VIX_DAILY_RETURNS     -0.0375      0.013     -2.834      0.005      -0.063      -0.012
# NSEI_RSI              -0.0537      0.016     -3.307      0.001      -0.086      -0.022
# DJI_RSI                0.0940      0.019      4.980      0.000       0.057       0.131
# NSEI_TSI               0.0328      0.009      3.603      0.000       0.015       0.051
# DJI_TSI               -0.0510      0.011     -4.676      0.000      -0.072      -0.030
# NSEI_SMA              -0.0037      0.002     -2.058      0.040      -0.007      -0.000
# DJI_SMA                0.0023      0.001      2.717      0.007       0.001       0.004
# NSEI_EMA               0.0036      0.002      1.996      0.046     6.4e-05       0.007
# DJI_EMA               -0.0022      0.001     -2.643      0.008      -0.004      -0.001
# ======================================================================================
# """



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                Feature          VIF
# 0   IXIC_DAILY_RETURNS     2.103745
# 1    HSI_DAILY_RETURNS     1.261519
# 2   N225_DAILY_RETURNS     1.376026
# 3    VIX_DAILY_RETURNS     2.045503
# 4             NSEI_RSI     9.386014
# 5              DJI_RSI     9.318598
# 6             NSEI_TSI     8.931781
# 7              DJI_TSI     8.379260
# 8             NSEI_SMA  7362.629711
# 9              DJI_SMA  2290.105561
# 10            NSEI_EMA  7358.427399
# 11             DJI_EMA  2293.751609

# Drop NSEI_SMA



# %% 6 -
X_train = X_train.drop("NSEI_SMA", axis = 1)
X_dropped.append("NSEI_SMA")



# %% 7 -
model = Logit(y_train, X_train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1208
# Method:                           MLE   Df Model:                           11
# Date:                Sun, 26 May 2024   Pseudo R-squ.:                  0.1491
# Time:                        12:56:04   Log-Likelihood:                -651.14
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 1.015e-42
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             -1.7730      1.240     -1.430      0.153      -4.204       0.658
# IXIC_DAILY_RETURNS     0.4072      0.075      5.453      0.000       0.261       0.554
# HSI_DAILY_RETURNS     -0.1288      0.053     -2.420      0.016      -0.233      -0.024
# N225_DAILY_RETURNS    -0.1845      0.069     -2.687      0.007      -0.319      -0.050
# VIX_DAILY_RETURNS     -0.0385      0.013     -2.916      0.004      -0.064      -0.013
# NSEI_RSI              -0.0332      0.013     -2.615      0.009      -0.058      -0.008
# DJI_RSI                0.0832      0.018      4.604      0.000       0.048       0.119
# NSEI_TSI               0.0223      0.007      2.984      0.003       0.008       0.037
# DJI_TSI               -0.0453      0.011     -4.300      0.000      -0.066      -0.025
# DJI_SMA                0.0013      0.001      1.875      0.061   -5.82e-05       0.003
# NSEI_EMA              -0.0001   5.93e-05     -1.866      0.062      -0.000    5.57e-06
# DJI_EMA               -0.0012      0.001     -1.786      0.074      -0.003       0.000
# ======================================================================================
# """

# Drop DJI_EMA



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                Feature          VIF
# 0   IXIC_DAILY_RETURNS     2.102070
# 1    HSI_DAILY_RETURNS     1.260945
# 2   N225_DAILY_RETURNS     1.368961
# 3    VIX_DAILY_RETURNS     2.039100
# 4             NSEI_RSI     5.954040
# 5              DJI_RSI     8.707519
# 6             NSEI_TSI     6.243538
# 7              DJI_TSI     7.876151
# 8              DJI_SMA  1581.669494
# 9             NSEI_EMA     8.867109
# 10             DJI_EMA  1588.519470

# Drop DJI_EMA



# %% 6 -
X_train = X_train.drop("DJI_EMA", axis = 1)
X_dropped.append("DJI_EMA")



# %% 7 -
model = Logit(y_train, X_train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1209
# Method:                           MLE   Df Model:                           10
# Date:                Sun, 26 May 2024   Pseudo R-squ.:                  0.1470
# Time:                        12:56:51   Log-Likelihood:                -652.75
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 9.726e-43
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             -0.8377      1.126     -0.744      0.457      -3.044       1.368
# IXIC_DAILY_RETURNS     0.4148      0.075      5.550      0.000       0.268       0.561
# HSI_DAILY_RETURNS     -0.1252      0.053     -2.350      0.019      -0.230      -0.021
# N225_DAILY_RETURNS    -0.1778      0.068     -2.603      0.009      -0.312      -0.044
# VIX_DAILY_RETURNS     -0.0418      0.013     -3.186      0.001      -0.067      -0.016
# NSEI_RSI              -0.0357      0.013     -2.838      0.005      -0.060      -0.011
# DJI_RSI                0.0646      0.015      4.391      0.000       0.036       0.093
# NSEI_TSI               0.0241      0.007      3.242      0.001       0.010       0.039
# DJI_TSI               -0.0359      0.009     -3.943      0.000      -0.054      -0.018
# DJI_SMA             6.438e-05   4.88e-05      1.319      0.187   -3.13e-05       0.000
# NSEI_EMA              -0.0001   5.91e-05     -2.027      0.043      -0.000   -3.98e-06
# ======================================================================================
# """

# Drop DJI_SMA



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  IXIC_DAILY_RETURNS  2.101025
# 1   HSI_DAILY_RETURNS  1.258702
# 2  N225_DAILY_RETURNS  1.365178
# 3   VIX_DAILY_RETURNS  1.996652
# 4            NSEI_RSI  5.822026
# 5             DJI_RSI  6.033863
# 6            NSEI_TSI  6.093634
# 7             DJI_TSI  6.125689
# 8             DJI_SMA  8.731810
# 9            NSEI_EMA  8.784389



# %% 6 -
X_train = X_train.drop("DJI_SMA", axis = 1)
X_dropped.append("DJI_SMA")



# %% 7 -
model = Logit(y_train, X_train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1210
# Method:                           MLE   Df Model:                            9
# Date:                Sun, 26 May 2024   Pseudo R-squ.:                  0.1459
# Time:                        12:57:42   Log-Likelihood:                -653.62
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 4.403e-43
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept              0.1844      0.816      0.226      0.821      -1.415       1.784
# IXIC_DAILY_RETURNS     0.4221      0.075      5.651      0.000       0.276       0.569
# HSI_DAILY_RETURNS     -0.1257      0.053     -2.361      0.018      -0.230      -0.021
# N225_DAILY_RETURNS    -0.1761      0.068     -2.580      0.010      -0.310      -0.042
# VIX_DAILY_RETURNS     -0.0414      0.013     -3.156      0.002      -0.067      -0.016
# NSEI_RSI              -0.0345      0.013     -2.745      0.006      -0.059      -0.010
# DJI_RSI                0.0604      0.014      4.207      0.000       0.032       0.088
# NSEI_TSI               0.0230      0.007      3.126      0.002       0.009       0.037
# DJI_TSI               -0.0320      0.009     -3.720      0.000      -0.049      -0.015
# NSEI_EMA           -4.675e-05   2.09e-05     -2.239      0.025   -8.77e-05   -5.83e-06
# ======================================================================================
# """



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  IXIC_DAILY_RETURNS  2.096054
# 1   HSI_DAILY_RETURNS  1.258701
# 2  N225_DAILY_RETURNS  1.365054
# 3   VIX_DAILY_RETURNS  1.996535
# 4            NSEI_RSI  5.798566
# 5             DJI_RSI  5.715626
# 6            NSEI_TSI  6.041940
# 7             DJI_TSI  5.431534
# 8            NSEI_EMA  1.063067

# Drop NSEI_RSI



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
# Model:                          Logit   Df Residuals:                     1211
# Method:                           MLE   Df Model:                            8
# Date:                Sun, 26 May 2024   Pseudo R-squ.:                  0.1409
# Time:                        12:59:17   Log-Likelihood:                -657.45
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 3.318e-42
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             -0.8555      0.723     -1.184      0.237      -2.272       0.561
# IXIC_DAILY_RETURNS     0.4422      0.074      5.938      0.000       0.296       0.588
# HSI_DAILY_RETURNS     -0.1401      0.053     -2.655      0.008      -0.244      -0.037
# N225_DAILY_RETURNS    -0.1937      0.068     -2.864      0.004      -0.326      -0.061
# VIX_DAILY_RETURNS     -0.0419      0.013     -3.202      0.001      -0.068      -0.016
# DJI_RSI                0.0450      0.013      3.433      0.001       0.019       0.071
# NSEI_TSI               0.0053      0.003      1.514      0.130      -0.002       0.012
# DJI_TSI               -0.0239      0.008     -2.982      0.003      -0.040      -0.008
# NSEI_EMA           -4.145e-05   2.07e-05     -1.999      0.046   -8.21e-05   -8.03e-07
# ======================================================================================
# """

# Drop NSEI_TSI



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  IXIC_DAILY_RETURNS  2.076339
# 1   HSI_DAILY_RETURNS  1.245502
# 2  N225_DAILY_RETURNS  1.353420
# 3   VIX_DAILY_RETURNS  1.996270
# 4             DJI_RSI  4.857160
# 5            NSEI_TSI  1.406000
# 6             DJI_TSI  4.811514
# 7            NSEI_EMA  1.058090



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
# Date:                Sun, 26 May 2024   Pseudo R-squ.:                  0.1394
# Time:                        13:00:00   Log-Likelihood:                -658.60
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 1.764e-42
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             -0.8911      0.723     -1.233      0.217      -2.307       0.525
# IXIC_DAILY_RETURNS     0.4511      0.075      6.053      0.000       0.305       0.597
# HSI_DAILY_RETURNS     -0.1391      0.053     -2.637      0.008      -0.242      -0.036
# N225_DAILY_RETURNS    -0.1944      0.067     -2.882      0.004      -0.327      -0.062
# VIX_DAILY_RETURNS     -0.0408      0.013     -3.132      0.002      -0.066      -0.015
# DJI_RSI                0.0443      0.013      3.381      0.001       0.019       0.070
# DJI_TSI               -0.0205      0.008     -2.668      0.008      -0.036      -0.005
# NSEI_EMA           -3.392e-05   2.01e-05     -1.687      0.092   -7.33e-05    5.49e-06
# ======================================================================================
# """

# Drop NSEI_EMA



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  IXIC_DAILY_RETURNS  2.074242
# 1   HSI_DAILY_RETURNS  1.245074
# 2  N225_DAILY_RETURNS  1.353326
# 3   VIX_DAILY_RETURNS  1.996129
# 4             DJI_RSI  4.851326
# 5             DJI_TSI  4.379422
# 6            NSEI_EMA  1.001952



# %% 6 -
X_train = X_train.drop("NSEI_EMA", axis = 1)
X_dropped.append("NSEI_EMA")



# %% 7 -
model = Logit(y_train, X_train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1213
# Method:                           MLE   Df Model:                            6
# Date:                Sun, 26 May 2024   Pseudo R-squ.:                  0.1375
# Time:                        13:00:41   Log-Likelihood:                -660.02
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
X_train_pred = model.predict(X_train)

fpr, tpr, thresholds = roc_curve(y_train, X_train_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "02_01", "01 - training data", "phase_03")



# %% 9 - Optimal Threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold is : {optimal_threshold}')
# Best Threshold is : 0.684



# %% 10 - AUC Curve
auc_roc = roc_auc_score(y_train, X_train_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7529115595469844



# %% 11 - Classification Report
X_train_pred_class = np.where(X_train_pred <= optimal_threshold,  0, 1)
print(classification_report(y_train, X_train_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.53      0.68      0.60       391
#          1.0       0.83      0.72      0.77       829
# 
#     accuracy                           0.70      1220
#    macro avg       0.68      0.70      0.68      1220
# weighted avg       0.73      0.70      0.71      1220



# %% 11 - 
table = pd.crosstab(X_train_pred_class, y_train)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0              265  234
# 1              126  595



# %% 11 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.684 is : 71.77%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.684 is : 67.77%



# %% 12 - 
X_test = X_test.drop(X_dropped, axis = 1)



# %% 12 - ROC Curve
X_test_pred= model.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, X_test_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "02_02", "02 - testing data", "phase_03")



# %% 13 - AUC Curve
auc_roc = roc_auc_score(y_test, X_test_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7520816812053925



# %% 14 - Classification Report
X_test_pred_class = np.where(X_test_pred <= optimal_threshold,  0, 1)
print(classification_report(y_test, X_test_pred_class))
#               precision    recall  f1-score   support

#          0.0       0.53      0.65      0.58        97
#          1.0       0.82      0.73      0.77       208

#     accuracy                           0.70       305
#    macro avg       0.67      0.69      0.67       305
# weighted avg       0.72      0.70      0.71       305



# %% 11 - 
table = pd.crosstab(X_test_pred_class, y_test)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0               63   57
# 1               34  151



# %% 11 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.684 is : 72.6%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.684 is : 64.95%
