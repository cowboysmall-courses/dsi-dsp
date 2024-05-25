
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
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#                 Feature            VIF
# 0    NSEI_DAILY_RETURNS       2.176961
# 1     DJI_DAILY_RETURNS       4.715275
# 2    IXIC_DAILY_RETURNS       4.049968
# 3     HSI_DAILY_RETURNS       1.394166
# 4    N225_DAILY_RETURNS       1.500336
# 5   GDAXI_DAILY_RETURNS       1.995993
# 6     VIX_DAILY_RETURNS       2.206740
# 7         NSEI_HL_RATIO   22304.765593
# 8          DJI_HL_RATIO   22236.524507
# 9              NSEI_RSI     307.296454
# 10              DJI_RSI     281.985502
# 11             NSEI_ROC       7.499415
# 12              DJI_ROC       6.786881
# 13             NSEI_AWE      32.138657
# 14              DJI_AWE      33.602893
# 15             NSEI_KAM   26859.944630
# 16              DJI_KAM   22811.209509
# 17             NSEI_TSI      26.428099
# 18              DJI_TSI      20.530831
# 19             NSEI_VPT      37.092655
# 20              DJI_VPT      12.833567
# 21             NSEI_ULC      16.678018
# 22              DJI_ULC      19.448731
# 23             NSEI_SMA  203721.230327
# 24              DJI_SMA  177187.163394
# 25             NSEI_EMA  215769.367512
# 26              DJI_EMA  193535.632802



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
# Model:                          Logit   Df Residuals:                     1194
# Method:                           MLE   Df Model:                           25
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1574
# Time:                        15:34:56   Log-Likelihood:                -644.79
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 3.380e-37
# =======================================================================================
#                           coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# NSEI_DAILY_RETURNS     -0.0991      0.095     -1.044      0.297      -0.285       0.087
# DJI_DAILY_RETURNS      -0.0932      0.131     -0.709      0.478      -0.351       0.164
# IXIC_DAILY_RETURNS      0.4270      0.095      4.492      0.000       0.241       0.613
# HSI_DAILY_RETURNS      -0.1204      0.056     -2.138      0.032      -0.231      -0.010
# N225_DAILY_RETURNS     -0.1728      0.072     -2.409      0.016      -0.313      -0.032
# GDAXI_DAILY_RETURNS     0.0401      0.077      0.519      0.604      -0.111       0.191
# VIX_DAILY_RETURNS      -0.0387      0.014     -2.791      0.005      -0.066      -0.012
# NSEI_HL_RATIO           8.2130     10.514      0.781      0.435     -12.394      28.820
# DJI_HL_RATIO          -10.9523     10.511     -1.042      0.297     -31.553       9.648
# NSEI_RSI               -0.0361      0.022     -1.668      0.095      -0.079       0.006
# DJI_RSI                 0.0867      0.022      3.982      0.000       0.044       0.129
# NSEI_ROC               -0.0243      0.049     -0.492      0.623      -0.121       0.073
# DJI_ROC                 0.0379      0.047      0.811      0.418      -0.054       0.130
# NSEI_AWE                0.0005      0.001      0.540      0.589      -0.001       0.002
# DJI_AWE                -0.0006      0.001     -0.909      0.364      -0.002       0.001
# NSEI_KAM               -0.0006      0.001     -0.815      0.415      -0.002       0.001
# DJI_KAM                 0.0005      0.000      1.287      0.198      -0.000       0.001
# NSEI_TSI                0.0221      0.014      1.559      0.119      -0.006       0.050
# DJI_TSI                -0.0475      0.016     -3.030      0.002      -0.078      -0.017
# NSEI_VPT             -2.78e-06   2.92e-06     -0.953      0.341    -8.5e-06    2.94e-06
# NSEI_ULC               -0.0700      0.086     -0.814      0.416      -0.239       0.099
# DJI_ULC                -0.0459      0.082     -0.558      0.577      -0.207       0.115
# NSEI_SMA               -0.0036      0.002     -1.703      0.089      -0.008       0.001
# DJI_SMA                 0.0021      0.001      2.194      0.028       0.000       0.004
# NSEI_EMA                0.0041      0.002      1.897      0.058      -0.000       0.008
# DJI_EMA                -0.0025      0.001     -2.447      0.014      -0.004      -0.000
# =======================================================================================
# """

# Drop NSEI_ROC



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#                 Feature            VIF
# 0    NSEI_DAILY_RETURNS       2.175550
# 1     DJI_DAILY_RETURNS       4.666297
# 2    IXIC_DAILY_RETURNS       4.046455
# 3     HSI_DAILY_RETURNS       1.393417
# 4    N225_DAILY_RETURNS       1.500229
# 5   GDAXI_DAILY_RETURNS       1.995751
# 6     VIX_DAILY_RETURNS       2.206730
# 7         NSEI_HL_RATIO   22303.031313
# 8          DJI_HL_RATIO   22210.402597
# 9              NSEI_RSI     306.825174
# 10              DJI_RSI     278.784295
# 11             NSEI_ROC       7.397570
# 12              DJI_ROC       6.765315
# 13             NSEI_AWE      32.107273
# 14              DJI_AWE      32.505485
# 15             NSEI_KAM   26064.100508
# 16              DJI_KAM   22807.932899
# 17             NSEI_TSI      26.316917
# 18              DJI_TSI      20.131305
# 19             NSEI_VPT      34.425891
# 20             NSEI_ULC      16.257484
# 21              DJI_ULC      16.847476
# 22             NSEI_SMA  202161.202252
# 23              DJI_SMA  170123.072964
# 24             NSEI_EMA  215744.844288
# 25              DJI_EMA  186486.755059



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
# Model:                          Logit   Df Residuals:                     1195
# Method:                           MLE   Df Model:                           24
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1572
# Time:                        15:36:55   Log-Likelihood:                -644.91
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 1.173e-37
# =======================================================================================
#                           coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# NSEI_DAILY_RETURNS     -0.1033      0.095     -1.090      0.275      -0.289       0.082
# DJI_DAILY_RETURNS      -0.0919      0.131     -0.701      0.484      -0.349       0.165
# IXIC_DAILY_RETURNS      0.4260      0.095      4.481      0.000       0.240       0.612
# HSI_DAILY_RETURNS      -0.1195      0.056     -2.125      0.034      -0.230      -0.009
# N225_DAILY_RETURNS     -0.1720      0.072     -2.401      0.016      -0.313      -0.032
# GDAXI_DAILY_RETURNS     0.0414      0.077      0.536      0.592      -0.110       0.193
# VIX_DAILY_RETURNS      -0.0388      0.014     -2.802      0.005      -0.066      -0.012
# NSEI_HL_RATIO           8.2315     10.520      0.782      0.434     -12.387      28.850
# DJI_HL_RATIO          -10.8891     10.516     -1.036      0.300     -31.499       9.721
# NSEI_RSI               -0.0397      0.020     -1.954      0.051      -0.080       0.000
# DJI_RSI                 0.0879      0.022      4.061      0.000       0.045       0.130
# DJI_ROC                 0.0267      0.041      0.651      0.515      -0.054       0.107
# NSEI_AWE                0.0003      0.001      0.403      0.687      -0.001       0.002
# DJI_AWE                -0.0006      0.001     -0.888      0.375      -0.002       0.001
# NSEI_KAM               -0.0006      0.001     -0.849      0.396      -0.002       0.001
# DJI_KAM                 0.0004      0.000      1.260      0.208      -0.000       0.001
# NSEI_TSI                0.0238      0.014      1.729      0.084      -0.003       0.051
# DJI_TSI                -0.0470      0.016     -3.005      0.003      -0.078      -0.016
# NSEI_VPT            -2.865e-06   2.91e-06     -0.985      0.325   -8.57e-06    2.84e-06
# NSEI_ULC               -0.0695      0.086     -0.807      0.420      -0.238       0.099
# DJI_ULC                -0.0451      0.082     -0.549      0.583      -0.206       0.116
# NSEI_SMA               -0.0033      0.002     -1.632      0.103      -0.007       0.001
# DJI_SMA                 0.0020      0.001      2.137      0.033       0.000       0.004
# NSEI_EMA                0.0039      0.002      1.835      0.067      -0.000       0.008
# DJI_EMA                -0.0024      0.001     -2.397      0.017      -0.004      -0.000
# =======================================================================================
# """

# Drop NSEI_AWE



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#                 Feature            VIF
# 0    NSEI_DAILY_RETURNS       2.148712
# 1     DJI_DAILY_RETURNS       4.662963
# 2    IXIC_DAILY_RETURNS       4.043012
# 3     HSI_DAILY_RETURNS       1.391740
# 4    N225_DAILY_RETURNS       1.498169
# 5   GDAXI_DAILY_RETURNS       1.992000
# 6     VIX_DAILY_RETURNS       2.205721
# 7         NSEI_HL_RATIO   22300.236458
# 8          DJI_HL_RATIO   22196.236374
# 9              NSEI_RSI     270.208622
# 10              DJI_RSI     274.360711
# 11              DJI_ROC       4.995783
# 12             NSEI_AWE      29.047334
# 13              DJI_AWE      32.455635
# 14             NSEI_KAM   25841.423783
# 15              DJI_KAM   22752.775590
# 16             NSEI_TSI      24.785568
# 17              DJI_TSI      20.042767
# 18             NSEI_VPT      34.274292
# 19             NSEI_ULC      16.250849
# 20              DJI_ULC      16.845262
# 21             NSEI_SMA  188644.007893
# 22              DJI_SMA  162777.923506
# 23             NSEI_EMA  205968.941888
# 24              DJI_EMA  178004.234410



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
# Model:                          Logit   Df Residuals:                     1196
# Method:                           MLE   Df Model:                           23
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1571
# Time:                        15:38:26   Log-Likelihood:                -644.99
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 3.845e-38
# =======================================================================================
#                           coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# NSEI_DAILY_RETURNS     -0.1060      0.094     -1.122      0.262      -0.291       0.079
# DJI_DAILY_RETURNS      -0.0915      0.131     -0.697      0.486      -0.349       0.166
# IXIC_DAILY_RETURNS      0.4272      0.095      4.494      0.000       0.241       0.613
# HSI_DAILY_RETURNS      -0.1190      0.056     -2.116      0.034      -0.229      -0.009
# N225_DAILY_RETURNS     -0.1707      0.072     -2.386      0.017      -0.311      -0.030
# GDAXI_DAILY_RETURNS     0.0422      0.077      0.546      0.585      -0.109       0.193
# VIX_DAILY_RETURNS      -0.0387      0.014     -2.791      0.005      -0.066      -0.012
# NSEI_HL_RATIO           8.6198     10.478      0.823      0.411     -11.916      29.156
# DJI_HL_RATIO          -11.2880     10.470     -1.078      0.281     -31.809       9.233
# NSEI_RSI               -0.0404      0.020     -1.996      0.046      -0.080      -0.001
# DJI_RSI                 0.0891      0.021      4.157      0.000       0.047       0.131
# DJI_ROC                 0.0217      0.039      0.556      0.578      -0.055       0.098
# DJI_AWE                -0.0004      0.000     -0.920      0.358      -0.001       0.000
# NSEI_KAM               -0.0006      0.001     -0.802      0.422      -0.002       0.001
# DJI_KAM                 0.0004      0.000      1.259      0.208      -0.000       0.001
# NSEI_TSI                0.0270      0.011      2.432      0.015       0.005       0.049
# DJI_TSI                -0.0498      0.014     -3.584      0.000      -0.077      -0.023
# NSEI_VPT            -2.802e-06    2.9e-06     -0.966      0.334   -8.49e-06    2.89e-06
# NSEI_ULC               -0.0767      0.084     -0.911      0.362      -0.242       0.088
# DJI_ULC                -0.0372      0.080     -0.466      0.641      -0.194       0.119
# NSEI_SMA               -0.0035      0.002     -1.763      0.078      -0.007       0.000
# DJI_SMA                 0.0021      0.001      2.217      0.027       0.000       0.004
# NSEI_EMA                0.0040      0.002      1.924      0.054   -7.63e-05       0.008
# DJI_EMA                -0.0024      0.001     -2.471      0.013      -0.004      -0.001
# =======================================================================================
# """

# Drop DJI_ULC



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#                 Feature            VIF
# 0    NSEI_DAILY_RETURNS       2.132974
# 1     DJI_DAILY_RETURNS       4.662537
# 2    IXIC_DAILY_RETURNS       4.042133
# 3     HSI_DAILY_RETURNS       1.391074
# 4    N225_DAILY_RETURNS       1.495038
# 5   GDAXI_DAILY_RETURNS       1.990559
# 6     VIX_DAILY_RETURNS       2.205281
# 7         NSEI_HL_RATIO   22097.223036
# 8          DJI_HL_RATIO   21967.606476
# 9              NSEI_RSI     268.541227
# 10              DJI_RSI     266.978575
# 11              DJI_ROC       4.476373
# 12              DJI_AWE      12.040403
# 13             NSEI_KAM   25345.602286
# 14              DJI_KAM   22752.526334
# 15             NSEI_TSI      15.892397
# 16              DJI_TSI      15.301046
# 17             NSEI_VPT      34.193361
# 18             NSEI_ULC      15.459747
# 19              DJI_ULC      15.749667
# 20             NSEI_SMA  179271.606501
# 21              DJI_SMA  159057.573074
# 22             NSEI_EMA  200653.694498
# 23              DJI_EMA  174442.153757



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
# Model:                          Logit   Df Residuals:                     1197
# Method:                           MLE   Df Model:                           22
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1570
# Time:                        15:40:01   Log-Likelihood:                -645.10
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 1.264e-38
# =======================================================================================
#                           coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# NSEI_DAILY_RETURNS     -0.1069      0.094     -1.132      0.258      -0.292       0.078
# DJI_DAILY_RETURNS      -0.0867      0.131     -0.659      0.510      -0.344       0.171
# IXIC_DAILY_RETURNS      0.4272      0.095      4.494      0.000       0.241       0.613
# HSI_DAILY_RETURNS      -0.1193      0.056     -2.122      0.034      -0.229      -0.009
# N225_DAILY_RETURNS     -0.1692      0.072     -2.366      0.018      -0.309      -0.029
# GDAXI_DAILY_RETURNS     0.0424      0.077      0.549      0.583      -0.109       0.194
# VIX_DAILY_RETURNS      -0.0382      0.014     -2.760      0.006      -0.065      -0.011
# NSEI_HL_RATIO           9.8001     10.172      0.963      0.335     -10.138      29.738
# DJI_HL_RATIO          -12.5438     10.125     -1.239      0.215     -32.389       7.301
# NSEI_RSI               -0.0403      0.020     -1.991      0.046      -0.080      -0.001
# DJI_RSI                 0.0884      0.021      4.133      0.000       0.046       0.130
# DJI_ROC                 0.0243      0.039      0.626      0.531      -0.052       0.100
# DJI_AWE                -0.0003      0.000     -0.851      0.395      -0.001       0.000
# NSEI_KAM               -0.0006      0.001     -0.804      0.421      -0.002       0.001
# DJI_KAM                 0.0004      0.000      1.200      0.230      -0.000       0.001
# NSEI_TSI                0.0259      0.011      2.391      0.017       0.005       0.047
# DJI_TSI                -0.0480      0.013     -3.601      0.000      -0.074      -0.022
# NSEI_VPT            -2.707e-06   2.89e-06     -0.935      0.350   -8.38e-06    2.97e-06
# NSEI_ULC               -0.0995      0.068     -1.453      0.146      -0.234       0.035
# NSEI_SMA               -0.0036      0.002     -1.800      0.072      -0.008       0.000
# DJI_SMA                 0.0022      0.001      2.371      0.018       0.000       0.004
# NSEI_EMA                0.0041      0.002      1.956      0.050   -8.05e-06       0.008
# DJI_EMA                -0.0025      0.001     -2.562      0.010      -0.004      -0.001
# =======================================================================================
# """

# Drop GDAXI_DAILY_RETURNS



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#                 Feature            VIF
# 0    NSEI_DAILY_RETURNS       2.132770
# 1     DJI_DAILY_RETURNS       4.653008
# 2    IXIC_DAILY_RETURNS       4.041291
# 3     HSI_DAILY_RETURNS       1.390014
# 4    N225_DAILY_RETURNS       1.491480
# 5   GDAXI_DAILY_RETURNS       1.989196
# 6     VIX_DAILY_RETURNS       2.200762
# 7         NSEI_HL_RATIO   20814.495073
# 8          DJI_HL_RATIO   20552.501973
# 9              NSEI_RSI     268.382140
# 10              DJI_RSI     266.616957
# 11              DJI_ROC       4.387119
# 12              DJI_AWE      11.612490
# 13             NSEI_KAM   25343.172290
# 14              DJI_KAM   22293.712001
# 15             NSEI_TSI      15.270973
# 16              DJI_TSI      14.282338
# 17             NSEI_VPT      33.730081
# 18             NSEI_ULC      10.297632
# 19             NSEI_SMA  177758.045912
# 20              DJI_SMA  149564.217116
# 21             NSEI_EMA  199465.459050
# 22              DJI_EMA  169224.897912



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
# Model:                          Logit   Df Residuals:                     1198
# Method:                           MLE   Df Model:                           21
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1568
# Time:                        15:41:24   Log-Likelihood:                -645.25
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 4.225e-39
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# NSEI_DAILY_RETURNS    -0.0945      0.092     -1.032      0.302      -0.274       0.085
# DJI_DAILY_RETURNS     -0.0653      0.125     -0.522      0.601      -0.310       0.180
# IXIC_DAILY_RETURNS     0.4249      0.095      4.478      0.000       0.239       0.611
# HSI_DAILY_RETURNS     -0.1153      0.056     -2.069      0.039      -0.224      -0.006
# N225_DAILY_RETURNS    -0.1681      0.071     -2.351      0.019      -0.308      -0.028
# VIX_DAILY_RETURNS     -0.0390      0.014     -2.840      0.005      -0.066      -0.012
# NSEI_HL_RATIO          9.2115     10.129      0.909      0.363     -10.641      29.064
# DJI_HL_RATIO         -11.9482     10.080     -1.185      0.236     -31.705       7.809
# NSEI_RSI              -0.0409      0.020     -2.025      0.043      -0.081      -0.001
# DJI_RSI                0.0889      0.021      4.161      0.000       0.047       0.131
# DJI_ROC                0.0236      0.039      0.609      0.543      -0.052       0.100
# DJI_AWE               -0.0003      0.000     -0.852      0.394      -0.001       0.000
# NSEI_KAM              -0.0006      0.001     -0.771      0.441      -0.002       0.001
# DJI_KAM                0.0004      0.000      1.236      0.217      -0.000       0.001
# NSEI_TSI               0.0262      0.011      2.418      0.016       0.005       0.047
# DJI_TSI               -0.0481      0.013     -3.612      0.000      -0.074      -0.022
# NSEI_VPT           -2.707e-06   2.89e-06     -0.936      0.349   -8.38e-06    2.96e-06
# NSEI_ULC              -0.0990      0.068     -1.448      0.148      -0.233       0.035
# NSEI_SMA              -0.0036      0.002     -1.804      0.071      -0.008       0.000
# DJI_SMA                0.0022      0.001      2.372      0.018       0.000       0.004
# NSEI_EMA               0.0041      0.002      1.947      0.051   -2.63e-05       0.008
# DJI_EMA               -0.0025      0.001     -2.576      0.010      -0.004      -0.001
# ======================================================================================
# """

# Drop DJI_DAILY_RETURNS



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#                Feature            VIF
# 0   NSEI_DAILY_RETURNS       2.006108
# 1    DJI_DAILY_RETURNS       4.178254
# 2   IXIC_DAILY_RETURNS       4.036743
# 3    HSI_DAILY_RETURNS       1.360958
# 4   N225_DAILY_RETURNS       1.487234
# 5    VIX_DAILY_RETURNS       2.176562
# 6        NSEI_HL_RATIO   20699.710809
# 7         DJI_HL_RATIO   20439.547864
# 8             NSEI_RSI     267.712613
# 9              DJI_RSI     266.111831
# 10             DJI_ROC       4.381888
# 11             DJI_AWE      11.610301
# 12            NSEI_KAM   25240.266310
# 13             DJI_KAM   22212.800576
# 14            NSEI_TSI      15.257807
# 15             DJI_TSI      14.278022
# 16            NSEI_VPT      33.717673
# 17            NSEI_ULC      10.297632
# 18            NSEI_SMA  177749.890957
# 19             DJI_SMA  149557.580920
# 20            NSEI_EMA  199414.692257
# 21             DJI_EMA  169093.039129



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
# Model:                          Logit   Df Residuals:                     1199
# Method:                           MLE   Df Model:                           20
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1566
# Time:                        15:42:53   Log-Likelihood:                -645.39
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 1.357e-39
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# NSEI_DAILY_RETURNS    -0.1016      0.091     -1.115      0.265      -0.280       0.077
# IXIC_DAILY_RETURNS     0.3958      0.076      5.180      0.000       0.246       0.546
# HSI_DAILY_RETURNS     -0.1112      0.055     -2.017      0.044      -0.219      -0.003
# N225_DAILY_RETURNS    -0.1715      0.071     -2.401      0.016      -0.311      -0.031
# VIX_DAILY_RETURNS     -0.0376      0.014     -2.778      0.005      -0.064      -0.011
# NSEI_HL_RATIO          9.4388     10.149      0.930      0.352     -10.452      29.330
# DJI_HL_RATIO         -12.0978     10.103     -1.197      0.231     -31.899       7.704
# NSEI_RSI              -0.0393      0.020     -1.964      0.050      -0.078   -8.39e-05
# DJI_RSI                0.0857      0.020      4.179      0.000       0.045       0.126
# DJI_ROC                0.0216      0.039      0.554      0.579      -0.055       0.098
# DJI_AWE               -0.0003      0.000     -0.843      0.399      -0.001       0.000
# NSEI_KAM              -0.0006      0.001     -0.801      0.423      -0.002       0.001
# DJI_KAM                0.0004      0.000      1.230      0.219      -0.000       0.001
# NSEI_TSI               0.0254      0.011      2.368      0.018       0.004       0.046
# DJI_TSI               -0.0464      0.013     -3.584      0.000      -0.072      -0.021
# NSEI_VPT           -2.765e-06   2.89e-06     -0.956      0.339   -8.44e-06     2.9e-06
# NSEI_ULC              -0.0987      0.069     -1.439      0.150      -0.233       0.036
# NSEI_SMA              -0.0035      0.002     -1.761      0.078      -0.007       0.000
# DJI_SMA                0.0021      0.001      2.311      0.021       0.000       0.004
# NSEI_EMA               0.0040      0.002      1.914      0.056   -9.51e-05       0.008
# DJI_EMA               -0.0024      0.001     -2.521      0.012      -0.004      -0.001
# ======================================================================================
# """

# Drop DJI_ROC



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#                Feature            VIF
# 0   NSEI_DAILY_RETURNS       1.958651
# 1   IXIC_DAILY_RETURNS       2.178000
# 2    HSI_DAILY_RETURNS       1.343312
# 3   N225_DAILY_RETURNS       1.470163
# 4    VIX_DAILY_RETURNS       2.128694
# 5        NSEI_HL_RATIO   20642.145814
# 6         DJI_HL_RATIO   20397.782709
# 7             NSEI_RSI     263.213916
# 8              DJI_RSI     255.658357
# 9              DJI_ROC       4.334257
# 10             DJI_AWE      11.603030
# 11            NSEI_KAM   25122.986054
# 12             DJI_KAM   22212.131697
# 13            NSEI_TSI      15.007320
# 14             DJI_TSI      13.909426
# 15            NSEI_VPT      33.709002
# 16            NSEI_ULC      10.284206
# 17            NSEI_SMA  177370.872173
# 18             DJI_SMA  147708.626976
# 19            NSEI_EMA  199347.181406
# 20             DJI_EMA  167197.208245



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
# Model:                          Logit   Df Residuals:                     1200
# Method:                           MLE   Df Model:                           19
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1564
# Time:                        15:44:51   Log-Likelihood:                -645.54
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 4.336e-40
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# NSEI_DAILY_RETURNS    -0.0987      0.091     -1.082      0.279      -0.277       0.080
# IXIC_DAILY_RETURNS     0.4016      0.076      5.303      0.000       0.253       0.550
# HSI_DAILY_RETURNS     -0.1132      0.055     -2.058      0.040      -0.221      -0.005
# N225_DAILY_RETURNS    -0.1671      0.071     -2.355      0.019      -0.306      -0.028
# VIX_DAILY_RETURNS     -0.0372      0.014     -2.749      0.006      -0.064      -0.011
# NSEI_HL_RATIO          9.6669     10.144      0.953      0.341     -10.214      29.548
# DJI_HL_RATIO         -12.3776     10.094     -1.226      0.220     -32.162       7.407
# NSEI_RSI              -0.0403      0.020     -2.020      0.043      -0.079      -0.001
# DJI_RSI                0.0891      0.020      4.567      0.000       0.051       0.127
# DJI_AWE               -0.0002      0.000     -0.642      0.521      -0.001       0.000
# NSEI_KAM              -0.0005      0.001     -0.748      0.454      -0.002       0.001
# DJI_KAM                0.0005      0.000      1.347      0.178      -0.000       0.001
# NSEI_TSI               0.0254      0.011      2.359      0.018       0.004       0.046
# DJI_TSI               -0.0482      0.013     -3.838      0.000      -0.073      -0.024
# NSEI_VPT            -2.62e-06   2.88e-06     -0.909      0.363   -8.27e-06    3.03e-06
# NSEI_ULC              -0.0904      0.067     -1.355      0.176      -0.221       0.040
# NSEI_SMA              -0.0036      0.002     -1.803      0.071      -0.007       0.000
# DJI_SMA                0.0020      0.001      2.247      0.025       0.000       0.004
# NSEI_EMA               0.0040      0.002      1.932      0.053   -5.96e-05       0.008
# DJI_EMA               -0.0023      0.001     -2.466      0.014      -0.004      -0.000
# ======================================================================================
# """

#  Drop DJI_AWE



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#                Feature            VIF
# 0   NSEI_DAILY_RETURNS       1.947747
# 1   IXIC_DAILY_RETURNS       2.128245
# 2    HSI_DAILY_RETURNS       1.337137
# 3   N225_DAILY_RETURNS       1.449116
# 4    VIX_DAILY_RETURNS       2.122953
# 5        NSEI_HL_RATIO   20588.779509
# 6         DJI_HL_RATIO   20334.461950
# 7             NSEI_RSI     260.100471
# 8              DJI_RSI     232.154003
# 9              DJI_AWE       7.780363
# 10            NSEI_KAM   24761.608222
# 11             DJI_KAM   21473.596140
# 12            NSEI_TSI      15.004505
# 13             DJI_TSI      13.030121
# 14            NSEI_VPT      33.227132
# 15            NSEI_ULC       9.693752
# 16            NSEI_SMA  175939.859030
# 17             DJI_SMA  139856.909041
# 18            NSEI_EMA  199024.180393
# 19             DJI_EMA  163713.220607



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
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1561
# Time:                        15:46:18   Log-Likelihood:                -645.75
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 1.414e-40
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# NSEI_DAILY_RETURNS    -0.1004      0.091     -1.101      0.271      -0.279       0.078
# IXIC_DAILY_RETURNS     0.4044      0.076      5.355      0.000       0.256       0.552
# HSI_DAILY_RETURNS     -0.1147      0.055     -2.085      0.037      -0.223      -0.007
# N225_DAILY_RETURNS    -0.1639      0.071     -2.312      0.021      -0.303      -0.025
# VIX_DAILY_RETURNS     -0.0367      0.013     -2.727      0.006      -0.063      -0.010
# NSEI_HL_RATIO         10.0746     10.138      0.994      0.320      -9.795      29.944
# DJI_HL_RATIO         -12.5608     10.102     -1.243      0.214     -32.361       7.239
# NSEI_RSI              -0.0406      0.020     -2.039      0.041      -0.080      -0.002
# DJI_RSI                0.0892      0.020      4.571      0.000       0.051       0.128
# NSEI_KAM              -0.0006      0.001     -0.848      0.396      -0.002       0.001
# DJI_KAM                0.0005      0.000      1.506      0.132      -0.000       0.001
# NSEI_TSI               0.0248      0.011      2.319      0.020       0.004       0.046
# DJI_TSI               -0.0517      0.011     -4.564      0.000      -0.074      -0.029
# NSEI_VPT           -1.982e-06   2.71e-06     -0.731      0.465    -7.3e-06    3.33e-06
# NSEI_ULC              -0.0669      0.056     -1.194      0.232      -0.177       0.043
# NSEI_SMA              -0.0033      0.002     -1.701      0.089      -0.007       0.000
# DJI_SMA                0.0020      0.001      2.291      0.022       0.000       0.004
# NSEI_EMA               0.0038      0.002      1.846      0.065      -0.000       0.008
# DJI_EMA               -0.0024      0.001     -2.584      0.010      -0.004      -0.001
# ======================================================================================
# """

# Drop NSEI_VPT



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#                Feature            VIF
# 0   NSEI_DAILY_RETURNS       1.946329
# 1   IXIC_DAILY_RETURNS       2.128213
# 2    HSI_DAILY_RETURNS       1.336322
# 3   N225_DAILY_RETURNS       1.441343
# 4    VIX_DAILY_RETURNS       2.118893
# 5        NSEI_HL_RATIO   20547.637970
# 6         DJI_HL_RATIO   20331.104896
# 7             NSEI_RSI     260.080097
# 8              DJI_RSI     231.794678
# 9             NSEI_KAM   24036.207035
# 10             DJI_KAM   20380.244444
# 11            NSEI_TSI      14.916381
# 12             DJI_TSI      10.384893
# 13            NSEI_VPT      28.917732
# 14            NSEI_ULC       6.184716
# 15            NSEI_SMA  163436.786812
# 16             DJI_SMA  139124.075783
# 17            NSEI_EMA  191290.056744
# 18             DJI_EMA  159520.997349



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
# Model:                          Logit   Df Residuals:                     1202
# Method:                           MLE   Df Model:                           17
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1558
# Time:                        15:47:49   Log-Likelihood:                -646.01
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 4.755e-41
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# NSEI_DAILY_RETURNS    -0.1062      0.091     -1.171      0.241      -0.284       0.071
# IXIC_DAILY_RETURNS     0.4026      0.075      5.347      0.000       0.255       0.550
# HSI_DAILY_RETURNS     -0.1145      0.055     -2.082      0.037      -0.222      -0.007
# N225_DAILY_RETURNS    -0.1624      0.071     -2.297      0.022      -0.301      -0.024
# VIX_DAILY_RETURNS     -0.0369      0.013     -2.743      0.006      -0.063      -0.011
# NSEI_HL_RATIO         10.4855     10.107      1.037      0.300      -9.323      30.294
# DJI_HL_RATIO         -12.2498     10.075     -1.216      0.224     -31.997       7.497
# NSEI_RSI              -0.0399      0.020     -2.009      0.044      -0.079      -0.001
# DJI_RSI                0.0896      0.020      4.587      0.000       0.051       0.128
# NSEI_KAM              -0.0006      0.001     -0.828      0.408      -0.002       0.001
# DJI_KAM                0.0005      0.000      1.584      0.113      -0.000       0.001
# NSEI_TSI               0.0251      0.011      2.343      0.019       0.004       0.046
# DJI_TSI               -0.0514      0.011     -4.543      0.000      -0.074      -0.029
# NSEI_ULC              -0.0465      0.048     -0.960      0.337      -0.141       0.048
# NSEI_SMA              -0.0031      0.002     -1.629      0.103      -0.007       0.001
# DJI_SMA                0.0021      0.001      2.374      0.018       0.000       0.004
# NSEI_EMA               0.0036      0.002      1.763      0.078      -0.000       0.008
# DJI_EMA               -0.0025      0.001     -2.731      0.006      -0.004      -0.001
# ======================================================================================
# """

# Drop NSEI_KAM



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#                Feature            VIF
# 0   NSEI_DAILY_RETURNS       1.912816
# 1   IXIC_DAILY_RETURNS       2.127833
# 2    HSI_DAILY_RETURNS       1.335981
# 3   N225_DAILY_RETURNS       1.440928
# 4    VIX_DAILY_RETURNS       2.116789
# 5        NSEI_HL_RATIO   20417.196149
# 6         DJI_HL_RATIO   20321.503277
# 7             NSEI_RSI     258.592453
# 8              DJI_RSI     231.792783
# 9             NSEI_KAM   24025.570092
# 10             DJI_KAM   20148.210200
# 11            NSEI_TSI      14.916020
# 12             DJI_TSI      10.356087
# 13            NSEI_ULC       4.604488
# 14            NSEI_SMA  160631.691076
# 15             DJI_SMA  138346.064896
# 16            NSEI_EMA  187216.182511
# 17             DJI_EMA  156367.857434



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
# Model:                          Logit   Df Residuals:                     1203
# Method:                           MLE   Df Model:                           16
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1553
# Time:                        15:49:08   Log-Likelihood:                -646.36
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 1.668e-41
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# NSEI_DAILY_RETURNS    -0.1055      0.091     -1.165      0.244      -0.283       0.072
# IXIC_DAILY_RETURNS     0.4015      0.075      5.343      0.000       0.254       0.549
# HSI_DAILY_RETURNS     -0.1148      0.055     -2.085      0.037      -0.223      -0.007
# N225_DAILY_RETURNS    -0.1643      0.071     -2.324      0.020      -0.303      -0.026
# VIX_DAILY_RETURNS     -0.0368      0.013     -2.741      0.006      -0.063      -0.010
# NSEI_HL_RATIO         10.1191     10.107      1.001      0.317      -9.691      29.929
# DJI_HL_RATIO         -11.8246     10.070     -1.174      0.240     -31.561       7.911
# NSEI_RSI              -0.0404      0.020     -2.041      0.041      -0.079      -0.002
# DJI_RSI                0.0909      0.019      4.671      0.000       0.053       0.129
# DJI_KAM                0.0005      0.000      1.531      0.126      -0.000       0.001
# NSEI_TSI               0.0250      0.011      2.340      0.019       0.004       0.046
# DJI_TSI               -0.0523      0.011     -4.646      0.000      -0.074      -0.030
# NSEI_ULC              -0.0436      0.048     -0.902      0.367      -0.138       0.051
# NSEI_SMA              -0.0031      0.002     -1.642      0.101      -0.007       0.001
# DJI_SMA                0.0021      0.001      2.382      0.017       0.000       0.004
# NSEI_EMA               0.0030      0.002      1.578      0.115      -0.001       0.007
# DJI_EMA               -0.0025      0.001     -2.719      0.007      -0.004      -0.001
# ======================================================================================
# """

# Drop NSEI_ULC



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#                Feature            VIF
# 0   NSEI_DAILY_RETURNS       1.912815
# 1   IXIC_DAILY_RETURNS       2.125981
# 2    HSI_DAILY_RETURNS       1.335853
# 3   N225_DAILY_RETURNS       1.438418
# 4    VIX_DAILY_RETURNS       2.116019
# 5        NSEI_HL_RATIO   20378.067438
# 6         DJI_HL_RATIO   20274.815933
# 7             NSEI_RSI     258.474743
# 8              DJI_RSI     229.640900
# 9              DJI_KAM   20086.173854
# 10            NSEI_TSI      14.914887
# 11             DJI_TSI      10.208547
# 12            NSEI_ULC       4.543394
# 13            NSEI_SMA  160566.383972
# 14             DJI_SMA  138336.300394
# 15            NSEI_EMA  160580.624527
# 16             DJI_EMA  156357.414920



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
# Model:                          Logit   Df Residuals:                     1204
# Method:                           MLE   Df Model:                           15
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1548
# Time:                        15:50:22   Log-Likelihood:                -646.76
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 6.006e-42
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# NSEI_DAILY_RETURNS    -0.1176      0.089     -1.319      0.187      -0.292       0.057
# IXIC_DAILY_RETURNS     0.4075      0.076      5.379      0.000       0.259       0.556
# HSI_DAILY_RETURNS     -0.1163      0.055     -2.116      0.034      -0.224      -0.009
# N225_DAILY_RETURNS    -0.1632      0.071     -2.306      0.021      -0.302      -0.025
# VIX_DAILY_RETURNS     -0.0360      0.013     -2.682      0.007      -0.062      -0.010
# NSEI_HL_RATIO          8.6619      9.983      0.868      0.386     -10.905      28.229
# DJI_HL_RATIO         -10.7175     10.004     -1.071      0.284     -30.325       8.891
# NSEI_RSI              -0.0380      0.020     -1.939      0.053      -0.076       0.000
# DJI_RSI                0.0925      0.019      4.782      0.000       0.055       0.130
# DJI_KAM                0.0004      0.000      1.340      0.180      -0.000       0.001
# NSEI_TSI               0.0260      0.011      2.459      0.014       0.005       0.047
# DJI_TSI               -0.0523      0.011     -4.653      0.000      -0.074      -0.030
# NSEI_SMA              -0.0028      0.002     -1.502      0.133      -0.007       0.001
# DJI_SMA                0.0022      0.001      2.656      0.008       0.001       0.004
# NSEI_EMA               0.0027      0.002      1.439      0.150      -0.001       0.006
# DJI_EMA               -0.0026      0.001     -2.847      0.004      -0.004      -0.001
# ======================================================================================
# """

# Drop NSEI_HL_RATIO



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#                Feature            VIF
# 0   NSEI_DAILY_RETURNS       1.898173
# 1   IXIC_DAILY_RETURNS       2.125974
# 2    HSI_DAILY_RETURNS       1.335416
# 3   N225_DAILY_RETURNS       1.435516
# 4    VIX_DAILY_RETURNS       2.115413
# 5        NSEI_HL_RATIO   19941.986204
# 6         DJI_HL_RATIO   20030.083884
# 7             NSEI_RSI     256.695887
# 8              DJI_RSI     226.813798
# 9              DJI_KAM   18070.542374
# 10            NSEI_TSI      14.523563
# 11             DJI_TSI      10.207439
# 12            NSEI_SMA  158218.637242
# 13             DJI_SMA  130548.878217
# 14            NSEI_EMA  158304.944222
# 15             DJI_EMA  154473.434413



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
# Model:                          Logit   Df Residuals:                     1205
# Method:                           MLE   Df Model:                           14
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1543
# Time:                        15:51:29   Log-Likelihood:                -647.14
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 2.044e-42
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# NSEI_DAILY_RETURNS    -0.1063      0.088     -1.207      0.227      -0.279       0.066
# IXIC_DAILY_RETURNS     0.4110      0.076      5.429      0.000       0.263       0.559
# HSI_DAILY_RETURNS     -0.1197      0.055     -2.179      0.029      -0.227      -0.012
# N225_DAILY_RETURNS    -0.1631      0.071     -2.305      0.021      -0.302      -0.024
# VIX_DAILY_RETURNS     -0.0365      0.013     -2.721      0.007      -0.063      -0.010
# DJI_HL_RATIO          -2.1095      1.282     -1.646      0.100      -4.621       0.402
# NSEI_RSI              -0.0411      0.019     -2.135      0.033      -0.079      -0.003
# DJI_RSI                0.0946      0.019      4.933      0.000       0.057       0.132
# DJI_KAM                0.0004      0.000      1.395      0.163      -0.000       0.001
# NSEI_TSI               0.0274      0.010      2.614      0.009       0.007       0.048
# DJI_TSI               -0.0526      0.011     -4.682      0.000      -0.075      -0.031
# NSEI_SMA              -0.0029      0.002     -1.559      0.119      -0.007       0.001
# DJI_SMA                0.0022      0.001      2.642      0.008       0.001       0.004
# NSEI_EMA               0.0028      0.002      1.488      0.137      -0.001       0.006
# DJI_EMA               -0.0026      0.001     -2.845      0.004      -0.004      -0.001
# ======================================================================================
# """

# Drop NSEI_DAILY_RETURNS



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#                Feature            VIF
# 0   NSEI_DAILY_RETURNS       1.869813
# 1   IXIC_DAILY_RETURNS       2.122621
# 2    HSI_DAILY_RETURNS       1.330698
# 3   N225_DAILY_RETURNS       1.435036
# 4    VIX_DAILY_RETURNS       2.108313
# 5         DJI_HL_RATIO     368.023180
# 6             NSEI_RSI     250.080668
# 7              DJI_RSI     224.002040
# 8              DJI_KAM   17993.497387
# 9             NSEI_TSI      14.292906
# 10             DJI_TSI      10.207353
# 11            NSEI_SMA  157958.445660
# 12             DJI_SMA  130456.990885
# 13            NSEI_EMA  158128.142264
# 14             DJI_EMA  154445.814684



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
# Model:                          Logit   Df Residuals:                     1206
# Method:                           MLE   Df Model:                           13
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1534
# Time:                        15:52:30   Log-Likelihood:                -647.88
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 9.471e-43
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# IXIC_DAILY_RETURNS     0.4054      0.075      5.377      0.000       0.258       0.553
# HSI_DAILY_RETURNS     -0.1333      0.054     -2.480      0.013      -0.239      -0.028
# N225_DAILY_RETURNS    -0.1807      0.069     -2.614      0.009      -0.316      -0.045
# VIX_DAILY_RETURNS     -0.0348      0.013     -2.608      0.009      -0.061      -0.009
# DJI_HL_RATIO          -1.7430      1.244     -1.401      0.161      -4.181       0.695
# NSEI_RSI              -0.0535      0.016     -3.284      0.001      -0.085      -0.022
# DJI_RSI                0.0986      0.019      5.209      0.000       0.062       0.136
# DJI_KAM                0.0004      0.000      1.364      0.173      -0.000       0.001
# NSEI_TSI               0.0335      0.009      3.643      0.000       0.015       0.051
# DJI_TSI               -0.0547      0.011     -4.911      0.000      -0.077      -0.033
# NSEI_SMA              -0.0036      0.002     -2.012      0.044      -0.007   -9.33e-05
# DJI_SMA                0.0024      0.001      2.828      0.005       0.001       0.004
# NSEI_EMA               0.0035      0.002      1.938      0.053   -4.01e-05       0.007
# DJI_EMA               -0.0027      0.001     -2.999      0.003      -0.005      -0.001
# ======================================================================================
# """

# Drop DJI_KAM



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#                Feature            VIF
# 0   IXIC_DAILY_RETURNS       2.111824
# 1    HSI_DAILY_RETURNS       1.261529
# 2   N225_DAILY_RETURNS       1.382952
# 3    VIX_DAILY_RETURNS       2.075153
# 4         DJI_HL_RATIO     346.028888
# 5             NSEI_RSI     175.658162
# 6              DJI_RSI     215.720532
# 7              DJI_KAM   17987.199599
# 8             NSEI_TSI      10.911745
# 9              DJI_TSI       9.891672
# 10            NSEI_SMA  142502.324372
# 11             DJI_SMA  127046.954614
# 12            NSEI_EMA  142705.258112
# 13             DJI_EMA  151364.198852



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
# Model:                          Logit   Df Residuals:                     1207
# Method:                           MLE   Df Model:                           12
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1521
# Time:                        15:53:42   Log-Likelihood:                -648.81
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 5.123e-43
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# IXIC_DAILY_RETURNS     0.3984      0.075      5.340      0.000       0.252       0.545
# HSI_DAILY_RETURNS     -0.1317      0.054     -2.454      0.014      -0.237      -0.027
# N225_DAILY_RETURNS    -0.1764      0.069     -2.548      0.011      -0.312      -0.041
# VIX_DAILY_RETURNS     -0.0371      0.013     -2.807      0.005      -0.063      -0.011
# DJI_HL_RATIO          -1.4398      1.225     -1.176      0.240      -3.840       0.961
# NSEI_RSI              -0.0527      0.016     -3.244      0.001      -0.085      -0.021
# DJI_RSI                0.0955      0.019      5.097      0.000       0.059       0.132
# NSEI_TSI               0.0322      0.009      3.529      0.000       0.014       0.050
# DJI_TSI               -0.0521      0.011     -4.763      0.000      -0.074      -0.031
# NSEI_SMA              -0.0036      0.002     -2.012      0.044      -0.007   -9.26e-05
# DJI_SMA                0.0023      0.001      2.760      0.006       0.001       0.004
# NSEI_EMA               0.0035      0.002      1.947      0.052    -2.4e-05       0.007
# DJI_EMA               -0.0023      0.001     -2.680      0.007      -0.004      -0.001
# ======================================================================================
# """

# Drop DJI_HL_RATIO



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#                Feature            VIF
# 0   IXIC_DAILY_RETURNS       2.108989
# 1    HSI_DAILY_RETURNS       1.260910
# 2   N225_DAILY_RETURNS       1.378959
# 3    VIX_DAILY_RETURNS       2.057019
# 4         DJI_HL_RATIO     334.707798
# 5             NSEI_RSI     175.428435
# 6              DJI_RSI     212.279205
# 7             NSEI_TSI      10.790545
# 8              DJI_TSI       9.615259
# 9             NSEI_SMA  142498.788596
# 10             DJI_SMA  126424.334134
# 11            NSEI_EMA  142704.888111
# 12             DJI_EMA  127474.532583



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
# Model:                          Logit   Df Residuals:                     1208
# Method:                           MLE   Df Model:                           11
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1512
# Time:                        15:54:46   Log-Likelihood:                -649.51
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 2.109e-43
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# IXIC_DAILY_RETURNS     0.4082      0.074      5.487      0.000       0.262       0.554
# HSI_DAILY_RETURNS     -0.1285      0.054     -2.398      0.016      -0.234      -0.023
# N225_DAILY_RETURNS    -0.1577      0.067     -2.343      0.019      -0.290      -0.026
# VIX_DAILY_RETURNS     -0.0393      0.013     -2.997      0.003      -0.065      -0.014
# NSEI_RSI              -0.0595      0.015     -3.900      0.000      -0.089      -0.030
# DJI_RSI                0.0836      0.016      5.303      0.000       0.053       0.115
# NSEI_TSI               0.0357      0.009      4.134      0.000       0.019       0.053
# DJI_TSI               -0.0451      0.009     -4.919      0.000      -0.063      -0.027
# NSEI_SMA              -0.0040      0.002     -2.292      0.022      -0.007      -0.001
# DJI_SMA                0.0021      0.001      2.547      0.011       0.000       0.004
# NSEI_EMA               0.0039      0.002      2.250      0.024       0.001       0.007
# DJI_EMA               -0.0020      0.001     -2.483      0.013      -0.004      -0.000
# ======================================================================================
# """



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#                Feature            VIF
# 0   IXIC_DAILY_RETURNS       2.093481
# 1    HSI_DAILY_RETURNS       1.256888
# 2   N225_DAILY_RETURNS       1.320269
# 3    VIX_DAILY_RETURNS       1.998960
# 4             NSEI_RSI     151.959026
# 5              DJI_RSI     146.984544
# 6             NSEI_TSI       9.543749
# 7              DJI_TSI       6.642379
# 8             NSEI_SMA  136045.085901
# 9              DJI_SMA  118381.012645
# 10            NSEI_EMA  135265.118262
# 11             DJI_EMA  121527.544326

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
# Model:                          Logit   Df Residuals:                     1209
# Method:                           MLE   Df Model:                           10
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1478
# Time:                        15:56:19   Log-Likelihood:                -652.16
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 5.541e-43
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# IXIC_DAILY_RETURNS     0.4185      0.074      5.627      0.000       0.273       0.564
# HSI_DAILY_RETURNS     -0.1236      0.053     -2.330      0.020      -0.228      -0.020
# N225_DAILY_RETURNS    -0.1634      0.067     -2.442      0.015      -0.295      -0.032
# VIX_DAILY_RETURNS     -0.0413      0.013     -3.162      0.002      -0.067      -0.016
# NSEI_RSI              -0.0386      0.012     -3.189      0.001      -0.062      -0.015
# DJI_RSI                0.0666      0.014      4.829      0.000       0.040       0.094
# NSEI_TSI               0.0251      0.007      3.475      0.001       0.011       0.039
# DJI_TSI               -0.0359      0.008     -4.380      0.000      -0.052      -0.020
# DJI_SMA                0.0008      0.001      1.374      0.170      -0.000       0.002
# NSEI_EMA           -7.256e-05   5.26e-05     -1.379      0.168      -0.000    3.06e-05
# DJI_EMA               -0.0008      0.001     -1.311      0.190      -0.002       0.000
# ======================================================================================
# """

# Drop DJI_EMA



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#                Feature           VIF
# 0   IXIC_DAILY_RETURNS      2.089203
# 1    HSI_DAILY_RETURNS      1.255398
# 2   N225_DAILY_RETURNS      1.318968
# 3    VIX_DAILY_RETURNS      1.982588
# 4             NSEI_RSI    100.493757
# 5              DJI_RSI    116.134111
# 6             NSEI_TSI      6.953843
# 7              DJI_TSI      5.377924
# 8              DJI_SMA  69599.620334
# 9             NSEI_EMA    134.006596
# 10             DJI_EMA  73839.655051

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
# Model:                          Logit   Df Residuals:                     1210
# Method:                           MLE   Df Model:                            9
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1466
# Time:                        15:57:35   Log-Likelihood:                -653.02
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 2.464e-43
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# IXIC_DAILY_RETURNS     0.4200      0.074      5.640      0.000       0.274       0.566
# HSI_DAILY_RETURNS     -0.1229      0.053     -2.314      0.021      -0.227      -0.019
# N225_DAILY_RETURNS    -0.1670      0.067     -2.503      0.012      -0.298      -0.036
# VIX_DAILY_RETURNS     -0.0427      0.013     -3.277      0.001      -0.068      -0.017
# NSEI_RSI              -0.0384      0.012     -3.171      0.002      -0.062      -0.015
# DJI_RSI                0.0586      0.012      4.749      0.000       0.034       0.083
# NSEI_TSI               0.0253      0.007      3.504      0.000       0.011       0.039
# DJI_TSI               -0.0323      0.008     -4.192      0.000      -0.047      -0.017
# DJI_SMA              3.94e-05   3.52e-05      1.118      0.264   -2.97e-05       0.000
# NSEI_EMA           -9.595e-05   4.95e-05     -1.939      0.053      -0.000    1.05e-06
# ======================================================================================
# """

# Drop DJI_SMA



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#               Feature         VIF
# 0  IXIC_DAILY_RETURNS    2.088812
# 1   HSI_DAILY_RETURNS    1.255045
# 2  N225_DAILY_RETURNS    1.317844
# 3   VIX_DAILY_RETURNS    1.968773
# 4            NSEI_RSI  100.425137
# 5             DJI_RSI   95.790055
# 6            NSEI_TSI    6.934749
# 7             DJI_TSI    4.880558
# 8             DJI_SMA  247.533430
# 9            NSEI_EMA  117.320217



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
# Model:                          Logit   Df Residuals:                     1211
# Method:                           MLE   Df Model:                            8
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1458
# Time:                        15:58:37   Log-Likelihood:                -653.65
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 8.245e-44
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# IXIC_DAILY_RETURNS     0.4212      0.075      5.646      0.000       0.275       0.567
# HSI_DAILY_RETURNS     -0.1268      0.053     -2.389      0.017      -0.231      -0.023
# N225_DAILY_RETURNS    -0.1804      0.066     -2.749      0.006      -0.309      -0.052
# VIX_DAILY_RETURNS     -0.0409      0.013     -3.161      0.002      -0.066      -0.016
# NSEI_RSI              -0.0331      0.011     -2.978      0.003      -0.055      -0.011
# DJI_RSI                0.0622      0.012      5.208      0.000       0.039       0.086
# NSEI_TSI               0.0224      0.007      3.334      0.001       0.009       0.035
# DJI_TSI               -0.0329      0.008     -4.274      0.000      -0.048      -0.018
# NSEI_EMA           -4.485e-05   1.91e-05     -2.347      0.019   -8.23e-05   -7.39e-06
# ======================================================================================
# """



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#               Feature        VIF
# 0  IXIC_DAILY_RETURNS   2.088404
# 1   HSI_DAILY_RETURNS   1.251447
# 2  N225_DAILY_RETURNS   1.277643
# 3   VIX_DAILY_RETURNS   1.944567
# 4            NSEI_RSI  86.368964
# 5             DJI_RSI  88.768714
# 6            NSEI_TSI   6.035485
# 7             DJI_TSI   4.853301
# 8            NSEI_EMA  17.284518

# Drop DJI_RSI



# %% 6 -
X_train = X_train.drop("DJI_RSI", axis = 1)
X_dropped.append("DJI_RSI")



# %% 7 -
model = Logit(y_train, X_train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1212
# Method:                           MLE   Df Model:                            7
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1274
# Time:                        16:00:00   Log-Likelihood:                -667.75
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 1.341e-38
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# IXIC_DAILY_RETURNS     0.4853      0.074      6.564      0.000       0.340       0.630
# HSI_DAILY_RETURNS     -0.1392      0.052     -2.656      0.008      -0.242      -0.036
# N225_DAILY_RETURNS    -0.1482      0.065     -2.289      0.022      -0.275      -0.021
# VIX_DAILY_RETURNS     -0.0478      0.013     -3.736      0.000      -0.073      -0.023
# NSEI_RSI               0.0184      0.005      3.579      0.000       0.008       0.029
# NSEI_TSI              -0.0053      0.004     -1.316      0.188      -0.013       0.003
# DJI_TSI                0.0008      0.004      0.196      0.844      -0.007       0.009
# NSEI_EMA           -1.013e-05   1.77e-05     -0.572      0.567   -4.49e-05    2.46e-05
# ======================================================================================
# """

# Drop NSEI_TSI



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#               Feature        VIF
# 0  IXIC_DAILY_RETURNS   2.027583
# 1   HSI_DAILY_RETURNS   1.247623
# 2  N225_DAILY_RETURNS   1.269937
# 3   VIX_DAILY_RETURNS   1.923154
# 4            NSEI_RSI  19.152503
# 5            NSEI_TSI   2.286736
# 6             DJI_TSI   1.536396
# 7            NSEI_EMA  15.404920



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
# Model:                          Logit   Df Residuals:                     1213
# Method:                           MLE   Df Model:                            6
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1263
# Time:                        16:01:12   Log-Likelihood:                -668.62
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 5.249e-39
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# IXIC_DAILY_RETURNS     0.4809      0.074      6.526      0.000       0.336       0.625
# HSI_DAILY_RETURNS     -0.1382      0.052     -2.633      0.008      -0.241      -0.035
# N225_DAILY_RETURNS    -0.1427      0.065     -2.202      0.028      -0.270      -0.016
# VIX_DAILY_RETURNS     -0.0483      0.013     -3.769      0.000      -0.073      -0.023
# NSEI_RSI               0.0149      0.004      3.403      0.001       0.006       0.023
# DJI_TSI               -0.0007      0.004     -0.188      0.851      -0.008       0.007
# NSEI_EMA           -2.051e-07    1.6e-05     -0.013      0.990   -3.16e-05    3.12e-05
# ======================================================================================
# """

# Drop NSEI_EMA



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#               Feature        VIF
# 0  IXIC_DAILY_RETURNS   2.027304
# 1   HSI_DAILY_RETURNS   1.247219
# 2  N225_DAILY_RETURNS   1.264870
# 3   VIX_DAILY_RETURNS   1.923151
# 4            NSEI_RSI  13.768515
# 5             DJI_TSI   1.362604
# 6            NSEI_EMA  12.545776



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
# Model:                          Logit   Df Residuals:                     1214
# Method:                           MLE   Df Model:                            5
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1263
# Time:                        16:02:01   Log-Likelihood:                -668.62
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 7.994e-40
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# IXIC_DAILY_RETURNS     0.4809      0.074      6.537      0.000       0.337       0.625
# HSI_DAILY_RETURNS     -0.1382      0.052     -2.636      0.008      -0.241      -0.035
# N225_DAILY_RETURNS    -0.1426      0.065     -2.204      0.028      -0.269      -0.016
# VIX_DAILY_RETURNS     -0.0483      0.013     -3.772      0.000      -0.073      -0.023
# NSEI_RSI               0.0148      0.001     11.022      0.000       0.012       0.017
# DJI_TSI               -0.0007      0.004     -0.192      0.848      -0.008       0.007
# ======================================================================================
# """

# Drop DJI_TSI



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  IXIC_DAILY_RETURNS  2.027112
# 1   HSI_DAILY_RETURNS  1.242466
# 2  N225_DAILY_RETURNS  1.260812
# 3   VIX_DAILY_RETURNS  1.923146
# 4            NSEI_RSI  1.237201
# 5             DJI_TSI  1.238509



# %% 6 -
X_train = X_train.drop("DJI_TSI", axis = 1)
X_dropped.append("DJI_TSI")



# %% 7 -
model = Logit(y_train, X_train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1215
# Method:                           MLE   Df Model:                            4
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1262
# Time:                        16:03:55   Log-Likelihood:                -668.64
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 1.095e-40
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# IXIC_DAILY_RETURNS     0.4791      0.073      6.574      0.000       0.336       0.622
# HSI_DAILY_RETURNS     -0.1378      0.052     -2.631      0.009      -0.240      -0.035
# N225_DAILY_RETURNS    -0.1429      0.065     -2.209      0.027      -0.270      -0.016
# VIX_DAILY_RETURNS     -0.0485      0.013     -3.803      0.000      -0.073      -0.024
# NSEI_RSI               0.0147      0.001     12.256      0.000       0.012       0.017
# ======================================================================================
# """



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  IXIC_DAILY_RETURNS  2.015598
# 1   HSI_DAILY_RETURNS  1.242026
# 2  N225_DAILY_RETURNS  1.260440
# 3   VIX_DAILY_RETURNS  1.917015
# 4            NSEI_RSI  1.013337



# %% 8 - ROC Curve
X_train_pred = model.predict(X_train)

fpr, tpr, thresholds = roc_curve(y_train, X_train_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "02_01", "01 - training data", "phase_03")



# %% 9 - Optimal Threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold is : {optimal_threshold}')
# Best Threshold is : 0.649



# %% 10 - AUC Curve
auc_roc = roc_auc_score(y_train, X_train_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.748043277729616



# %% 11 - Classification Report
X_train_pred_class = np.where(X_train_pred <= optimal_threshold,  0, 1)
print(classification_report(y_train, X_train_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.57      0.61      0.59       391
#          1.0       0.81      0.78      0.80       829
# 
#     accuracy                           0.73      1220
#    macro avg       0.69      0.70      0.69      1220
# weighted avg       0.73      0.73      0.73      1220



# %% 11 - 
table = pd.crosstab(X_train_pred_class, y_train)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0              237  179
# 1              154  650



# %% 11 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.639 is : 78.41%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.639 is : 60.61%



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
# AUC ROC: 0.7566415543219667



# %% 14 - Classification Report
X_test_pred_class = np.where(X_test_pred <= optimal_threshold,  0, 1)
print(classification_report(y_test, X_test_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.60      0.61      0.60        97
#          1.0       0.82      0.81      0.81       208
# 
#     accuracy                           0.74       305
#    macro avg       0.71      0.71      0.71       305
# weighted avg       0.75      0.74      0.74       305



# %% 11 - 
table = pd.crosstab(X_test_pred_class, y_test)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0               59   40
# 1               38  168



# %% 11 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.639 is : 80.77%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.639 is : 60.82%
