
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



# %% 3 -
data = pd.concat([master["NSEI_OPEN_DIR"].shift(-1), master[ALL_COLS]], axis = 1)
data.dropna(inplace = True)
data.head()



# %% 3 -
X = data[ALL_COLS]
y = data['NSEI_OPEN_DIR']



# %% 4 -
model = Logit(y, X).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1525
# Model:                          Logit   Df Residuals:                     1498
# Method:                           MLE   Df Model:                           26
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1621
# Time:                        13:58:22   Log-Likelihood:                -801.00
# converged:                       True   LL-Null:                       -955.98
# Covariance Type:            nonrobust   LLR p-value:                 2.144e-50
# =======================================================================================
#                           coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# NSEI_DAILY_RETURNS     -0.1376      0.086     -1.603      0.109      -0.306       0.031
# DJI_DAILY_RETURNS      -0.0455      0.117     -0.388      0.698      -0.276       0.185
# IXIC_DAILY_RETURNS      0.4203      0.084      5.027      0.000       0.256       0.584
# HSI_DAILY_RETURNS      -0.1157      0.050     -2.295      0.022      -0.214      -0.017
# N225_DAILY_RETURNS     -0.1573      0.065     -2.436      0.015      -0.284      -0.031
# GDAXI_DAILY_RETURNS     0.0353      0.069      0.510      0.610      -0.101       0.171
# VIX_DAILY_RETURNS      -0.0392      0.013     -3.091      0.002      -0.064      -0.014
# NSEI_HL_RATIO          -3.6829      9.467     -0.389      0.697     -22.239      14.873
# DJI_HL_RATIO            0.6229      9.443      0.066      0.947     -17.885      19.131
# NSEI_RSI               -0.0307      0.019     -1.622      0.105      -0.068       0.006
# DJI_RSI                 0.0867      0.020      4.390      0.000       0.048       0.125
# NSEI_ROC               -0.0712      0.043     -1.652      0.099      -0.156       0.013
# DJI_ROC                 0.0391      0.041      0.945      0.345      -0.042       0.120
# NSEI_AWE                0.0004      0.001      0.522      0.602      -0.001       0.002
# DJI_AWE                -0.0004      0.001     -0.710      0.477      -0.002       0.001
# NSEI_KAM            -1.754e-05      0.001     -0.026      0.979      -0.001       0.001
# DJI_KAM                 0.0002      0.000      0.822      0.411      -0.000       0.001
# NSEI_TSI                0.0264      0.013      2.103      0.035       0.002       0.051
# DJI_TSI                -0.0554      0.014     -3.844      0.000      -0.084      -0.027
# NSEI_VPT            -2.867e-06   2.69e-06     -1.067      0.286   -8.13e-06     2.4e-06
# DJI_VPT              -5.75e-10   2.66e-09     -0.216      0.829   -5.79e-09    4.64e-09
# NSEI_ULC               -0.0295      0.077     -0.384      0.701      -0.180       0.121
# DJI_ULC                -0.0942      0.078     -1.213      0.225      -0.246       0.058
# NSEI_SMA               -0.0044      0.002     -2.320      0.020      -0.008      -0.001
# DJI_SMA                 0.0021      0.001      2.432      0.015       0.000       0.004
# NSEI_EMA                0.0043      0.002      2.219      0.026       0.001       0.008
# DJI_EMA                -0.0022      0.001     -2.476      0.013      -0.004      -0.000
# =======================================================================================
# """



# list of insignificant variables
# 
# NSEI_DAILY_RETURNS     -0.1376      0.086     -1.603      0.109      -0.306       0.031
# DJI_DAILY_RETURNS      -0.0455      0.117     -0.388      0.698      -0.276       0.185
# GDAXI_DAILY_RETURNS     0.0353      0.069      0.510      0.610      -0.101       0.171
# NSEI_HL_RATIO          -3.6829      9.467     -0.389      0.697     -22.239      14.873
# DJI_HL_RATIO            0.6229      9.443      0.066      0.947     -17.885      19.131
# NSEI_RSI               -0.0307      0.019     -1.622      0.105      -0.068       0.006
# NSEI_ROC               -0.0712      0.043     -1.652      0.099      -0.156       0.013
# DJI_ROC                 0.0391      0.041      0.945      0.345      -0.042       0.120
# NSEI_AWE                0.0004      0.001      0.522      0.602      -0.001       0.002
# DJI_AWE                -0.0004      0.001     -0.710      0.477      -0.002       0.001
# NSEI_KAM            -1.754e-05      0.001     -0.026      0.979      -0.001       0.001
# DJI_KAM                 0.0002      0.000      0.822      0.411      -0.000       0.001
# NSEI_VPT            -2.867e-06   2.69e-06     -1.067      0.286   -8.13e-06     2.4e-06
# DJI_VPT              -5.75e-10   2.66e-09     -0.216      0.829   -5.79e-09    4.64e-09
# NSEI_ULC               -0.0295      0.077     -0.384      0.701      -0.180       0.121
# DJI_ULC                -0.0942      0.078     -1.213      0.225      -0.246       0.058



# %% 5 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#                 Feature            VIF
# 0    NSEI_DAILY_RETURNS       2.053221
# 1     DJI_DAILY_RETURNS       4.600217
# 2    IXIC_DAILY_RETURNS       4.040925
# 3     HSI_DAILY_RETURNS       1.376547
# 4    N225_DAILY_RETURNS       1.430670
# 5   GDAXI_DAILY_RETURNS       1.887660
# 6     VIX_DAILY_RETURNS       2.280270
# 7         NSEI_HL_RATIO   19468.409450
# 8          DJI_HL_RATIO   19233.465757
# 9              NSEI_RSI     272.948531
# 10              DJI_RSI     282.060710
# 11             NSEI_ROC       7.833744
# 12              DJI_ROC       6.984218
# 13             NSEI_AWE      32.997542
# 14              DJI_AWE      34.244026
# 15             NSEI_KAM   26598.408682
# 16              DJI_KAM   21000.271095
# 17             NSEI_TSI      25.035203
# 18              DJI_TSI      20.579202
# 19             NSEI_VPT      36.843303
# 20              DJI_VPT      12.795937
# 21             NSEI_ULC      17.076637
# 22              DJI_ULC      19.522275
# 23             NSEI_SMA  200757.401758
# 24              DJI_SMA  171444.903163
# 25             NSEI_EMA  212525.885857
# 26              DJI_EMA  189079.030611



# list of colinear variables
# 
# 7         NSEI_HL_RATIO   19468.409450
# 8          DJI_HL_RATIO   19233.465757
# 9              NSEI_RSI     272.948531
# 10              DJI_RSI     282.060710
# 11             NSEI_ROC       7.833744
# 12              DJI_ROC       6.984218
# 13             NSEI_AWE      32.997542
# 14              DJI_AWE      34.244026
# 15             NSEI_KAM   26598.408682
# 16              DJI_KAM   21000.271095
# 17             NSEI_TSI      25.035203
# 18              DJI_TSI      20.579202
# 19             NSEI_VPT      36.843303
# 20              DJI_VPT      12.795937
# 21             NSEI_ULC      17.076637
# 22              DJI_ULC      19.522275
# 23             NSEI_SMA  200757.401758
# 24              DJI_SMA  171444.903163
# 25             NSEI_EMA  212525.885857
# 26              DJI_EMA  189079.030611



# %% 6 -
y_pred = model.predict(X)



# %% 7 - ROC Curve
fpr, tpr, thresholds = roc_curve(y, y_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "01_01", "01 - with all data", "phase_03")



# %% 8 - find optimal threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold is : {optimal_threshold}')
# Best Threshold is : 0.622



# %% 9 - AUC Curve
auc_roc = roc_auc_score(y, y_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7663223042509131



# %% 10 - Classification Report
y_pred_class = np.where(y_pred <= optimal_threshold,  0, 1)
print(classification_report(y, y_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.62      0.58      0.60       488
#          1.0       0.81      0.83      0.82      1037
# 
#     accuracy                           0.75      1525
#    macro avg       0.71      0.70      0.71      1525
# weighted avg       0.75      0.75      0.75      1525



# %% 11 - 
table = pd.crosstab(y_pred_class, y)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0              283  177
# 1              205  860



# %% 12 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.622 is : 82.93%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.622 is : 57.99%
