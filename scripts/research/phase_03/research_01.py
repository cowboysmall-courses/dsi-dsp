
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
model = Logit(y, X).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1525
# Model:                          Logit   Df Residuals:                     1497
# Method:                           MLE   Df Model:                           27
# Date:                Sun, 26 May 2024   Pseudo R-squ.:                  0.1659
# Time:                        12:22:24   Log-Likelihood:                -797.38
# converged:                       True   LL-Null:                       -955.98
# Covariance Type:            nonrobust   LLR p-value:                 2.685e-51
# =======================================================================================
#                           coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# Intercept              41.2311     14.601      2.824      0.005      12.615      69.848
# NSEI_DAILY_RETURNS     -0.1382      0.087     -1.594      0.111      -0.308       0.032
# DJI_DAILY_RETURNS      -0.0271      0.116     -0.233      0.816      -0.255       0.200
# IXIC_DAILY_RETURNS      0.4186      0.083      5.014      0.000       0.255       0.582
# HSI_DAILY_RETURNS      -0.1148      0.050     -2.277      0.023      -0.214      -0.016
# N225_DAILY_RETURNS     -0.1566      0.064     -2.440      0.015      -0.282      -0.031
# GDAXI_DAILY_RETURNS     0.0252      0.069      0.363      0.717      -0.111       0.161
# VIX_DAILY_RETURNS      -0.0342      0.013     -2.676      0.007      -0.059      -0.009
# NSEI_HL_RATIO         -21.4490     10.676     -2.009      0.045     -42.373      -0.525
# DJI_HL_RATIO          -22.4774     12.709     -1.769      0.077     -47.388       2.433
# NSEI_RSI               -0.0339      0.019     -1.793      0.073      -0.071       0.003
# DJI_RSI                 0.0858      0.020      4.309      0.000       0.047       0.125
# NSEI_ROC               -0.0763      0.043     -1.765      0.078      -0.161       0.008
# DJI_ROC                 0.0298      0.042      0.715      0.475      -0.052       0.112
# NSEI_AWE                0.0004      0.001      0.549      0.583      -0.001       0.002
# DJI_AWE                -0.0003      0.001     -0.486      0.627      -0.001       0.001
# NSEI_KAM             7.615e-05      0.001      0.112      0.911      -0.001       0.001
# DJI_KAM                 0.0003      0.000      0.972      0.331      -0.000       0.001
# NSEI_TSI                0.0274      0.013      2.190      0.029       0.003       0.052
# DJI_TSI                -0.0556      0.014     -3.852      0.000      -0.084      -0.027
# NSEI_VPT            -2.907e-06    2.7e-06     -1.077      0.282    -8.2e-06    2.38e-06
# DJI_VPT             -2.313e-09   2.75e-09     -0.840      0.401   -7.71e-09    3.09e-09
# NSEI_ULC                0.0104      0.078      0.133      0.894      -0.143       0.163
# DJI_ULC                -0.0452      0.080     -0.564      0.573      -0.202       0.112
# NSEI_SMA               -0.0039      0.002     -2.067      0.039      -0.008      -0.000
# DJI_SMA                 0.0024      0.001      2.724      0.006       0.001       0.004
# NSEI_EMA                0.0037      0.002      1.905      0.057      -0.000       0.008
# DJI_EMA                -0.0026      0.001     -2.782      0.005      -0.004      -0.001
# =======================================================================================
# """



# list of insignificant variables
# 
# NSEI_DAILY_RETURNS     -0.1382      0.087     -1.594      0.111      -0.308       0.032
# DJI_DAILY_RETURNS      -0.0271      0.116     -0.233      0.816      -0.255       0.200
# GDAXI_DAILY_RETURNS     0.0252      0.069      0.363      0.717      -0.111       0.161
# DJI_HL_RATIO          -22.4774     12.709     -1.769      0.077     -47.388       2.433
# NSEI_RSI               -0.0339      0.019     -1.793      0.073      -0.071       0.003
# NSEI_ROC               -0.0763      0.043     -1.765      0.078      -0.161       0.008
# DJI_ROC                 0.0298      0.042      0.715      0.475      -0.052       0.112
# NSEI_AWE                0.0004      0.001      0.549      0.583      -0.001       0.002
# DJI_AWE                -0.0003      0.001     -0.486      0.627      -0.001       0.001
# NSEI_KAM             7.615e-05      0.001      0.112      0.911      -0.001       0.001
# DJI_KAM                 0.0003      0.000      0.972      0.331      -0.000       0.001
# NSEI_VPT            -2.907e-06    2.7e-06     -1.077      0.282    -8.2e-06    2.38e-06
# DJI_VPT             -2.313e-09   2.75e-09     -0.840      0.401   -7.71e-09    3.09e-09
# NSEI_ULC                0.0104      0.078      0.133      0.894      -0.143       0.163
# DJI_ULC                -0.0452      0.080     -0.564      0.573      -0.202       0.112
# NSEI_EMA                0.0037      0.002      1.905      0.057      -0.000       0.008




# %% 5 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                 Feature           VIF
# 0    NSEI_DAILY_RETURNS      2.054876
# 1     DJI_DAILY_RETURNS      4.695492
# 2    IXIC_DAILY_RETURNS      4.033191
# 3     HSI_DAILY_RETURNS      1.376423
# 4    N225_DAILY_RETURNS      1.435879
# 5   GDAXI_DAILY_RETURNS      1.912839
# 6     VIX_DAILY_RETURNS      2.375511
# 7         NSEI_HL_RATIO      2.452368
# 8          DJI_HL_RATIO      3.008081
# 9              NSEI_RSI     15.093909
# 10              DJI_RSI     12.430268
# 11             NSEI_ROC      7.646693
# 12              DJI_ROC      6.981107
# 13             NSEI_AWE     31.548467
# 14              DJI_AWE     33.399217
# 15             NSEI_KAM   1424.310763
# 16              DJI_KAM    384.514456
# 17             NSEI_TSI     21.138834
# 18              DJI_TSI     18.274111
# 19             NSEI_VPT     19.319901
# 20              DJI_VPT      2.993374
# 21             NSEI_ULC     10.145682
# 22              DJI_ULC     11.565404
# 23             NSEI_SMA  10639.108241
# 24              DJI_SMA   3240.412536
# 25             NSEI_EMA  11260.305603
# 26              DJI_EMA   3564.987671



# list of colinear variables
# 
# 9              NSEI_RSI     15.093909
# 10              DJI_RSI     12.430268
# 11             NSEI_ROC      7.646693
# 12              DJI_ROC      6.981107
# 13             NSEI_AWE     31.548467
# 14              DJI_AWE     33.399217
# 15             NSEI_KAM   1424.310763
# 16              DJI_KAM    384.514456
# 17             NSEI_TSI     21.138834
# 18              DJI_TSI     18.274111
# 19             NSEI_VPT     19.319901
# 21             NSEI_ULC     10.145682
# 22              DJI_ULC     11.565404
# 23             NSEI_SMA  10639.108241
# 24              DJI_SMA   3240.412536
# 25             NSEI_EMA  11260.305603
# 26              DJI_EMA   3564.987671



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
# Best Threshold is : 0.689



# %% 9 - AUC Curve
auc_roc = roc_auc_score(y, y_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7692547860315854



# %% 10 - Classification Report
y_pred_class = np.where(y_pred <= optimal_threshold,  0, 1)
print(classification_report(y, y_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.54      0.69      0.61       488
#          1.0       0.83      0.72      0.77      1037
# 
#     accuracy                           0.71      1525
#    macro avg       0.69      0.71      0.69      1525
# weighted avg       0.74      0.71      0.72      1525



# %% 11 - 
table = pd.crosstab(y_pred_class, y)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0              339  287
# 1              149  750



# %% 12 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.689 is : 72.32%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.689 is : 69.47%
