
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
from cowboysmall.feature import COLUMNS
from cowboysmall.feature.indicators import get_indicators, get_ratios, INDICATORS, RATIOS
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
model = Logit(y, X).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1525
# Model:                          Logit   Df Residuals:                     1511
# Method:                           MLE   Df Model:                           13
# Date:                Mon, 10 Jun 2024   Pseudo R-squ.:                  0.1525
# Time:                        22:46:12   Log-Likelihood:                -810.21
# converged:                       True   LL-Null:                       -955.98
# Covariance Type:            nonrobust   LLR p-value:                 1.423e-54
# =======================================================================================
#                           coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# Intercept              22.6252     11.276      2.006      0.045       0.525      44.726
# NSEI_DAILY_RETURNS     -0.2002      0.080     -2.510      0.012      -0.357      -0.044
# DJI_DAILY_RETURNS      -0.0036      0.115     -0.031      0.975      -0.229       0.222
# IXIC_DAILY_RETURNS      0.4305      0.083      5.213      0.000       0.269       0.592
# HSI_DAILY_RETURNS      -0.1079      0.050     -2.161      0.031      -0.206      -0.010
# N225_DAILY_RETURNS     -0.1399      0.063     -2.204      0.028      -0.264      -0.015
# GDAXI_DAILY_RETURNS     0.0421      0.069      0.615      0.539      -0.092       0.176
# VIX_DAILY_RETURNS      -0.0391      0.012     -3.177      0.001      -0.063      -0.015
# NSEI_HL_RATIO          -6.2392      9.670     -0.645      0.519     -25.192      12.713
# DJI_HL_RATIO          -16.7583     10.910     -1.536      0.125     -38.142       4.626
# NSEI_RSI               -0.0211      0.013     -1.624      0.104      -0.047       0.004
# DJI_RSI                 0.0510      0.014      3.679      0.000       0.024       0.078
# NSEI_TSI                0.0163      0.007      2.269      0.023       0.002       0.030
# DJI_TSI                -0.0320      0.008     -3.957      0.000      -0.048      -0.016
# =======================================================================================
# """



# list of insignificant variables
# 
# DJI_DAILY_RETURNS      -0.0036      0.115     -0.031      0.975      -0.229       0.222
# GDAXI_DAILY_RETURNS     0.0421      0.069      0.615      0.539      -0.092       0.176
# NSEI_HL_RATIO          -6.2392      9.670     -0.645      0.519     -25.192      12.713
# DJI_HL_RATIO          -16.7583     10.910     -1.536      0.125     -38.142       4.626
# NSEI_RSI               -0.0211      0.013     -1.624      0.104      -0.047       0.004





# %% 5 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                 Feature       VIF
# 0    NSEI_DAILY_RETURNS  1.754727
# 1     DJI_DAILY_RETURNS  4.485704
# 2    IXIC_DAILY_RETURNS  4.005105
# 3     HSI_DAILY_RETURNS  1.354715
# 4    N225_DAILY_RETURNS  1.388446
# 5   GDAXI_DAILY_RETURNS  1.861360
# 6     VIX_DAILY_RETURNS  2.229051
# 7         NSEI_HL_RATIO  1.717471
# 8          DJI_HL_RATIO  2.121525
# 9              NSEI_RSI  7.587992
# 10              DJI_RSI  6.337163
# 11             NSEI_TSI  7.040845
# 12              DJI_TSI  5.864600



# list of colinear variables
# 
# 9              NSEI_RSI  7.587992
# 10              DJI_RSI  6.337163
# 11             NSEI_TSI  7.040845
# 12              DJI_TSI  5.864600



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
# Best Threshold is : 0.641



# %% 9 - AUC Curve
auc_roc = roc_auc_score(y, y_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7637929399117884



# %% 10 - Classification Report
y_pred_class = np.where(y_pred <= optimal_threshold,  0, 1)
print(classification_report(y, y_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.61      0.59      0.60       488
#          1.0       0.81      0.82      0.81      1037
# 
#     accuracy                           0.75      1525
#    macro avg       0.71      0.70      0.71      1525
# weighted avg       0.74      0.75      0.74      1525



# %% 11 - 
table = pd.crosstab(y_pred_class, y)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0              288  188
# 1              200  849



# %% 12 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)

print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Sensitivity for cut-off 0.641 is : 81.87%
# Specificity for cut-off 0.641 is : 59.02%
