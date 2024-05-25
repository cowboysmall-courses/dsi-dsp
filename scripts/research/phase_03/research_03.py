
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

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import classification_report, roc_curve, roc_auc_score

from cowboysmall.data.file import read_master_file
from cowboysmall.model.logit import pruned_logit
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
model, dropped = pruned_logit(X, y)
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1525
# Model:                          Logit   Df Residuals:                     1519
# Method:                           MLE   Df Model:                            5
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1279
# Time:                        16:37:31   Log-Likelihood:                -833.75
# converged:                       True   LL-Null:                       -955.98
# Covariance Type:            nonrobust   LLR p-value:                 8.530e-51
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# NSEI_DAILY_RETURNS    -0.2227      0.065     -3.405      0.001      -0.351      -0.094
# IXIC_DAILY_RETURNS     0.4907      0.066      7.389      0.000       0.361       0.621
# HSI_DAILY_RETURNS     -0.1206      0.046     -2.598      0.009      -0.212      -0.030
# VIX_DAILY_RETURNS     -0.0498      0.012     -4.300      0.000      -0.073      -0.027
# NSEI_TSI               0.0057      0.003      2.076      0.038       0.000       0.011
# NSEI_SMA            5.135e-05   4.66e-06     11.009      0.000    4.22e-05    6.05e-05
# ======================================================================================
# """



# %% 5 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.209435
# 1  IXIC_DAILY_RETURNS  2.033175
# 2   HSI_DAILY_RETURNS  1.190048
# 3   VIX_DAILY_RETURNS  1.994609
# 4            NSEI_TSI  1.229180
# 5            NSEI_SMA  1.222505



# %% 6 - ROC Curve
y_pred = model.predict(X.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y, y_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "03_01", "01 - with all data", "phase_03")



# %% 7 - find optimal threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold is : {optimal_threshold}')
# Best Threshold is : 0.673



# %% 8 - AUC Curve
auc_roc = roc_auc_score(y, y_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7475635107577027



# %% 10 - Classification Report
y_pred_class = np.where(y_pred <= optimal_threshold,  0, 1)
print(classification_report(y, y_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.52      0.67      0.59       488
#          1.0       0.82      0.71      0.76      1037
# 
#     accuracy                           0.70      1525
#    macro avg       0.67      0.69      0.68      1525
# weighted avg       0.73      0.70      0.71      1525



# %% 11 - 
table = pd.crosstab(y_pred_class, y)
table
# NSEI_OPEN_DIR    0.0  1.0
# predicted_class          
# 0                328  299
# 1                160  738



# %% 12 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.681 is : 71.17%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.681 is : 67.21%
