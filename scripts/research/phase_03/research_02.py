
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

from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

from gsma import COLUMNS
from gsma.data.file import read_master_file
from gsma.model.logit import pruned_logit
from gsma.feature.indicators import calculate_rsi
from gsma.plots import plt, sns



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

X = data.drop(["NSEI_OPEN_DIR"], axis = 1)
y = data['NSEI_OPEN_DIR']



# %% 4 -
model, dropped = pruned_logit(X, y)
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1548
# Model:                          Logit   Df Residuals:                     1543
# Method:                           MLE   Df Model:                            4
# Date:                Sat, 20 Apr 2024   Pseudo R-squ.:                  0.1395
# Time:                        12:23:18   Log-Likelihood:                -836.09
# converged:                       True   LL-Null:                       -971.63
# Covariance Type:            nonrobust   LLR p-value:                 1.881e-57
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# NSEI_DAILY_RETURNS    -0.2404      0.065     -3.674      0.000      -0.369      -0.112
# IXIC_DAILY_RETURNS     0.4921      0.066      7.473      0.000       0.363       0.621
# HSI_DAILY_RETURNS     -0.1136      0.046     -2.459      0.014      -0.204      -0.023
# VIX_DAILY_RETURNS     -0.0517      0.011     -4.567      0.000      -0.074      -0.030
# NSEI_HL_RATIO          0.8570      0.061     14.076      0.000       0.738       0.976
# ======================================================================================
# """



# %% 5 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.198188
# 1  IXIC_DAILY_RETURNS  1.925040
# 2   HSI_DAILY_RETURNS  1.188433
# 3   VIX_DAILY_RETURNS  1.883128
# 4       NSEI_HL_RATIO  1.012427



# %% 6 - ROC Curve
X['predicted'] = model.predict(X.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y, X['predicted'])

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "01_01", "01 - with all data", "phase_03")



# %% 7 - find optimal threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold is : {optimal_threshold}')
# Best Threshold is : 0.681



# %% 8 - AUC Curve
auc_roc = roc_auc_score(y, X['predicted'])
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7546257564415991



# %% 10 - Classification Report
X['predicted_class'] = np.where(X['predicted'] <= optimal_threshold,  0, 1)
print(classification_report(y, X['predicted_class']))
#               precision    recall  f1-score   support
# 
#            0       0.53      0.69      0.60       497
#            1       0.83      0.71      0.77      1051
# 
#     accuracy                           0.71      1548
#    macro avg       0.68      0.70      0.68      1548
# weighted avg       0.73      0.71      0.71      1548



# %% 11 - 
table = pd.crosstab(X['predicted_class'], y)
table
# NSEI_OPEN_DIR      0    1
# predicted_class          
# 0                342  300
# 1                155  751



# %% 12 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.681 is : 71.46%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.681 is : 68.81%
