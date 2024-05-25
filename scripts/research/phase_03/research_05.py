
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



# %% 4 -
train, test = train_test_split(data, test_size = 0.2, random_state = 1337)



# %% 5 -
model = logit('NSEI_OPEN_DIR ~ IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + N225_DAILY_RETURNS + VIX_DAILY_RETURNS + DJI_RSI', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1214
# Method:                           MLE   Df Model:                            5
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1328
# Time:                        17:10:31   Log-Likelihood:                -663.59
# converged:                       True   LL-Null:                       -765.23
# Covariance Type:            nonrobust   LLR p-value:                 5.605e-42
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept              0.0858      0.345      0.249      0.804      -0.591       0.762
# IXIC_DAILY_RETURNS     0.4639      0.074      6.253      0.000       0.318       0.609
# HSI_DAILY_RETURNS     -0.1297      0.053     -2.470      0.013      -0.233      -0.027
# N225_DAILY_RETURNS    -0.1533      0.066     -2.337      0.019      -0.282      -0.025
# VIX_DAILY_RETURNS     -0.0470      0.013     -3.693      0.000      -0.072      -0.022
# DJI_RSI                0.0142      0.006      2.251      0.024       0.002       0.026
# ======================================================================================
# """



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  IXIC_DAILY_RETURNS  2.054828
# 1   HSI_DAILY_RETURNS  1.242647
# 2  N225_DAILY_RETURNS  1.287734
# 3   VIX_DAILY_RETURNS  1.917594
# 4             DJI_RSI  1.117161



# %% 8 - ROC Curve
train['predicted'] = model.predict(train)

fpr, tpr, thresholds = roc_curve(train['NSEI_OPEN_DIR'], train['predicted'])

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "05_01", "01 - training data", "phase_03")



# %% 9 - Optimal Threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold is : {optimal_threshold}')
# Best Threshold is : 0.686



# %% 10 - AUC Curve
auc_roc = roc_auc_score(train['NSEI_OPEN_DIR'], train['predicted'])
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7505884821018144



# %% 11 - Classification Report
train['predicted_class'] = np.where(train['predicted'] <= optimal_threshold,  0, 1)
print(classification_report(train['NSEI_OPEN_DIR'], train['predicted_class']))
#               precision    recall  f1-score   support
# 
#          0.0       0.54      0.67      0.60       391
#          1.0       0.82      0.73      0.78       829
# 
#     accuracy                           0.71      1220
#    macro avg       0.68      0.70      0.69      1220
# weighted avg       0.73      0.71      0.72      1220



# %% 11 - 
table = pd.crosstab(train['predicted_class'], train['NSEI_OPEN_DIR'])
table
# NSEI_OPEN_DIR    0.0  1.0
# predicted_class          
# 0                261  222
# 1                130  607



# %% 11 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.63 is : 73.22%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.63 is : 66.75%



# %% 12 - ROC Curve
test['predicted'] = model.predict(test)

fpr, tpr, thresholds = roc_curve(test['NSEI_OPEN_DIR'], test['predicted'])

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "05_02", "02 - testing data", "phase_03")



# %% 13 - AUC Curve
auc_roc = roc_auc_score(test['NSEI_OPEN_DIR'], test['predicted'])
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.749603489294211



# %% 14 - Classification Report
test['predicted_class'] = np.where(test['predicted'] <= optimal_threshold,  0, 1)
print(classification_report(test['NSEI_OPEN_DIR'], test['predicted_class']))
#               precision    recall  f1-score   support

#          0.0       0.52      0.67      0.59        97
#          1.0       0.82      0.72      0.77       208

#     accuracy                           0.70       305
#    macro avg       0.67      0.69      0.68       305
# weighted avg       0.73      0.70      0.71       305



# %% 11 - 
table = pd.crosstab(test['predicted_class'], test['NSEI_OPEN_DIR'])
table
# NSEI_OPEN_DIR    0.0  1.0
# predicted_class          
# 0                 65   59
# 1                 32  149



# %% 11 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.63 is : 71.63%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.63 is : 67.01%
