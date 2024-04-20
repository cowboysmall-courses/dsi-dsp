
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

from sklearn.model_selection import train_test_split
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)



# %% 5 -
model, dropped = pruned_logit(X_train, y_train)
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1238
# Model:                          Logit   Df Residuals:                     1233
# Method:                           MLE   Df Model:                            4
# Date:                Sat, 20 Apr 2024   Pseudo R-squ.:                  0.1309
# Time:                        13:55:34   Log-Likelihood:                -669.02
# converged:                       True   LL-Null:                       -769.79
# Covariance Type:            nonrobust   LLR p-value:                 1.749e-42
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# NSEI_DAILY_RETURNS    -0.2050      0.070     -2.917      0.004      -0.343      -0.067
# IXIC_DAILY_RETURNS     0.3818      0.071      5.340      0.000       0.242       0.522
# HSI_DAILY_RETURNS     -0.1150      0.052     -2.232      0.026      -0.216      -0.014
# VIX_DAILY_RETURNS     -0.0615      0.013     -4.740      0.000      -0.087      -0.036
# DJI_RSI                0.0150      0.001     12.545      0.000       0.013       0.017
# ======================================================================================
# """



# %% 10 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.195876
# 1  IXIC_DAILY_RETURNS  2.076824
# 2   HSI_DAILY_RETURNS  1.193871
# 3   VIX_DAILY_RETURNS  2.003012
# 4             DJI_RSI  1.026218



# %% 8 - ROC Curve
X_train['predicted'] = model.predict(X_train.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y_train, X_train['predicted'])

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "02_01", "01 - training data", "phase_03")



# %% 9 - Optimal Threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold is : {optimal_threshold}')
# Best Threshold is : 0.686



# %% 10 - AUC Curve
auc_roc = roc_auc_score(y_train, X_train['predicted'])
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7459808975136446



# %% 11 - Classification Report
X_train['predicted_class'] = np.where(X_train['predicted'] <= optimal_threshold,  0, 1)
print(classification_report(y_train, X_train['predicted_class']))
#               precision    recall  f1-score   support
# 
#            0       0.53      0.66      0.59       388
#            1       0.83      0.73      0.77       850
# 
#     accuracy                           0.71      1238
#    macro avg       0.68      0.70      0.68      1238
# weighted avg       0.73      0.71      0.72      1238



# %% 11 - Confusion Matrix
table = pd.crosstab(X_train['predicted_class'], y_train)
table
# NSEI_OPEN_DIR      0    1
# predicted_class          
# 0                258  232
# 1                130  618



# %% 11 - Sensitivity / Specificity
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.686 is : 72.71%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.686 is : 66.49%



# %% 12 - ROC Curve
X_test['predicted'] = model.predict(X_test.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y_test, X_test['predicted'])

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "02_02", "02 - test data", "phase_03")



# %% 13 - AUC Curve
auc_roc = roc_auc_score(y_test, X_test['predicted'])
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7498744808069743



# %% 14 - Classification Report
X_test['predicted_class'] = np.where(X_test['predicted'] <= optimal_threshold,  0, 1)
print(classification_report(y_test, X_test['predicted_class']))
#               precision    recall  f1-score   support
# 
#            0       0.50      0.66      0.57       109
#            1       0.78      0.64      0.70       201
# 
#     accuracy                           0.65       310
#    macro avg       0.64      0.65      0.63       310
# weighted avg       0.68      0.65      0.65       310



# %% 11 - Confusion Matrix
table = pd.crosstab(X_test['predicted_class'], y_test)
table
# NSEI_OPEN_DIR     0    1
# predicted_class         
# 0                72   73
# 1                37  128



# %% 11 - Sensitivity / Specificity
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.686 is : 63.68%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.686 is : 66.06%
