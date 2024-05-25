
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
from sklearn.model_selection import train_test_split

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
X.insert(loc = 0, column = "Intercept", value = pd.Series([1] * X.shape[0]))



# %% 4 -
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337)



# %% 5 -
model, dropped = pruned_logit(X_train, y_train)
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1215
# Method:                           MLE   Df Model:                            4
# Date:                Sat, 25 May 2024   Pseudo R-squ.:                  0.1262
# Time:                        16:44:35   Log-Likelihood:                -668.64
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



# %% 10 -
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
X_train['predicted'] = model.predict(X_train.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y_train, X_train['predicted'])

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "04_01", "01 - training data", "phase_03")



# %% 9 - Optimal Threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold is : {optimal_threshold}')
# Best Threshold is : 0.649



# %% 10 - AUC Curve
auc_roc = roc_auc_score(y_train, X_train['predicted'])
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.748043277729616



# %% 11 - Classification Report
X_train['predicted_class'] = np.where(X_train['predicted'] <= optimal_threshold,  0, 1)
print(classification_report(y_train, X_train['predicted_class']))
#               precision    recall  f1-score   support
# 
#          0.0       0.57      0.61      0.59       391
#          1.0       0.81      0.78      0.80       829
# 
#     accuracy                           0.73      1220
#    macro avg       0.69      0.70      0.69      1220
# weighted avg       0.73      0.73      0.73      1220



# %% 11 - Confusion Matrix
table = pd.crosstab(X_train['predicted_class'], y_train)
table
# NSEI_OPEN_DIR    0.0  1.0
# predicted_class          
# 0                237  179
# 1                154  650



# %% 11 - Sensitivity / Specificity
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.686 is : 78.41%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.686 is : 60.61%



# %% 12 - ROC Curve
X_test['predicted'] = model.predict(X_test.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y_test, X_test['predicted'])

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "04_02", "02 - test data", "phase_03")



# %% 13 - AUC Curve
auc_roc = roc_auc_score(y_test, X_test['predicted'])
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7566415543219667



# %% 14 - Classification Report
X_test['predicted_class'] = np.where(X_test['predicted'] <= optimal_threshold,  0, 1)
print(classification_report(y_test, X_test['predicted_class']))
#               precision    recall  f1-score   support
# 
#          0.0       0.60      0.61      0.60        97
#          1.0       0.82      0.81      0.81       208
# 
#     accuracy                           0.74       305
#    macro avg       0.71      0.71      0.71       305
# weighted avg       0.75      0.74      0.74       305



# %% 11 - Confusion Matrix
table = pd.crosstab(X_test['predicted_class'], y_test)
table
# NSEI_OPEN_DIR    0.0  1.0
# predicted_class          
# 0                 59   40
# 1                 38  168



# %% 11 - Sensitivity / Specificity
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.686 is : 80.77%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.686 is : 60.82%
