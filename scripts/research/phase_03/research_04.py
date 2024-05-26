
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
X.insert(loc = 0, column = "Intercept", value = 1)



# %% 4 -
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337)



# %% 5 -
model, dropped = pruned_logit(X_train, y_train)
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1220
# Model:                          Logit   Df Residuals:                     1213
# Method:                           MLE   Df Model:                            6
# Date:                Sun, 26 May 2024   Pseudo R-squ.:                  0.1375
# Time:                        13:13:53   Log-Likelihood:                -660.02
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



# %% 10 -
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
y_pred = model.predict(X_train.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y_train, y_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "04_01", "01 - training data", "phase_03")



# %% 9 - Optimal Threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold is : {optimal_threshold}')
# Best Threshold is : 0.684



# %% 10 - AUC Curve
auc_roc = roc_auc_score(y_train, y_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7529115595469844



# %% 11 - Classification Report
y_pred_class = np.where(y_pred <= optimal_threshold,  0, 1)
print(classification_report(y_train, y_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.53      0.68      0.60       391
#          1.0       0.83      0.72      0.77       829
# 
#     accuracy                           0.70      1220
#    macro avg       0.68      0.70      0.68      1220
# weighted avg       0.73      0.70      0.71      1220



# %% 11 - Confusion Matrix
table = pd.crosstab(y_pred_class, y_train)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0              265  234
# 1              126  595



# %% 11 - Sensitivity / Specificity
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.684 is : 71.77%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.684 is : 67.77%



# %% 12 - ROC Curve
y_test_pred = model.predict(X_test.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "04_02", "02 - test data", "phase_03")



# %% 13 - AUC Curve
auc_roc = roc_auc_score(y_test, y_test_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7520816812053925



# %% 14 - Classification Report
y_test_pred_class = np.where(y_test_pred <= optimal_threshold,  0, 1)
print(classification_report(y_test, y_test_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.53      0.65      0.58        97
#          1.0       0.82      0.73      0.77       208
# 
#     accuracy                           0.70       305
#    macro avg       0.67      0.69      0.67       305
# weighted avg       0.72      0.70      0.71       305



# %% 11 - Confusion Matrix
table = pd.crosstab(y_test_pred_class, y_test)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0               63   57
# 1               34  151



# %% 11 - Sensitivity / Specificity
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.684 is : 72.6%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.684 is : 64.95%
