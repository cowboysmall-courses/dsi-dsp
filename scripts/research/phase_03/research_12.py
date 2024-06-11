
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

from imblearn.over_sampling import ADASYN

from cowboysmall.data.file import read_master_file
from cowboysmall.model.logit import prune
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


# %% 3 -
remedy = ADASYN(random_state = 0)
X, y = remedy.fit_resample(X, y)


# %% 4 -
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337)



# %% 5 -
model, dropped = prune(X_train, y_train)
# dropping GDAXI_DAILY_RETURNS with p-value 0.7301496300423773
# dropping NSEI_RSI with p-value 0.5754154150385363
# dropping DJI_DAILY_RETURNS with p-value 0.2834143257149382
# dropping NSEI_HL_RATIO with p-value 0.15746260405792234
# dropping DJI_RSI with vif 5.433087536046928
# dropping N225_DAILY_RETURNS with p-value 0.12689146376306823



# %% 5 -
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1616
# Model:                          Logit   Df Residuals:                     1608
# Method:                           MLE   Df Model:                            7
# Date:                Mon, 10 Jun 2024   Pseudo R-squ.:                  0.1281
# Time:                        23:34:26   Log-Likelihood:                -975.86
# converged:                       True   LL-Null:                       -1119.2
# Covariance Type:            nonrobust   LLR p-value:                 4.119e-58
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             26.7431      9.354      2.859      0.004       8.410      45.077
# NSEI_DAILY_RETURNS    -0.2940      0.064     -4.621      0.000      -0.419      -0.169
# IXIC_DAILY_RETURNS     0.5120      0.063      8.169      0.000       0.389       0.635
# HSI_DAILY_RETURNS     -0.1171      0.044     -2.693      0.007      -0.202      -0.032
# VIX_DAILY_RETURNS     -0.0409      0.011     -3.611      0.000      -0.063      -0.019
# DJI_HL_RATIO         -26.3016      9.234     -2.848      0.004     -44.399      -8.204
# NSEI_TSI               0.0055      0.003      1.983      0.047    6.35e-05       0.011
# DJI_TSI               -0.0077      0.004     -2.138      0.033      -0.015      -0.001
# ======================================================================================
# """



# %% 10 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.188922
# 1  IXIC_DAILY_RETURNS  2.015730
# 2   HSI_DAILY_RETURNS  1.182919
# 3   VIX_DAILY_RETURNS  2.029126
# 4        DJI_HL_RATIO  1.459482
# 5            NSEI_TSI  1.427790
# 6             DJI_TSI  1.563099



# %% 8 - ROC Curve
y_pred = model.predict(X_train.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y_train, y_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "08_01", "01 - training data", "phase_03")



# %% 9 - Optimal Threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold: {optimal_threshold}')
# Best Threshold: 0.532



# %% 10 - AUC Curve
auc_roc = roc_auc_score(y_train, y_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7467426223097979



# %% 11 - Classification Report
y_pred_class = np.where(y_pred <= optimal_threshold,  0, 1)
print(classification_report(y_train, y_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.69      0.67      0.68       781
#          1.0       0.70      0.71      0.70       835
# 
#     accuracy                           0.69      1616
#    macro avg       0.69      0.69      0.69      1616
# weighted avg       0.69      0.69      0.69      1616



# %% 11 - Confusion Matrix
table = pd.crosstab(y_pred_class, y_train)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0              521  239
# 1              260  596



# %% 11 - Sensitivity / Specificity
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)

print(f"Sensitivity: {sensitivity}%")
print(f"Specificity: {specificity}%")
# Sensitivity: 71.38%
# Specificity: 66.71%



# %% 12 - ROC Curve
y_test_pred = model.predict(X_test.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "08_02", "02 - test data", "phase_03")



# %% 13 - AUC Curve
auc_roc = roc_auc_score(y_test, y_test_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7515973272204068



# %% 14 - Classification Report
y_test_pred_class = np.where(y_test_pred <= optimal_threshold,  0, 1)
print(classification_report(y_test, y_test_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.67      0.69      0.68       203
#          1.0       0.68      0.66      0.67       202
# 
#     accuracy                           0.68       405
#    macro avg       0.68      0.68      0.68       405
# weighted avg       0.68      0.68      0.68       405



# %% 11 - Confusion Matrix
table = pd.crosstab(y_test_pred_class, y_test)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0              141   69
# 1               62  133



# %% 11 - Sensitivity / Specificity
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)

print(f"Sensitivity: {sensitivity}%")
print(f"Specificity: {specificity}%")
# Sensitivity: 65.84%
# Specificity: 69.46%
