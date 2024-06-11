
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

from imblearn.over_sampling import SMOTE

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
remedy = SMOTE(random_state = 0)
X, y = remedy.fit_resample(X, y)


# %% 4 -
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337)



# %% 5 -
model, dropped = prune(X_train, y_train)
# dropping GDAXI_DAILY_RETURNS with p-value 0.9598750548300169
# dropping DJI_DAILY_RETURNS with p-value 0.5822990513868496
# dropping NSEI_RSI with p-value 0.4994225865712918
# dropping NSEI_HL_RATIO with p-value 0.3682870269572075
# dropping DJI_RSI with vif 5.469357531595281
# dropping DJI_TSI with p-value 0.14694393043136192
# dropping N225_DAILY_RETURNS with p-value 0.06921090444968113



# %% 5 -
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1659
# Model:                          Logit   Df Residuals:                     1652
# Method:                           MLE   Df Model:                            6
# Date:                Mon, 10 Jun 2024   Pseudo R-squ.:                  0.1571
# Time:                        23:41:30   Log-Likelihood:                -969.24
# converged:                       True   LL-Null:                       -1149.9
# Covariance Type:            nonrobust   LLR p-value:                 5.531e-75
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             19.8483      8.404      2.362      0.018       3.377      36.320
# NSEI_DAILY_RETURNS    -0.2568      0.064     -4.024      0.000      -0.382      -0.132
# IXIC_DAILY_RETURNS     0.5376      0.064      8.393      0.000       0.412       0.663
# HSI_DAILY_RETURNS     -0.1594      0.044     -3.647      0.000      -0.245      -0.074
# VIX_DAILY_RETURNS     -0.0563      0.012     -4.853      0.000      -0.079      -0.034
# DJI_HL_RATIO         -19.5842      8.299     -2.360      0.018     -35.849      -3.319
# NSEI_TSI               0.0058      0.003      2.176      0.030       0.001       0.011
# ======================================================================================
# """



# %% 10 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.204102
# 1  IXIC_DAILY_RETURNS  2.080181
# 2   HSI_DAILY_RETURNS  1.198179
# 3   VIX_DAILY_RETURNS  2.065043
# 4        DJI_HL_RATIO  1.323426
# 5            NSEI_TSI  1.239002



# %% 8 - ROC Curve
y_pred = model.predict(X_train.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y_train, y_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "09_01", "01 - training data", "phase_03")



# %% 9 - Optimal Threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold: {optimal_threshold}')
# Best Threshold: 0.486



# %% 10 - AUC Curve
auc_roc = roc_auc_score(y_train, y_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7665557283415931



# %% 11 - Classification Report
y_pred_class = np.where(y_pred <= optimal_threshold,  0, 1)
print(classification_report(y_train, y_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.74      0.63      0.68       830
#          1.0       0.68      0.78      0.72       829
# 
#     accuracy                           0.70      1659
#    macro avg       0.71      0.70      0.70      1659
# weighted avg       0.71      0.70      0.70      1659



# %% 11 - Confusion Matrix
table = pd.crosstab(y_pred_class, y_train)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0              520  185
# 1              310  644



# %% 11 - Sensitivity / Specificity
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)

print(f"Sensitivity: {sensitivity}%")
print(f"Specificity: {specificity}%")
# Sensitivity: 77.68%
# Specificity: 62.65%



# %% 12 - ROC Curve
y_test_pred = model.predict(X_test.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "09_02", "02 - test data", "phase_03")



# %% 13 - AUC Curve
auc_roc = roc_auc_score(y_test, y_test_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.724126718691936



# %% 14 - Classification Report
y_test_pred_class = np.where(y_test_pred <= optimal_threshold,  0, 1)
print(classification_report(y_test, y_test_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.68      0.64      0.66       207
#          1.0       0.66      0.70      0.68       208
# 
#     accuracy                           0.67       415
#    macro avg       0.67      0.67      0.67       415
# weighted avg       0.67      0.67      0.67       415



# %% 11 - Confusion Matrix
table = pd.crosstab(y_test_pred_class, y_test)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0              132   63
# 1               75  145



# %% 11 - Sensitivity / Specificity
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)

print(f"Sensitivity: {sensitivity}%")
print(f"Specificity: {specificity}%")
# Sensitivity: 69.71%
# Specificity: 63.77%
