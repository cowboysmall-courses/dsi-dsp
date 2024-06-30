
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
from cowboysmall.model.logit import prune
from cowboysmall.feature import COLUMNS, INDICATORS, RATIOS
from cowboysmall.feature.indicators import get_indicators, get_ratios
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
model, dropped = prune(X, y)
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1525
# Model:                          Logit   Df Residuals:                     1519
# Method:                           MLE   Df Model:                            5
# Date:                Mon, 10 Jun 2024   Pseudo R-squ.:                  0.1408
# Time:                        23:13:17   Log-Likelihood:                -821.37
# converged:                       True   LL-Null:                       -955.98
# Covariance Type:            nonrobust   LLR p-value:                 4.148e-56
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             23.2887      8.548      2.724      0.006       6.534      40.043
# NSEI_DAILY_RETURNS    -0.2438      0.067     -3.666      0.000      -0.374      -0.113
# IXIC_DAILY_RETURNS     0.4642      0.066      7.049      0.000       0.335       0.593
# HSI_DAILY_RETURNS     -0.1157      0.046     -2.500      0.012      -0.206      -0.025
# VIX_DAILY_RETURNS     -0.0494      0.012     -4.276      0.000      -0.072      -0.027
# DJI_HL_RATIO         -22.1657      8.450     -2.623      0.009     -38.727      -5.605
# ======================================================================================
# """



# %% 5 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.215929
# 1  IXIC_DAILY_RETURNS  2.032330
# 2   HSI_DAILY_RETURNS  1.189606
# 3   VIX_DAILY_RETURNS  2.022404
# 4        DJI_HL_RATIO  1.059471



# %% 6 - ROC Curve
y_pred_prob = model.predict(X.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y, y_pred_prob)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "Logistic Model")



# %% 7 - find optimal threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold is : {optimal_threshold}')
# Best Threshold is : 0.625



# %% 8 - AUC Curve
auc_roc = roc_auc_score(y, y_pred_prob)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7558896643849693



# %% 10 - Classification Report
y_pred_class = np.where(y_pred_prob <= optimal_threshold,  0, 1)
print(classification_report(y, y_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.64      0.55      0.59       488
#          1.0       0.80      0.85      0.83      1037
# 
#     accuracy                           0.76      1525
#    macro avg       0.72      0.70      0.71      1525
# weighted avg       0.75      0.76      0.75      1525



# %% 11 - 
table = pd.crosstab(y_pred_class, y)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0              267  151
# 1              221  886



# %% 12 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)

print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Sensitivity for cut-off 0.625 is : 85.44%
# Specificity for cut-off 0.625 is : 54.71%
