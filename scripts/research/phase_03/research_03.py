
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
# Model:                          Logit   Df Residuals:                     1518
# Method:                           MLE   Df Model:                            6
# Date:                Sun, 26 May 2024   Pseudo R-squ.:                  0.1393
# Time:                        13:09:17   Log-Likelihood:                -822.85
# converged:                       True   LL-Null:                       -955.98
# Covariance Type:            nonrobust   LLR p-value:                 1.369e-54
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             -1.2747      0.587     -2.170      0.030      -2.426      -0.123
# IXIC_DAILY_RETURNS     0.4613      0.068      6.808      0.000       0.328       0.594
# HSI_DAILY_RETURNS     -0.1483      0.048     -3.117      0.002      -0.242      -0.055
# N225_DAILY_RETURNS    -0.1772      0.061     -2.928      0.003      -0.296      -0.059
# VIX_DAILY_RETURNS     -0.0389      0.012     -3.307      0.001      -0.062      -0.016
# DJI_RSI                0.0423      0.012      3.606      0.000       0.019       0.065
# DJI_TSI               -0.0208      0.007     -3.000      0.003      -0.034      -0.007
# ======================================================================================
# """



# %% 5 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  IXIC_DAILY_RETURNS  2.097736
# 1   HSI_DAILY_RETURNS  1.237885
# 2  N225_DAILY_RETURNS  1.319463
# 3   VIX_DAILY_RETURNS  2.082150
# 4             DJI_RSI  4.927840
# 5             DJI_TSI  4.459507



# %% 6 - ROC Curve
y_pred = model.predict(X.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y, y_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "03_01", "01 - with all data", "phase_03")



# %% 7 - find optimal threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold is : {optimal_threshold}')
# Best Threshold is : 0.675



# %% 8 - AUC Curve
auc_roc = roc_auc_score(y, y_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7533296710245507



# %% 10 - Classification Report
y_pred_class = np.where(y_pred <= optimal_threshold,  0, 1)
print(classification_report(y, y_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.54      0.66      0.60       488
#          1.0       0.82      0.74      0.78      1037
# 
#     accuracy                           0.71      1525
#    macro avg       0.68      0.70      0.69      1525
# weighted avg       0.73      0.71      0.72      1525



# %% 11 - 
table = pd.crosstab(y_pred_class, y)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0              322  272
# 1              166  765



# %% 12 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.675 is : 73.77%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.675 is : 65.98%
