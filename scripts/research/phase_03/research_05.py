
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
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

from gsma import COLUMNS
from gsma.data.file import read_master_file
from gsma.feature.indicators import calculate_rsi
from gsma.plots import plt, sns



# %% 2 -
master = read_master_file()

master["NSEI_OPEN_DIR"] = np.where(master["NSEI_OPEN"] > master["NSEI_CLOSE"].shift(), 1, 0)

master["NSEI_HL_RATIO"] = master["NSEI_HIGH"] / master["NSEI_LOW"]
master["DJI_HL_RATIO"]  = master["DJI_HIGH"] / master["DJI_LOW"]

master["NSEI_RSI"]      = calculate_rsi(master["NSEI_CLOSE"])
master["DJI_RSI"]       = calculate_rsi(master["DJI_CLOSE"])

EXTRA_COLS = ["NSEI_HL_RATIO", "DJI_HL_RATIO", "NSEI_RSI", "DJI_RSI"]



# %% 2 -
# master[["NSEI_OPEN_DIR"]].to_csv("./data/NSEI_OPEN_DIR_02.csv")

counts = master['NSEI_OPEN_DIR'].value_counts().reset_index()
counts.columns = ['NSEI_OPEN_DIR', 'Freq']
print(counts)
#    NSEI_OPEN_DIR  Freq
# 0              1  1064
# 1              0   499



# %% 3 -
data = pd.concat([master["NSEI_OPEN_DIR"].shift(-1), master[COLUMNS + EXTRA_COLS]], axis = 1)
data.dropna(inplace = True)
data.head()



# %% 4 -
train, test = train_test_split(data, test_size = 0.2)



# %% 5 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + VIX_DAILY_RETURNS + NSEI_HL_RATIO + DJI_HL_RATIO + DJI_RSI', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1238
# Model:                          Logit   Df Residuals:                     1230
# Method:                           MLE   Df Model:                            7
# Date:                Sat, 20 Apr 2024   Pseudo R-squ.:                  0.1661
# Time:                        15:15:08   Log-Likelihood:                -652.59
# converged:                       True   LL-Null:                       -782.58
# Covariance Type:            nonrobust   LLR p-value:                 2.074e-52
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             19.8863     12.030      1.653      0.098      -3.692      43.465
# NSEI_DAILY_RETURNS    -0.2199      0.074     -2.966      0.003      -0.365      -0.075
# IXIC_DAILY_RETURNS     0.4662      0.075      6.231      0.000       0.320       0.613
# HSI_DAILY_RETURNS     -0.1111      0.052     -2.117      0.034      -0.214      -0.008
# VIX_DAILY_RETURNS     -0.0646      0.013     -4.799      0.000      -0.091      -0.038
# NSEI_HL_RATIO          6.7555     10.551      0.640      0.522     -13.924      27.435
# DJI_HL_RATIO         -25.7212     11.652     -2.208      0.027     -48.558      -2.884
# DJI_RSI                0.0029      0.005      0.650      0.516      -0.006       0.012
# ======================================================================================
# """



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.218317
# 1  IXIC_DAILY_RETURNS  1.866691
# 2   HSI_DAILY_RETURNS  1.205521
# 3   VIX_DAILY_RETURNS  1.887012
# 4       NSEI_HL_RATIO  1.488511
# 5        DJI_HL_RATIO  1.882807
# 6             DJI_RSI  1.272838



# %% 8 - ROC Curve
train['predicted'] = model.predict(train)

fpr, tpr, thresholds = roc_curve(train['NSEI_OPEN_DIR'], train['predicted'])

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "04_01", "01 - training data", "phase_03")



# %% 9 - Optimal Threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold is : {optimal_threshold}')
# Best Threshold is : 0.63



# %% 10 - AUC Curve
auc_roc = roc_auc_score(train['NSEI_OPEN_DIR'], train['predicted'])
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7625124123723565



# %% 11 - Classification Report
train['predicted_class'] = np.where(train['predicted'] <= optimal_threshold,  0, 1)
print(classification_report(train['NSEI_OPEN_DIR'], train['predicted_class']))
#               precision    recall  f1-score   support
# 
#            0       0.64      0.58      0.61       405
#            1       0.81      0.84      0.82       833
# 
#     accuracy                           0.76      1238
#    macro avg       0.72      0.71      0.72      1238
# weighted avg       0.75      0.76      0.75      1238



# %% 11 - 
table = pd.crosstab(train['predicted_class'], train['NSEI_OPEN_DIR'])
table
# NSEI_OPEN_DIR      0    1
# predicted_class          
# 0                236  134
# 1                169  699



# %% 11 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.63 is : 83.91%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.63 is : 58.27%



# %% 12 - ROC Curve
test['predicted'] = model.predict(test)

fpr, tpr, thresholds = roc_curve(test['NSEI_OPEN_DIR'], test['predicted'])

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "04_02", "02 - testing data", "phase_03")



# %% 13 - AUC Curve
auc_roc = roc_auc_score(test['NSEI_OPEN_DIR'], test['predicted'])
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7280115676106901



# %% 14 - Classification Report
test['predicted_class'] = np.where(test['predicted'] <= optimal_threshold,  0, 1)
print(classification_report(test['NSEI_OPEN_DIR'], test['predicted_class']))
#               precision    recall  f1-score   support
# 
#            0       0.56      0.51      0.53        92
#            1       0.80      0.83      0.82       218
# 
#     accuracy                           0.74       310
#    macro avg       0.68      0.67      0.67       310
# weighted avg       0.73      0.74      0.73       310



# %% 11 - 
table = pd.crosstab(test['predicted_class'], test['NSEI_OPEN_DIR'])
table
# NSEI_OPEN_DIR     0    1
# predicted_class         
# 0                47   37
# 1                45  181



# %% 11 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.63 is : 83.03%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.63 is : 51.09%
