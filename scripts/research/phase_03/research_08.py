
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


# %% 3 -
remedy = ADASYN(random_state = 0)
X, y = remedy.fit_resample(X, y)


# %% 4 -
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337)



# %% 5 -
model, dropped = pruned_logit(X_train, y_train)
# dropping DJI_AWE with p-value 0.9510085575443713
# dropping NSEI_KAM with p-value 0.878847553932829
# dropping DJI_ROC with p-value 0.6833807002778988
# dropping DJI_VPT with p-value 0.5859345264104551
# dropping NSEI_ROC with p-value 0.6102982964903532
# dropping DJI_ULC with p-value 0.6261175925650881
# dropping NSEI_AWE with p-value 0.47731754651671865
# dropping NSEI_ULC with p-value 0.63127349007024
# dropping GDAXI_DAILY_RETURNS with p-value 0.3595764485507975
# dropping DJI_DAILY_RETURNS with p-value 0.2450482051717371
# dropping DJI_KAM with p-value 0.12125951529142992
# dropping DJI_HL_RATIO with p-value 0.05117815872136156
# dropping NSEI_VPT with p-value 0.09922517060049062
# dropping NSEI_SMA with vif 8841.610507825955
# dropping NSEI_RSI with p-value 0.5185776063098049
# dropping DJI_EMA with p-value 0.22095246967703774
# dropping DJI_SMA with p-value 0.09831407286405831



# %% 5 -
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1665
# Model:                          Logit   Df Residuals:                     1654
# Method:                           MLE   Df Model:                           10
# Date:                Fri, 31 May 2024   Pseudo R-squ.:                  0.1809
# Time:                        18:17:41   Log-Likelihood:                -945.24
# converged:                       True   LL-Null:                       -1154.0
# Covariance Type:            nonrobust   LLR p-value:                 1.739e-83
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             19.2830      8.723      2.211      0.027       2.186      36.380
# NSEI_DAILY_RETURNS    -0.2782      0.069     -4.055      0.000      -0.413      -0.144
# IXIC_DAILY_RETURNS     0.5234      0.068      7.712      0.000       0.390       0.656
# HSI_DAILY_RETURNS     -0.1212      0.048     -2.526      0.012      -0.215      -0.027
# N225_DAILY_RETURNS    -0.1218      0.061     -2.004      0.045      -0.241      -0.003
# VIX_DAILY_RETURNS     -0.0678      0.012     -5.606      0.000      -0.092      -0.044
# NSEI_HL_RATIO        -20.0430      8.494     -2.360      0.018     -36.691      -3.395
# DJI_RSI                0.0339      0.012      2.933      0.003       0.011       0.057
# NSEI_TSI               0.0079      0.003      2.534      0.011       0.002       0.014
# DJI_TSI               -0.0215      0.007     -3.066      0.002      -0.035      -0.008
# NSEI_EMA           -4.003e-05   1.76e-05     -2.272      0.023   -7.46e-05    -5.5e-06
# ======================================================================================
# """



# %% 10 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.255195
# 1  IXIC_DAILY_RETURNS  2.142674
# 2   HSI_DAILY_RETURNS  1.307116
# 3  N225_DAILY_RETURNS  1.349896
# 4   VIX_DAILY_RETURNS  2.114771
# 5       NSEI_HL_RATIO  1.210444
# 6             DJI_RSI  4.993466
# 7            NSEI_TSI  1.544530
# 8             DJI_TSI  4.861734
# 9            NSEI_EMA  1.098825



# %% 8 - ROC Curve
y_pred = model.predict(X_train.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y_train, y_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "08_01", "01 - training data", "phase_03")



# %% 9 - Optimal Threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold: {optimal_threshold}')
# Best Threshold: 0.505



# %% 10 - AUC Curve
auc_roc = roc_auc_score(y_train, y_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7864322985812082



# %% 11 - Classification Report
y_pred_class = np.where(y_pred <= optimal_threshold,  0, 1)
print(classification_report(y_train, y_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.73      0.68      0.70       824
#          1.0       0.71      0.76      0.73       841
# 
#     accuracy                           0.72      1665
#    macro avg       0.72      0.72      0.72      1665
# weighted avg       0.72      0.72      0.72      1665



# %% 11 - Confusion Matrix
table = pd.crosstab(y_pred_class, y_train)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0
# 0              558  204
# 1              266  637



# %% 11 - Sensitivity / Specificity
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity: {sensitivity}%")
# Sensitivity: 75.74%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity: {specificity}%")
# Specificity: 67.72%



# %% 12 - ROC Curve
y_test_pred = model.predict(X_test.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "08_02", "02 - test data", "phase_03")



# %% 13 - AUC Curve
auc_roc = roc_auc_score(y_test, y_test_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7691845969156893



# %% 14 - Classification Report
y_test_pred_class = np.where(y_test_pred <= optimal_threshold,  0, 1)
print(classification_report(y_test, y_test_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.74      0.71      0.72       221
#          1.0       0.69      0.72      0.70       196
# 
#     accuracy                           0.71       417
#    macro avg       0.71      0.72      0.71       417
# weighted avg       0.72      0.71      0.71       417



# %% 11 - Confusion Matrix
table = pd.crosstab(y_test_pred_class, y_test)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0
# 0              156   54
# 1               65  142



# %% 11 - Sensitivity / Specificity
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity: {sensitivity}%")
# Sensitivity: 72.45%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity: {specificity}%")
# Specificity: 70.59%
