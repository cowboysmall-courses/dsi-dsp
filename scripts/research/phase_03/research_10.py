
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

from imblearn.over_sampling import RandomOverSampler

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
remedy = RandomOverSampler(random_state = 0)
X, y = remedy.fit_resample(X, y)


# %% 4 -
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337)



# %% 5 -
model, dropped = pruned_logit(X_train, y_train)
# dropping DJI_VPT with p-value 0.9927247573053275
# dropping NSEI_ULC with p-value 0.9834423168085367
# dropping GDAXI_DAILY_RETURNS with p-value 0.9055089083716026
# dropping NSEI_AWE with p-value 0.8327754530751741
# dropping DJI_ULC with p-value 0.6445575704291319
# dropping DJI_DAILY_RETURNS with p-value 0.5671020701222045
# dropping DJI_AWE with p-value 0.4957507357257642
# dropping NSEI_RSI with p-value 0.4288381836417712
# dropping NSEI_EMA with p-value 0.3669734738564341
# dropping DJI_ROC with p-value 0.27366010854256273
# dropping NSEI_ROC with p-value 0.461113004372243
# dropping NSEI_HL_RATIO with p-value 0.2648241873157965
# dropping NSEI_VPT with p-value 0.20559770889982631
# dropping NSEI_KAM with p-value 0.15936659283928178
# dropping DJI_SMA with p-value 0.10600879459458454
# dropping DJI_EMA with p-value 0.10493567101610185
# dropping DJI_KAM with p-value 0.09378003244146865
# dropping DJI_RSI with vif 5.013790481111531
# dropping DJI_TSI with p-value 0.0939226246333018
# dropping NSEI_TSI with p-value 0.20572130483023332
# dropping NSEI_SMA with p-value 0.07426761248215988



# %% 5 -
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1659
# Model:                          Logit   Df Residuals:                     1652
# Method:                           MLE   Df Model:                            6
# Date:                Fri, 31 May 2024   Pseudo R-squ.:                  0.1610
# Time:                        21:09:52   Log-Likelihood:                -964.82
# converged:                       True   LL-Null:                       -1149.9
# Covariance Type:            nonrobust   LLR p-value:                 7.019e-77
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             25.7098      7.995      3.216      0.001      10.040      41.379
# NSEI_DAILY_RETURNS    -0.1882      0.061     -3.082      0.002      -0.308      -0.069
# IXIC_DAILY_RETURNS     0.4732      0.061      7.748      0.000       0.353       0.593
# HSI_DAILY_RETURNS     -0.1008      0.045     -2.259      0.024      -0.188      -0.013
# N225_DAILY_RETURNS    -0.1242      0.055     -2.255      0.024      -0.232      -0.016
# VIX_DAILY_RETURNS     -0.0628      0.011     -5.558      0.000      -0.085      -0.041
# DJI_HL_RATIO         -25.2887      7.905     -3.199      0.001     -40.782      -9.795
# ======================================================================================
# """



# %% 10 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.215590
# 1  IXIC_DAILY_RETURNS  2.138505
# 2   HSI_DAILY_RETURNS  1.295138
# 3  N225_DAILY_RETURNS  1.267026
# 4   VIX_DAILY_RETURNS  2.124434
# 5        DJI_HL_RATIO  1.093974



# %% 8 - ROC Curve
y_pred = model.predict(X_train.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y_train, y_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "10_01", "01 - training data", "phase_03")



# %% 9 - Optimal Threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold: {optimal_threshold}')
# Best Threshold: 0.505



# %% 10 - AUC Curve
auc_roc = roc_auc_score(y_train, y_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.767804147833796



# %% 11 - Classification Report
y_pred_class = np.where(y_pred <= optimal_threshold,  0, 1)
print(classification_report(y_train, y_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.75      0.66      0.70       830
#          1.0       0.69      0.77      0.73       829
# 
#     accuracy                           0.72      1659
#    macro avg       0.72      0.72      0.72      1659
# weighted avg       0.72      0.72      0.72      1659



# %% 11 - Confusion Matrix
table = pd.crosstab(y_pred_class, y_train)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0              548  187
# 1              282  642



# %% 11 - Sensitivity / Specificity
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity: {sensitivity}%")
# Sensitivity: 77.44%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity: {specificity}%")
# Specificity: 66.02%



# %% 12 - ROC Curve
y_test_pred = model.predict(X_test.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "10_02", "02 - test data", "phase_03")



# %% 13 - AUC Curve
auc_roc = roc_auc_score(y_test, y_test_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7234299516908211



# %% 14 - Classification Report
y_test_pred_class = np.where(y_test_pred <= optimal_threshold,  0, 1)
print(classification_report(y_test, y_test_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.66      0.64      0.65       207
#          1.0       0.65      0.67      0.66       208
# 
#     accuracy                           0.65       415
#    macro avg       0.65      0.65      0.65       415
# weighted avg       0.65      0.65      0.65       415



# %% 11 - Confusion Matrix
table = pd.crosstab(y_test_pred_class, y_test)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0              132   69
# 1               75  139



# %% 11 - Sensitivity / Specificity
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity: {sensitivity}%")
# Sensitivity: 66.83%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity: {specificity}%")
# Specificity: 63.77%
