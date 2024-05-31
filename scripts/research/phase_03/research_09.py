
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
remedy = SMOTE(random_state = 0)
X, y = remedy.fit_resample(X, y)


# %% 4 -
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337)



# %% 5 -
model, dropped = pruned_logit(X_train, y_train)
# dropping DJI_VPT with p-value 0.9503407346883762
# dropping NSEI_ULC with p-value 0.9419580759752988
# dropping NSEI_KAM with p-value 0.9009616390962344
# dropping DJI_DAILY_RETURNS with p-value 0.7858801172293697
# dropping NSEI_HL_RATIO with p-value 0.40790007928874816
# dropping GDAXI_DAILY_RETURNS with p-value 0.34922079603916245
# dropping DJI_ROC with p-value 0.3188058144489745
# dropping DJI_AWE with p-value 0.34563480937881996
# dropping DJI_ULC with p-value 0.34703032031464
# dropping NSEI_VPT with p-value 0.4313142264901323
# dropping NSEI_AWE with p-value 0.18466276921805558
# dropping NSEI_ROC with p-value 0.2806128454743049
# dropping DJI_KAM with p-value 0.11713209162036956
# dropping NSEI_RSI with p-value 0.09659158283625409
# dropping NSEI_EMA with p-value 0.32200395077865207
# dropping NSEI_SMA with p-value 0.11985429976898684
# dropping DJI_SMA with vif 1568.1244448151604
# dropping DJI_EMA with p-value 0.055576576132791156
# dropping DJI_RSI with vif 5.2921843100754815
# dropping DJI_TSI with p-value 0.14808713473515234
# dropping NSEI_TSI with p-value 0.13966441258032655



# %% 5 -
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1659
# Model:                          Logit   Df Residuals:                     1652
# Method:                           MLE   Df Model:                            6
# Date:                Fri, 31 May 2024   Pseudo R-squ.:                  0.1716
# Time:                        18:44:26   Log-Likelihood:                -952.66
# converged:                       True   LL-Null:                       -1149.9
# Covariance Type:            nonrobust   LLR p-value:                 4.149e-82
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             26.6137      8.343      3.190      0.001      10.261      42.966
# NSEI_DAILY_RETURNS    -0.2180      0.065     -3.334      0.001      -0.346      -0.090
# IXIC_DAILY_RETURNS     0.5772      0.066      8.779      0.000       0.448       0.706
# HSI_DAILY_RETURNS     -0.0979      0.047     -2.063      0.039      -0.191      -0.005
# N225_DAILY_RETURNS    -0.1580      0.058     -2.718      0.007      -0.272      -0.044
# VIX_DAILY_RETURNS     -0.0604      0.012     -5.166      0.000      -0.083      -0.037
# DJI_HL_RATIO         -26.1851      8.249     -3.174      0.002     -42.354     -10.016
# ======================================================================================
# """



# %% 10 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.245344
# 1  IXIC_DAILY_RETURNS  2.020044
# 2   HSI_DAILY_RETURNS  1.315941
# 3  N225_DAILY_RETURNS  1.304717
# 4   VIX_DAILY_RETURNS  1.999569
# 5        DJI_HL_RATIO  1.083216



# %% 8 - ROC Curve
y_pred = model.predict(X_train.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y_train, y_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "09_01", "01 - training data", "phase_03")



# %% 9 - Optimal Threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold: {optimal_threshold}')
# Best Threshold: 0.503



# %% 10 - AUC Curve
auc_roc = roc_auc_score(y_train, y_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7759850015260075



# %% 11 - Classification Report
y_pred_class = np.where(y_pred <= optimal_threshold,  0, 1)
print(classification_report(y_train, y_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.74      0.66      0.70       830
#          1.0       0.70      0.77      0.73       829
# 
#     accuracy                           0.72      1659
#    macro avg       0.72      0.72      0.72      1659
# weighted avg       0.72      0.72      0.72      1659



# %% 11 - Confusion Matrix
table = pd.crosstab(y_pred_class, y_train)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0              549  188
# 1              281  641



# %% 11 - Sensitivity / Specificity
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity: {sensitivity}%")
# Sensitivity: 77.32%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity: {specificity}%")
# Specificity: 66.14%



# %% 12 - ROC Curve
y_test_pred = model.predict(X_test.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "09_02", "02 - test data", "phase_03")



# %% 13 - AUC Curve
auc_roc = roc_auc_score(y_test, y_test_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7434968413229283



# %% 14 - Classification Report
y_test_pred_class = np.where(y_test_pred <= optimal_threshold,  0, 1)
print(classification_report(y_test, y_test_pred_class))
#               precision    recall  f1-score   support

#          0.0       0.67      0.65      0.66       207
#          1.0       0.66      0.68      0.67       208

#     accuracy                           0.67       415
#    macro avg       0.67      0.67      0.66       415
# weighted avg       0.67      0.67      0.67       415



# %% 11 - Confusion Matrix
table = pd.crosstab(y_test_pred_class, y_test)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0              135   67
# 1               72  141



# %% 11 - Sensitivity / Specificity
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity: {sensitivity}%")
# Sensitivity: 67.79%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity: {specificity}%")
# Specificity: 65.22%
