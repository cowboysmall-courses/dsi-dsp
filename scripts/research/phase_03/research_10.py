
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
remedy = RandomOverSampler(random_state = 0)
X, y = remedy.fit_resample(X, y)


# %% 4 -
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337)



# %% 5 -
model, dropped = prune(X_train, y_train)
# dropping NSEI_RSI with p-value 0.9812360918367136
# dropping NSEI_HL_RATIO with p-value 0.9555690785367489
# dropping GDAXI_DAILY_RETURNS with p-value 0.7309944978936169
# dropping DJI_DAILY_RETURNS with p-value 0.7987178548468027
# dropping NSEI_TSI with p-value 0.10327168907758438



# %% 5 -
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1659
# Model:                          Logit   Df Residuals:                     1650
# Method:                           MLE   Df Model:                            8
# Date:                Mon, 10 Jun 2024   Pseudo R-squ.:                  0.1644
# Time:                        23:47:27   Log-Likelihood:                -960.84
# converged:                       True   LL-Null:                       -1149.9
# Covariance Type:            nonrobust   LLR p-value:                 8.662e-77
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             22.7498      9.372      2.427      0.015       4.380      41.119
# NSEI_DAILY_RETURNS    -0.2038      0.062     -3.299      0.001      -0.325      -0.083
# IXIC_DAILY_RETURNS     0.4445      0.062      7.156      0.000       0.323       0.566
# HSI_DAILY_RETURNS     -0.1012      0.045     -2.255      0.024      -0.189      -0.013
# N225_DAILY_RETURNS    -0.1594      0.057     -2.804      0.005      -0.271      -0.048
# VIX_DAILY_RETURNS     -0.0585      0.011     -5.113      0.000      -0.081      -0.036
# DJI_HL_RATIO         -23.8037      9.133     -2.606      0.009     -41.703      -5.904
# DJI_RSI                0.0295      0.011      2.688      0.007       0.008       0.051
# DJI_TSI               -0.0172      0.006     -2.712      0.007      -0.030      -0.005
# ======================================================================================
# """



# %% 10 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.221564
# 1  IXIC_DAILY_RETURNS  2.199315
# 2   HSI_DAILY_RETURNS  1.295595
# 3  N225_DAILY_RETURNS  1.327927
# 4   VIX_DAILY_RETURNS  2.240810
# 5        DJI_HL_RATIO  1.444492
# 6             DJI_RSI  4.957651
# 7             DJI_TSI  4.422130



# %% 8 - ROC Curve
y_pred = model.predict(X_train.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y_train, y_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "10_01", "01 - training data", "phase_03")



# %% 9 - Optimal Threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold: {optimal_threshold}')
# Best Threshold: 0.488



# %% 10 - AUC Curve
auc_roc = roc_auc_score(y_train, y_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7699332916709055



# %% 11 - Classification Report
y_pred_class = np.where(y_pred <= optimal_threshold,  0, 1)
print(classification_report(y_train, y_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.75      0.64      0.69       830
#          1.0       0.69      0.79      0.74       829
# 
#     accuracy                           0.72      1659
#    macro avg       0.72      0.72      0.72      1659
# weighted avg       0.72      0.72      0.72      1659



# %% 11 - Confusion Matrix
table = pd.crosstab(y_pred_class, y_train)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0              533  173
# 1              297  656



# %% 11 - Sensitivity / Specificity
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)

print(f"Sensitivity: {sensitivity}%")
print(f"Specificity: {specificity}%")
# Sensitivity: 79.13%
# Specificity: 64.22%



# %% 12 - ROC Curve
y_test_pred = model.predict(X_test.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "10_02", "02 - test data", "phase_03")



# %% 13 - AUC Curve
auc_roc = roc_auc_score(y_test, y_test_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7293989223337048



# %% 14 - Classification Report
y_test_pred_class = np.where(y_test_pred <= optimal_threshold,  0, 1)
print(classification_report(y_test, y_test_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.68      0.62      0.65       207
#          1.0       0.65      0.71      0.68       208
# 
#     accuracy                           0.67       415
#    macro avg       0.67      0.67      0.67       415
# weighted avg       0.67      0.67      0.67       415



# %% 11 - Confusion Matrix
table = pd.crosstab(y_test_pred_class, y_test)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0              129   60
# 1               78  148



# %% 11 - Sensitivity / Specificity
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)

print(f"Sensitivity: {sensitivity}%")
print(f"Specificity: {specificity}%")
# Sensitivity: 71.15%
# Specificity: 62.32%
