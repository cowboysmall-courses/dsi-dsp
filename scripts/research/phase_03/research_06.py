
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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from cowboysmall.data.file import read_master_file
from cowboysmall.feature import COLUMNS
from cowboysmall.feature.indicators import get_indicators, get_ratios, INDICATORS, RATIOS
from cowboysmall.plots import plt, sns



# %% 2 -
ALL_COLS = COLUMNS + RATIOS + INDICATORS
FEATURES = ["IXIC_DAILY_RETURNS", "HSI_DAILY_RETURNS", "N225_DAILY_RETURNS", "VIX_DAILY_RETURNS", "DJI_RSI", "DJI_TSI"]



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
X = data.loc[:, FEATURES]
y = data.loc[:, "NSEI_OPEN_DIR"]



# %% 4 -
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337)



# %% 5 -
model = LogisticRegression(max_iter = 1000)
model.fit(X_train, y_train)



# %% 8 - 
y_pred_prob = model.predict_proba(X_train)

fpr, tpr, thresholds = roc_curve(y_train, y_pred_prob[:, 1])

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "06_01", "01 - training data", "phase_03")



# %% 9 - Optimal Threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold is : {optimal_threshold}')
# Best Threshold is : 0.684



# %% 10 - AUC Curve
auc_roc = roc_auc_score(y_train, y_pred_prob[:, 1])
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7530102826256637



# %% 11 - Classification Report
y_pred_class = np.where(y_pred_prob[:, 1] <= optimal_threshold,  0, 1)
print(classification_report(y_train, y_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.53      0.68      0.60       391
#          1.0       0.83      0.72      0.77       829
# 
#     accuracy                           0.71      1220
#    macro avg       0.68      0.70      0.68      1220
# weighted avg       0.73      0.71      0.71      1220



# %% 11 - 
table = pd.crosstab(y_pred_class, y_train)
table
# NSEI_OPEN_DIR    0.0  1.0
# predicted_class          
# 0                265  233
# 1                126  596



# %% 11 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)

print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Sensitivity for cut-off 0.684 is : 71.89%
# Specificity for cut-off 0.684 is : 67.77%



# %% 12 - ROC Curve
y_pred_prob = model.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob[:, 1])

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "06_02", "02 - testing data", "phase_03")



# %% 13 - AUC Curve
auc_roc = roc_auc_score(y_test, y_pred_prob[:, 1])
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7521808088818398



# %% 14 - Classification Report
y_pred_class = np.where(y_pred_prob[:, 1] <= optimal_threshold,  0, 1)
print(classification_report(y_test, y_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.53      0.65      0.58        97
#          1.0       0.82      0.73      0.77       208
# 
#     accuracy                           0.70       305
#    macro avg       0.67      0.69      0.67       305
# weighted avg       0.72      0.70      0.71       305



# %% 11 - 
table = pd.crosstab(y_pred_class, y_test)
table
# NSEI_OPEN_DIR    0.0  1.0
# predicted_class          
# 0                 63   57
# 1                 34  151



# %% 11 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)

print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Sensitivity for cut-off 0.684 is : 72.6%
# Specificity for cut-off 0.684 is : 64.95%
