
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

from cowboysmall.data.file import read_master_file
from cowboysmall.feature.indicators import get_indicators, INDICATORS
from cowboysmall.plots import plt, sns










# %% 2 -
INDICES  = ['NSEI', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']
COLUMNS  = [f"{index}_DAILY_RETURNS" for index in INDICES]
RATIOS   = ["NSEI_HL_RATIO", "DJI_HL_RATIO"]

ALL_COLS = COLUMNS + RATIOS + INDICATORS

FEATURES = ["IXIC_DAILY_RETURNS", "HSI_DAILY_RETURNS", "N225_DAILY_RETURNS", "VIX_DAILY_RETURNS", "DJI_RSI", "DJI_TSI"]



# %% 2 -
master = read_master_file()



# %% 2 -
master["NSEI_OPEN_DIR"] = np.where(master["NSEI_OPEN"] > master["NSEI_CLOSE"].shift(), 1, 0)



# %% 2 -
master["NSEI_HL_RATIO"] = master["NSEI_HIGH"] / master["NSEI_LOW"]
master["DJI_HL_RATIO"]  = master["DJI_HIGH"] / master["DJI_LOW"]



# %% 2 -
master = get_indicators(master)



# %% 3 -
data = pd.concat([master["NSEI_OPEN_DIR"].shift(-1), master[ALL_COLS]], axis = 1)
data.dropna(inplace = True)
data.head()



# %% 3 -
X = data[ALL_COLS]
y = data["NSEI_OPEN_DIR"]



# %% 4 -
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337)



# %% 1 -
model = GaussianNB()
model.fit(X_train, y_train)










# %% 6 - ROC Curve
y_train_pred = model.predict_proba(X_train)



# %% 6 -
train_fpr, train_tpr, train_thresholds = roc_curve(y_train, y_train_pred[:, 1])

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(train_fpr, train_tpr, "02_01", "Naive Bayes (Train Data)", "phase_04")



# %% 7 - find optimal threshold
optimal_threshold = round(train_thresholds[np.argmax(train_tpr - train_fpr)], 3)
print(f"Optimal Threshold: {optimal_threshold}")
# Optimal Threshold: 0.608



# %% 8 - AUC Curve
train_auc_roc = roc_auc_score(y_train, y_train_pred[:, 1])
print(f"AUC ROC (Train Data): {train_auc_roc}")
# AUC ROC (Train Data): 0.5482647876374025



# %% 10 - Classification Report
y_train_pred_class = np.where(y_train_pred[:, 1] <= optimal_threshold,  0, 1)
print(classification_report(y_train, y_train_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.38      0.45      0.41       391
#          1.0       0.72      0.66      0.69       829
# 
#     accuracy                           0.59      1220
#    macro avg       0.55      0.55      0.55      1220
# weighted avg       0.61      0.59      0.60      1220



# %% 11 - 
table = pd.crosstab(y_train_pred_class, y_train)
print(table)
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0              176  284
# 1              215  545



# %% 12 - 
train_sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
train_specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)

print(f"Sensitivity for cut-off {optimal_threshold}: {train_sensitivity}%")
print(f"Specificity for cut-off {optimal_threshold}: {train_specificity}%")
# Sensitivity for cut-off 0.608: 65.74%
# Specificity for cut-off 0.608: 45.01%










# %% 6 - ROC Curve
y_test_pred = model.predict_proba(X_test)



# %% 6 -
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_test_pred[:, 1])

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(test_fpr, test_tpr, "02_02", "Naive Bayes (Test Data)", "phase_04")



# %% 8 - AUC Curve
test_auc_roc = roc_auc_score(y_test, y_test_pred[:, 1])
print(f"AUC ROC (Test Data): {test_auc_roc}")
# AUC ROC (Test Data): 0.5299861221252974



# %% 10 - Classification Report
y_test_pred_class = np.where(y_test_pred[:, 1] <= optimal_threshold,  0, 1)
print(classification_report(y_test, y_test_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.34      0.41      0.37        97
#          1.0       0.70      0.62      0.66       208
# 
#     accuracy                           0.56       305
#    macro avg       0.52      0.52      0.52       305
# weighted avg       0.58      0.56      0.57       305



# %% 11 - 
table = pd.crosstab(y_test_pred_class, y_test)
print(table)
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0               40   78
# 1               57  130



# %% 12 - 
test_sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
test_specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)

print(f"Sensitivity for cut-off {optimal_threshold}: {test_sensitivity}%")
print(f"Specificity for cut-off {optimal_threshold}: {test_specificity}%")
# Sensitivity for cut-off 0.608: 62.5%
# Specificity for cut-off 0.608: 41.24%










# %% 13 - 
print(f"AUC ROC (Train Data): {train_auc_roc}")
print(f"AUC ROC  (Test Data): {test_auc_roc}")
# AUC ROC (Train Data): 0.5482647876374025
# AUC ROC  (Test Data): 0.5299861221252974
