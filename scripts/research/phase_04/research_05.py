
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from cowboysmall.data.file import read_master_file
from cowboysmall.feature import COLUMNS, INDICATORS, RATIOS
from cowboysmall.feature.indicators import get_indicators, get_ratios
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
model = RandomForestClassifier(random_state = 1337)
model.fit(X_train, y_train)










# %% 6 - ROC Curve
y_train_pred_prob = model.predict_proba(X_train)



# %% 6 -
train_fpr, train_tpr, train_thresholds = roc_curve(y_train, y_train_pred_prob[:, 1])

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(train_fpr, train_tpr, "Random Forest (Train Data)")



# %% 7 - find optimal threshold
optimal_threshold = round(train_thresholds[np.argmax(train_tpr - train_fpr)], 3)
print(f"Optimal Threshold: {optimal_threshold}")
# Optimal Threshold: 0.62



# %% 8 - AUC Curve
train_auc_roc = roc_auc_score(y_train, y_train_pred_prob[:, 1])
print(f"AUC ROC (Train Data): {train_auc_roc}")
# AUC ROC (Train Data): 1.0



# %% 10 - Classification Report
y_train_pred_class = np.where(y_train_pred_prob[:, 1] <= optimal_threshold,  0, 1)
print(classification_report(y_train, y_train_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       1.00      1.00      1.00       391
#          1.0       1.00      1.00      1.00       829
# 
#     accuracy                           1.00      1220
#    macro avg       1.00      1.00      1.00      1220
# weighted avg       1.00      1.00      1.00      1220



# %% 11 - 
table = pd.crosstab(y_train_pred_class, y_train)
print(table)
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0              391    1
# 1                0  828



# %% 12 - 
train_sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
train_specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)

print(f"Sensitivity for cut-off {optimal_threshold}: {train_sensitivity}%")
print(f"Specificity for cut-off {optimal_threshold}: {train_specificity}%")
# Sensitivity for cut-off 0.65: 99.88%
# Specificity for cut-off 0.65: 100.0%










# %% 6 - ROC Curve
y_test_pred_prob = model.predict_proba(X_test)



# %% 6 -
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_test_pred_prob[:, 1])

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(test_fpr, test_tpr, "Random Forest (Test Data)")



# %% 8 - AUC Curve
test_auc_roc = roc_auc_score(y_test, y_test_pred_prob[:, 1])
print(f"AUC ROC (Test Data): {test_auc_roc}")
# AUC ROC (Test Data): 0.7643487311657415



# %% 10 - Classification Report
y_test_pred_class = np.where(y_test_pred_prob[:, 1] <= optimal_threshold,  0, 1)
print(classification_report(y_test, y_test_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.56      0.63      0.59        97
#          1.0       0.82      0.77      0.79       208
# 
#     accuracy                           0.72       305
#    macro avg       0.69      0.70      0.69       305
# weighted avg       0.73      0.72      0.73       305



# %% 11 - 
table = pd.crosstab(y_test_pred_class, y_test)
print(table)
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0               61   48
# 1               36  160



# %% 12 - 
test_sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
test_specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)

print(f"Sensitivity for cut-off {optimal_threshold}: {test_sensitivity}%")
print(f"Specificity for cut-off {optimal_threshold}: {test_specificity}%")
# Sensitivity for cut-off 0.62: 76.92%
# Specificity for cut-off 0.62: 62.89%










# %% 13 - 
print(f"AUC ROC (Train Data): {train_auc_roc}")
print(f"AUC ROC  (Test Data): {test_auc_roc}")
# AUC ROC (Train Data): 1.0
# AUC ROC  (Test Data): 0.7643487311657415
