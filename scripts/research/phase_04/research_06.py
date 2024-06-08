
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
y = data['NSEI_OPEN_DIR']



# %% 4 -
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337)



# %% 1 -
model  = MLPClassifier(hidden_layer_sizes = (100, 100, 100), max_iter = 1000, alpha = 0.00001, random_state = 1337)

scaler = MinMaxScaler()
# scaler = StandardScaler()

model.fit(scaler.fit_transform(X_train), y_train)



# %% 6 - ROC Curve
y_pred = model.predict_proba(scaler.transform(X_test))



# %% 6 -
fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1])

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "06_01", "Neural Network", "phase_04")



# %% 7 - find optimal threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold is : {optimal_threshold}')
# Best Threshold is : 0.606



# %% 8 - AUC Curve
auc_roc = roc_auc_score(y_test, y_pred[:, 1])
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7532216494845361



# %% 10 - Classification Report
y_pred_class = np.where(y_pred[:, 1] <= optimal_threshold,  0, 1)
print(classification_report(y_test, y_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.66      0.56      0.60        97
#          1.0       0.81      0.87      0.84       208
# 
#     accuracy                           0.77       305
#    macro avg       0.73      0.71      0.72       305
# weighted avg       0.76      0.77      0.76       305



# %% 11 - 
table = pd.crosstab(y_pred_class, y_test)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0               54   28
# 1               43  180



# %% 12 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)

print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Sensitivity for cut-off 0.606 is : 86.54%
# Specificity for cut-off 0.606 is : 55.67%

# %%
