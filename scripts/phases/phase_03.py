
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



# %% 0 - import required libraries
import pandas as pd
import numpy as np

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler

from cowboysmall.data.file import read_master_file
from cowboysmall.model.logit import prune
from cowboysmall.feature import COLUMNS, INDICATORS, RATIOS, ALL_COLS
from cowboysmall.feature.indicators import get_indicators, get_ratios
from cowboysmall.plots import plt, sns




# %% 1 - read master data + create NSEI_OPEN_DIR
data = read_master_file()
data = get_ratios(data)
data = get_indicators(data)

data["NSEI_OPEN_DIR"] = np.where(data["NSEI_OPEN"] > data["NSEI_CLOSE"].shift(), 1, 0)

data = pd.concat([data["NSEI_OPEN_DIR"].shift(-1), data[ALL_COLS]], axis = 1)
data = data.dropna()

X = data[ALL_COLS]
y = data['NSEI_OPEN_DIR']

X.insert(loc = 0, column = "Intercept", value = 1)




# %% 2 - partition into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337)




# %% 3 - create binary logistic regression
model, dropped = prune(X_train, y_train)




# %% 4 - confirm no multicolinearity
vif = pd.DataFrame()
vif["Feature"] = model.model.exog_names[1:]
vif["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
print(vif)




# %% 5 - confirm all variables are significant
model.summary()




# %% 6 - ROC Curve for train data
y_pred = model.predict(X_train.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y_train, y_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "PHASE_03_01", "01 - train data", "phase_03")

auc_roc = roc_auc_score(y_train, y_pred)
print(f'AUC ROC: {auc_roc}')




# %% 7 - obtain optimal threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Optimal Threshold: {optimal_threshold}')




# %% 8 - obtain sensitivity and specificity for train data
y_pred_class = np.where(y_pred <= optimal_threshold, 0, 1)
print(classification_report(y_train, y_pred_class))

table = pd.crosstab(y_pred_class, y_train)
print(f"\n{table}\n")

sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)

print(f"Sensitivity: {sensitivity}%")
print(f"Specificity: {specificity}%")




# %% 8 - ROC Curve for test data
y_test_pred = model.predict(X_test.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "PHASE_03_02", "02 - test data", "phase_03")

auc_roc = roc_auc_score(y_test, y_test_pred)
print(f'AUC ROC: {auc_roc}')




# %% 9 - obtain sensitivity and specificity for test data
y_test_pred_class = np.where(y_test_pred <= optimal_threshold, 0, 1)
print(classification_report(y_test, y_test_pred_class))

table = pd.crosstab(y_test_pred_class, y_test)
print(f"\n{table}\n")

sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)

print(f"Sensitivity: {sensitivity}%")
print(f"Specificity: {specificity}%")
