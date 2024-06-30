
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
import matplotlib.pyplot as plot

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
import torch.nn as nn

from cowboysmall.data.file import read_master_file
from cowboysmall.feature import COLUMNS, INDICATORS, RATIOS
from cowboysmall.feature.indicators import get_indicators, get_ratios
from cowboysmall.plots import plt, sns
from cowboysmall.model.training import train_batched



# %% 1 -
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



# %% 2 -
ALL_COLS = COLUMNS + RATIOS + INDICATORS
FEATURES = ["IXIC_DAILY_RETURNS", "HSI_DAILY_RETURNS", "N225_DAILY_RETURNS", "VIX_DAILY_RETURNS", "DJI_RSI", "DJI_TSI"]



# %% 2 -
class MLP(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()

        self.input  = nn.Linear(input_dim, 64)
        self.act_in = nn.ReLU()

        self.hidden1 = nn.Linear(64, 64)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(64, 64)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(64, 64)
        self.act3 = nn.ReLU()
        self.hidden4 = nn.Linear(64, 64)
        self.act4 = nn.ReLU()
        self.hidden5 = nn.Linear(64, 64)
        self.act5 = nn.ReLU()
        self.hidden6 = nn.Linear(64, 64)
        self.act6 = nn.ReLU()
        self.hidden7 = nn.Linear(64, 64)
        self.act7 = nn.ReLU()

        self.output  = nn.Linear(64, output_dim)
        self.act_out = nn.Sigmoid()

    def forward(self, x):
        x = self.act_in(self.input(x))

        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act4(self.hidden4(x))
        x = self.act5(self.hidden5(x))
        x = self.act6(self.hidden6(x))
        x = self.act7(self.hidden7(x))

        return self.act_out(self.output(x))



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
scaler = MinMaxScaler(feature_range = (0, 1))

X = scaler.fit_transform(data[ALL_COLS].values) 
y = data['NSEI_OPEN_DIR'].values[:, None]



# %% 4 -
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337)


# %% 1 -
X_train = torch.from_numpy(X_train).type(torch.Tensor)
X_test  = torch.from_numpy(X_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)

if torch.cuda.is_available():
    X_train = X_train.cuda()
    X_test  = X_test.cuda()
    y_train = y_train.cuda()



# %% 1 -
input_dim  = X.shape[1]
output_dim = 1

model     = MLP(input_dim, output_dim)
model     = model.to(device)

criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr = 0.0001)



# %% 1 -
losses = train_batched(X_train, y_train, model, criterion, optimiser, batch_size = 100, shuffle = True)



# %% 1 -
plot.figure(figsize = (15, 6))
plot.plot(losses, label = "Loss")
plot.legend()
plot.show()













# %% 1 -
y_train_pred_prob = model(X_train).detach().cpu().numpy()



# %% 6 - ROC Curve
train_fpr, train_tpr, train_thresholds = roc_curve(y_train, y_train_pred_prob)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(train_fpr, train_tpr, "Neural Network")



# %% 7 - find optimal threshold
optimal_threshold = round(train_thresholds[np.argmax(train_tpr - train_fpr)], 3)
print(f'Best Threshold is : {optimal_threshold}')
# Best Threshold is : 0.5429999828338623



# %% 8 - AUC Curve
train_auc_roc = roc_auc_score(y_train, y_train_pred_prob)
print(f'AUC ROC: {train_auc_roc}')
# AUC ROC: 0.7608049167327517



# %% 10 - Classification Report
y_train_pred_class = np.where(y_train_pred_prob <= optimal_threshold,  0, 1)
print(classification_report(y_train, y_train_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.69      0.55      0.61        97
#          1.0       0.81      0.88      0.84       208
# 
#     accuracy                           0.78       305
#    macro avg       0.75      0.72      0.73       305
# weighted avg       0.77      0.78      0.77       305



# %% 11 - 
table = pd.crosstab(y_train_pred_class[:, 0], y_train[:, 0])
print(table)
# col_0  0.0  1.0
# row_0          
# 0       53   24
# 1       44  184



# %% 12 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)

print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Sensitivity for cut-off 0.5429999828338623 is : 88.46%
# Specificity for cut-off 0.5429999828338623 is : 54.64%












# %% 1 -
y_test_pred_prob = model(X_test).detach().cpu().numpy()



# %% 6 - ROC Curve
test_fpr, test_tpr, test_thresholds = roc_curve(y_test, y_test_pred_prob)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(test_fpr, test_tpr, "Neural Network")



# %% 8 - AUC Curve
test_auc_roc = roc_auc_score(y_test, y_test_pred_prob)
print(f'AUC ROC: {test_auc_roc}')
# AUC ROC: 0.7608049167327517



# %% 10 - Classification Report
y_test_pred_class = np.where(y_test_pred_prob <= optimal_threshold,  0, 1)
print(classification_report(y_test, y_test_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.69      0.55      0.61        97
#          1.0       0.81      0.88      0.84       208
# 
#     accuracy                           0.78       305
#    macro avg       0.75      0.72      0.73       305
# weighted avg       0.77      0.78      0.77       305



# %% 11 - 
table = pd.crosstab(y_test_pred_class[:, 0], y_test[:, 0])
print(table)
# col_0  0.0  1.0
# row_0          
# 0       53   24
# 1       44  184



# %% 12 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)

print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Sensitivity for cut-off 0.5429999828338623 is : 88.46%
# Specificity for cut-off 0.5429999828338623 is : 54.64%
