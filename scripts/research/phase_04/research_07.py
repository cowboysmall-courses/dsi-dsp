
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
from cowboysmall.feature.indicators import get_indicators, INDICATORS
from cowboysmall.plots import plt, sns



# %% 2 -
INDICES  = ['NSEI', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']
COLUMNS  = [f"{index}_DAILY_RETURNS" for index in INDICES]
RATIOS   = ["NSEI_HL_RATIO", "DJI_HL_RATIO"]

ALL_COLS = COLUMNS + RATIOS + INDICATORS

FEATURES = ["IXIC_DAILY_RETURNS", "HSI_DAILY_RETURNS", "N225_DAILY_RETURNS", "VIX_DAILY_RETURNS", "DJI_RSI", "DJI_TSI"]



# %% 2 -
class MLP(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()

        self.input  = nn.Linear(input_dim, 16)
        self.act_in = nn.ReLU()

        self.hidden1 = nn.Linear(16, 16)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(16, 16)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(16, 16)
        self.act3 = nn.ReLU()
        self.hidden4 = nn.Linear(16, 16)
        self.act4 = nn.ReLU()
        self.hidden5 = nn.Linear(16, 16)
        self.act5 = nn.ReLU()

        self.output  = nn.Linear(16, output_dim)
        self.act_out = nn.Sigmoid()

    def forward(self, x):
        x = self.act_in(self.input(x))

        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act4(self.hidden4(x))
        x = self.act5(self.hidden5(x))

        return self.act_out(self.output(x))



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
scaler = MinMaxScaler(feature_range = (0, 1))

X = scaler.fit_transform(data[ALL_COLS].values) 
y = data['NSEI_OPEN_DIR'].values[:, None]




# %% 4 -
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337)


# %% 1 -
X_train = torch.from_numpy(X_train).type(torch.Tensor)
X_test  = torch.from_numpy(X_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)



# %% 1 -
input_dim  = X.shape[1]
output_dim = 1

model     = MLP(input_dim, output_dim)
criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr = 0.01)



# %% 1 -
epochs = 500
losses = []

for epoch in range(epochs):
    out  = model(X_train)
    loss = criterion(out, y_train)

    losses.append(loss.item())

    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    if epoch % 10 == 9:
        print(f"Epoch {epoch + 1:>3} - MSE: {loss.item()}")



# %% 1 -
plot.figure(figsize = (15, 6))
plot.plot(losses, label = "Loss")
plot.legend()
plot.show()



# %% 1 -
y_pred = model(X_test).detach().cpu().numpy()



# %% 6 - ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "07_01", "Neural Network", "phase_04")



# %% 7 - find optimal threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold is : {optimal_threshold}')
# Best Threshold is : 0.606



# %% 8 - AUC Curve
auc_roc = roc_auc_score(y_test, y_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7532216494845361



# %% 10 - Classification Report
y_pred_class = np.where(y_pred <= optimal_threshold,  0, 1)
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
table = pd.crosstab(y_pred_class[:, 0], y_test[:, 0])
print(table)
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
