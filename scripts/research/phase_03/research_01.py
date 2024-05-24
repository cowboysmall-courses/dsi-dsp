
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
import ta

from statsmodels.formula.api import logit
from statsmodels.api import Logit
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import classification_report, roc_curve, roc_auc_score

from cowboysmall.data.file import read_master_file
from cowboysmall.feature.indicators import calculate_rsi
from cowboysmall.plots import plt, sns



# %% 2 -
INDICES    = ['NSEI', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']
COLUMNS    = [f"{index}_DAILY_RETURNS" for index in INDICES]
EXTRA_COLS = ["NSEI_HL_RATIO", "DJI_HL_RATIO", "NSEI_RSI", "DJI_RSI", "NSEI_ROC", "DJI_ROC"]

ALL_COLS   = COLUMNS + EXTRA_COLS


# %% 2 -
master = read_master_file()

master["NSEI_OPEN_DIR"] = np.where(master["NSEI_OPEN"] > master["NSEI_CLOSE"].shift(), 1, 0)

master["NSEI_HL_RATIO"] = master["NSEI_HIGH"] / master["NSEI_LOW"]
master["DJI_HL_RATIO"]  = master["DJI_HIGH"] / master["DJI_LOW"]

# master["NSEI_RSI"]      = calculate_rsi(master["NSEI_CLOSE"])
# master["DJI_RSI"]       = calculate_rsi(master["DJI_CLOSE"])

master["NSEI_RSI"]      = ta.momentum.rsi(master["NSEI_CLOSE"])
master["DJI_RSI"]       = ta.momentum.rsi(master["DJI_CLOSE"])

master["NSEI_ROC"]      = ta.momentum.roc(master["NSEI_CLOSE"])
master["DJI_ROC"]       = ta.momentum.roc(master["DJI_CLOSE"])



# %% 2 -
# master[["NSEI_OPEN_DIR"]].to_csv("./data/NSEI_OPEN_DIR_01.csv")

counts = master['NSEI_OPEN_DIR'].value_counts().reset_index()
counts.columns = ['NSEI_OPEN_DIR', 'Freq']
print(counts)
#    NSEI_OPEN_DIR  Freq
# 0              1  1064
# 1              0   499



# %% 3 -
data = pd.concat([master["NSEI_OPEN_DIR"].shift(-1), master[ALL_COLS]], axis = 1)
data.dropna(inplace = True)
data.head()



# %% 3 -
X = data[ALL_COLS]
y = data['NSEI_OPEN_DIR']



# %% 4 -
# model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + DJI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + N225_DAILY_RETURNS + GDAXI_DAILY_RETURNS + VIX_DAILY_RETURNS + NSEI_HL_RATIO + DJI_HL_RATIO + NSEI_RSI + DJI_RSI', data = data).fit()
model = Logit(y, X).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1549
# Model:                          Logit   Df Residuals:                     1536
# Method:                           MLE   Df Model:                           12
# Date:                Fri, 24 May 2024   Pseudo R-squ.:                  0.1436
# Time:                        21:17:38   Log-Likelihood:                -833.08
# converged:                       True   LL-Null:                       -972.76
# Covariance Type:            nonrobust   LLR p-value:                 9.946e-53
# =======================================================================================
#                           coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# NSEI_DAILY_RETURNS     -0.2679      0.074     -3.626      0.000      -0.413      -0.123
# DJI_DAILY_RETURNS       0.1440      0.116      1.240      0.215      -0.084       0.372
# IXIC_DAILY_RETURNS      0.4052      0.081      4.979      0.000       0.246       0.565
# HSI_DAILY_RETURNS      -0.0946      0.049     -1.929      0.054      -0.191       0.002
# N225_DAILY_RETURNS     -0.0967      0.062     -1.557      0.119      -0.219       0.025
# GDAXI_DAILY_RETURNS     0.0428      0.069      0.623      0.533      -0.092       0.177
# VIX_DAILY_RETURNS      -0.0449      0.012     -3.697      0.000      -0.069      -0.021
# NSEI_HL_RATIO           2.2713      8.648      0.263      0.793     -14.678      19.221
# DJI_HL_RATIO           -2.1682      8.591     -0.252      0.801     -19.006      14.670
# NSEI_RSI                0.0101      0.008      1.197      0.231      -0.006       0.027
# DJI_RSI                 0.0040      0.009      0.421      0.674      -0.014       0.022
# NSEI_ROC               -0.0245      0.034     -0.720      0.471      -0.091       0.042
# DJI_ROC                 0.0102      0.032      0.316      0.752      -0.053       0.074
# =======================================================================================
# """



# list of insignificant variables
# 
# DJI_DAILY_RETURNS       0.1440      0.116      1.240      0.215      -0.084       0.372
# HSI_DAILY_RETURNS      -0.0946      0.049     -1.929      0.054      -0.191       0.002
# N225_DAILY_RETURNS     -0.0967      0.062     -1.557      0.119      -0.219       0.025
# GDAXI_DAILY_RETURNS     0.0428      0.069      0.623      0.533      -0.092       0.177
# NSEI_HL_RATIO           2.2713      8.648      0.263      0.793     -14.678      19.221
# DJI_HL_RATIO           -2.1682      8.591     -0.252      0.801     -19.006      14.670
# NSEI_RSI                0.0101      0.008      1.197      0.231      -0.006       0.027
# DJI_RSI                 0.0040      0.009      0.421      0.674      -0.014       0.022
# NSEI_ROC               -0.0245      0.034     -0.720      0.471      -0.091       0.042
# DJI_ROC                 0.0102      0.032      0.316      0.752      -0.053       0.074



# %% 5 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
vif_data
#                 Feature       VIF
# 0    NSEI_DAILY_RETURNS  1.427738
# 1     DJI_DAILY_RETURNS  4.327318
# 2    IXIC_DAILY_RETURNS  3.917106
# 3     HSI_DAILY_RETURNS  1.360934
# 4    N225_DAILY_RETURNS  1.340041
# 5   GDAXI_DAILY_RETURNS  1.835791
# 6     VIX_DAILY_RETURNS  2.059399
# 7         NSEI_HL_RATIO  1.640911
# 8          DJI_HL_RATIO  1.920980
# 9              NSEI_RSI  1.440536
# 10              DJI_RSI  1.532050



# %% 6 -
data['predicted'] = model.predict(data)



# %% 7 - ROC Curve
fpr, tpr, thresholds = roc_curve(data['NSEI_OPEN_DIR'], data['predicted'])

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "01_01", "01 - with all data", "phase_03")



# %% 8 - find optimal threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold is : {optimal_threshold}')
# Best Threshold is : 0.685



# %% 9 - AUC Curve
auc_roc = roc_auc_score(data['NSEI_OPEN_DIR'], data['predicted'])
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.762826243857053



# %% 10 - Classification Report
data['predicted_class'] = np.where(data['predicted'] <= optimal_threshold,  0, 1)
print(classification_report(data['NSEI_OPEN_DIR'], data['predicted_class']))
#               precision    recall  f1-score   support
# 
#            0       0.56      0.68      0.61       497
#            1       0.83      0.75      0.79      1051
# 
#     accuracy                           0.73      1548
#    macro avg       0.70      0.71      0.70      1548
# weighted avg       0.74      0.73      0.73      1548



# %% 11 - 
table = pd.crosstab(data['predicted_class'], data['NSEI_OPEN_DIR'])
table
# NSEI_OPEN_DIR      0    1
# predicted_class          
# 0                338  266
# 1                159  785



# %% 12 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.685 is : 74.69%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.685 is : 68.01%
