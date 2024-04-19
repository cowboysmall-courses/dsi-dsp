
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
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

from statsmodels.formula.api import logit
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

from gsma import COLUMNS
from gsma.data.file import read_master_file
# from gsma.plots import plt, sns


# %% 2 -
def calculate_rsi(values, window = 14):
    delta_up = values.diff()
    delta_dn = delta_up.copy()

    delta_up[delta_up < 0] = 0
    delta_dn[delta_dn > 0] = 0

    mean_up = delta_up.rolling(window).mean()
    mean_dn = delta_dn.rolling(window).mean().abs()

    return (mean_up / (mean_up + mean_dn)) * 100



# %% 3 -
master = read_master_file()

master["NSEI_OPEN_DIR"] = np.where(master["NSEI_OPEN"] > master["NSEI_CLOSE"].shift(), 1, 0)

master["NSEI_HL_RATIO"] = master["NSEI_HIGH"] / master["NSEI_LOW"]
master["DJI_HL_RATIO"]  = master["DJI_HIGH"] / master["DJI_LOW"]

master["NSEI_RSI"]      = calculate_rsi(master["NSEI_CLOSE"])
master["DJI_RSI"]       = calculate_rsi(master["DJI_CLOSE"])



# %% 4 -
data = pd.concat([master["NSEI_OPEN_DIR"], master[COLUMNS].shift(), master["NSEI_HL_RATIO"].shift(), master["DJI_HL_RATIO"].shift(), master["NSEI_RSI"].shift(), master["DJI_RSI"].shift()], axis = 1)
data.dropna(inplace = True)
data.head()



# %% 5 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + DJI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + N225_DAILY_RETURNS + GDAXI_DAILY_RETURNS + VIX_DAILY_RETURNS + NSEI_HL_RATIO + DJI_HL_RATIO + NSEI_RSI + DJI_RSI', data = data).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1548
# Model:                          Logit   Df Residuals:                     1536
# Method:                           MLE   Df Model:                           11
# Date:                Fri, 19 Apr 2024   Pseudo R-squ.:                  0.1460
# Time:                        11:59:55   Log-Likelihood:                -829.77
# converged:                       True   LL-Null:                       -971.63
# Covariance Type:            nonrobust   LLR p-value:                 2.343e-54
# =======================================================================================
#                           coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# Intercept              26.7722     11.188      2.393      0.017       4.843      48.701
# NSEI_DAILY_RETURNS     -0.2618      0.072     -3.632      0.000      -0.403      -0.121
# DJI_DAILY_RETURNS       0.1380      0.113      1.219      0.223      -0.084       0.360
# IXIC_DAILY_RETURNS      0.3996      0.081      4.924      0.000       0.241       0.559
# HSI_DAILY_RETURNS      -0.0937      0.049     -1.911      0.056      -0.190       0.002
# N225_DAILY_RETURNS     -0.0926      0.062     -1.497      0.134      -0.214       0.029
# GDAXI_DAILY_RETURNS     0.0436      0.068      0.639      0.523      -0.090       0.177
# VIX_DAILY_RETURNS      -0.0450      0.012     -3.710      0.000      -0.069      -0.021
# NSEI_HL_RATIO          -7.3651      9.574     -0.769      0.442     -26.130      11.400
# DJI_HL_RATIO          -18.3269     10.488     -1.747      0.081     -38.882       2.228
# NSEI_RSI                0.0018      0.004      0.449      0.653      -0.006       0.009
# DJI_RSI                -0.0003      0.004     -0.071      0.943      -0.009       0.008
# =======================================================================================
# """




# DJI_DAILY_RETURNS       0.1380      0.113      1.219      0.223      -0.084       0.360
# HSI_DAILY_RETURNS      -0.0937      0.049     -1.911      0.056      -0.190       0.002
# N225_DAILY_RETURNS     -0.0926      0.062     -1.497      0.134      -0.214       0.029
# GDAXI_DAILY_RETURNS     0.0436      0.068      0.639      0.523      -0.090       0.177
# NSEI_HL_RATIO          -7.3651      9.574     -0.769      0.442     -26.130      11.400
# NSEI_RSI                0.0018      0.004      0.449      0.653      -0.006       0.009
# DJI_RSI                -0.0003      0.004     -0.071      0.943      -0.009       0.008



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
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




# %% 7 -
data['predicted'] = model.predict(data)



# %% 8 - ROC Curve
# plt.plot_setup()
# sns.sns_setup()

fpr, tpr, thresholds = roc_curve(data['NSEI_OPEN_DIR'], data['predicted'])

plt.figure(figsize = (8, 6))

plt.plot(fpr, tpr, label = 'ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label = 'Random Guess')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc = 'lower right')

plt.show()



# %% 9 - AUC Curve
auc_roc = roc_auc_score(data['NSEI_OPEN_DIR'], data['predicted'])
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.762826243857053


# %% 10 - find optimal threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold is : {optimal_threshold}')
# Best Threshold is : 0.685



# %% 11 - Classification Report
data['predicted_class'] = (data['predicted'] > optimal_threshold).astype(int)
print(classification_report(data['NSEI_OPEN_DIR'], data['predicted_class']))
#               precision    recall  f1-score   support
# 
#            0       0.56      0.68      0.61       497
#            1       0.83      0.75      0.79      1051
# 
#     accuracy                           0.73      1548
#    macro avg       0.70      0.71      0.70      1548
# weighted avg       0.74      0.73      0.73      1548



# %% 12 - 
table = pd.crosstab(data['predicted_class'], data['NSEI_OPEN_DIR'])
table
# NSEI_OPEN_DIR      0    1
# predicted_class          
# 0                338  266
# 1                159  785



# %% 13 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.685 is : 74.69%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.685 is : 68.01%
