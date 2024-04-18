
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
    delta      = values.diff()

    delta_up   = delta.copy()
    delta_down = delta.copy()

    delta_up[delta_up < 0] = 0
    delta_down[delta_down > 0] = 0

    mean_up   = delta_up.rolling(window).mean()
    mean_down = delta_down.rolling(window).mean().abs()

    return (mean_up / (mean_up + mean_down)) * 100



# %% 2 -
master = read_master_file()

master["NSEI_OPEN_DIR"] = np.where(master["NSEI_OPEN"] > master["NSEI_CLOSE"].shift(), 1, 0)
master["NSEI_HL_RATIO"] = (master["NSEI_HIGH"] / master["NSEI_LOW"])
master["DJI_HL_RATIO"]  = (master["DJI_HIGH"] / master["DJI_LOW"])
master["NSEI_RSI"]      = calculate_rsi(master["NSEI_CLOSE"])
master["DJI_RSI"]       = calculate_rsi(master["DJI_CLOSE"])



# %% 3 -
data = pd.concat([master["NSEI_OPEN_DIR"], master[COLUMNS].shift(), master["NSEI_HL_RATIO"].shift(), master["DJI_HL_RATIO"].shift(), master["NSEI_RSI"].shift(), master["DJI_RSI"].shift()], axis = 1)
data.dropna(inplace = True)
data.head()



# %% 4 -
train, test = train_test_split(data, test_size = 0.2)



# %% 5 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + DJI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + N225_DAILY_RETURNS + GDAXI_DAILY_RETURNS + VIX_DAILY_RETURNS + NSEI_HL_RATIO + DJI_HL_RATIO + NSEI_RSI + DJI_RSI', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1249
# Model:                          Logit   Df Residuals:                     1239
# Method:                           MLE   Df Model:                            9
# Date:                Wed, 17 Apr 2024   Pseudo R-squ.:                  0.1535
# Time:                        11:27:29   Log-Likelihood:                -661.71
# converged:                       True   LL-Null:                       -781.69
# Covariance Type:            nonrobust   LLR p-value:                 1.311e-46
# =======================================================================================
#                           coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# Intercept              40.7749     10.645      3.831      0.000      19.912      61.638
# NSEI_DAILY_RETURNS     -0.2539      0.076     -3.345      0.001      -0.403      -0.105
# DJI_DAILY_RETURNS       0.0682      0.120      0.571      0.568      -0.166       0.302
# IXIC_DAILY_RETURNS      0.4097      0.091      4.486      0.000       0.231       0.589
# HSI_DAILY_RETURNS      -0.1343      0.055     -2.438      0.015      -0.242      -0.026
# N225_DAILY_RETURNS     -0.0825      0.066     -1.240      0.215      -0.213       0.048
# GDAXI_DAILY_RETURNS     0.0575      0.075      0.763      0.446      -0.090       0.205
# VIX_DAILY_RETURNS      -0.0516      0.013     -3.882      0.000      -0.078      -0.026
# NSEI_HL_RATIO         -14.3712      9.634     -1.492      0.136     -33.254       4.511
# DJI_HL_RATIO          -25.0754     10.594     -2.367      0.018     -45.839      -4.312
# =======================================================================================
# """



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                Feature       VIF
# 0   NSEI_DAILY_RETURNS  1.375919
# 1    DJI_DAILY_RETURNS  4.351087
# 2   IXIC_DAILY_RETURNS  4.103444
# 3    HSI_DAILY_RETURNS  1.346927
# 4   N225_DAILY_RETURNS  1.340750
# 5  GDAXI_DAILY_RETURNS  1.891221
# 6    VIX_DAILY_RETURNS  2.112126
# 7        NSEI_HL_RATIO  1.530588
# 8         DJI_HL_RATIO  1.593921



# %% 7 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + N225_DAILY_RETURNS + GDAXI_DAILY_RETURNS + VIX_DAILY_RETURNS + NSEI_HL_RATIO + DJI_HL_RATIO', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1249
# Model:                          Logit   Df Residuals:                     1240
# Method:                           MLE   Df Model:                            8
# Date:                Wed, 17 Apr 2024   Pseudo R-squ.:                  0.1533
# Time:                        11:29:28   Log-Likelihood:                -661.87
# converged:                       True   LL-Null:                       -781.69
# Covariance Type:            nonrobust   LLR p-value:                 2.711e-47
# =======================================================================================
#                           coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# Intercept              40.0827     10.309      3.888      0.000      19.878      60.287
# NSEI_DAILY_RETURNS     -0.2518      0.076     -3.320      0.001      -0.400      -0.103
# IXIC_DAILY_RETURNS      0.4403      0.074      5.937      0.000       0.295       0.586
# HSI_DAILY_RETURNS      -0.1375      0.055     -2.510      0.012      -0.245      -0.030
# N225_DAILY_RETURNS     -0.0788      0.066     -1.194      0.232      -0.208       0.051
# GDAXI_DAILY_RETURNS     0.0711      0.071      0.996      0.319      -0.069       0.211
# VIX_DAILY_RETURNS      -0.0534      0.013     -4.155      0.000      -0.079      -0.028
# NSEI_HL_RATIO         -13.6743      9.420     -1.452      0.147     -32.138       4.789
# DJI_HL_RATIO          -25.0859     10.497     -2.390      0.017     -45.659      -4.512
# =======================================================================================
# """



# %% 8 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                Feature       VIF
# 0   NSEI_DAILY_RETURNS  1.369319
# 1   IXIC_DAILY_RETURNS  2.263584
# 2    HSI_DAILY_RETURNS  1.332201
# 3   N225_DAILY_RETURNS  1.332324
# 4  GDAXI_DAILY_RETURNS  1.717464
# 5    VIX_DAILY_RETURNS  2.049962
# 6        NSEI_HL_RATIO  1.524168
# 7         DJI_HL_RATIO  1.592250



# %% 9 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + N225_DAILY_RETURNS + VIX_DAILY_RETURNS + NSEI_HL_RATIO + DJI_HL_RATIO', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1249
# Model:                          Logit   Df Residuals:                     1241
# Method:                           MLE   Df Model:                            7
# Date:                Wed, 17 Apr 2024   Pseudo R-squ.:                  0.1526
# Time:                        11:30:57   Log-Likelihood:                -662.37
# converged:                       True   LL-Null:                       -781.69
# Covariance Type:            nonrobust   LLR p-value:                 7.268e-48
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             39.8380     10.265      3.881      0.000      19.718      59.958
# NSEI_DAILY_RETURNS    -0.2345      0.073     -3.192      0.001      -0.378      -0.090
# IXIC_DAILY_RETURNS     0.4570      0.072      6.341      0.000       0.316       0.598
# HSI_DAILY_RETURNS     -0.1338      0.055     -2.453      0.014      -0.241      -0.027
# N225_DAILY_RETURNS    -0.0674      0.065     -1.042      0.298      -0.194       0.059
# VIX_DAILY_RETURNS     -0.0552      0.013     -4.358      0.000      -0.080      -0.030
# NSEI_HL_RATIO        -14.2323      9.417     -1.511      0.131     -32.689       4.224
# DJI_HL_RATIO         -24.2871     10.473     -2.319      0.020     -44.814      -3.761
# ======================================================================================
# """



# %% 10 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.268557
# 1  IXIC_DAILY_RETURNS  2.079045
# 2   HSI_DAILY_RETURNS  1.321853
# 3  N225_DAILY_RETURNS  1.291874
# 4   VIX_DAILY_RETURNS  2.007826
# 5       NSEI_HL_RATIO  1.519542
# 6        DJI_HL_RATIO  1.589102



# %% 7 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + VIX_DAILY_RETURNS + NSEI_HL_RATIO + DJI_HL_RATIO', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1249
# Model:                          Logit   Df Residuals:                     1242
# Method:                           MLE   Df Model:                            6
# Date:                Wed, 17 Apr 2024   Pseudo R-squ.:                  0.1519
# Time:                        11:32:38   Log-Likelihood:                -662.92
# converged:                       True   LL-Null:                       -781.69
# Covariance Type:            nonrobust   LLR p-value:                 1.876e-48
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             38.4641     10.013      3.841      0.000      18.839      58.089
# NSEI_DAILY_RETURNS    -0.2513      0.072     -3.480      0.001      -0.393      -0.110
# IXIC_DAILY_RETURNS     0.4534      0.072      6.282      0.000       0.312       0.595
# HSI_DAILY_RETURNS     -0.1519      0.052     -2.942      0.003      -0.253      -0.051
# VIX_DAILY_RETURNS     -0.0555      0.013     -4.377      0.000      -0.080      -0.031
# NSEI_HL_RATIO        -13.8623      9.252     -1.498      0.134     -31.997       4.272
# DJI_HL_RATIO         -23.3000     10.439     -2.232      0.026     -43.760      -2.840
# ======================================================================================
# """



# %% 10 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.233650
# 1  IXIC_DAILY_RETURNS  2.066982
# 2   HSI_DAILY_RETURNS  1.179232
# 3   VIX_DAILY_RETURNS  2.004351
# 4       NSEI_HL_RATIO  1.518209
# 5        DJI_HL_RATIO  1.577404



# %% 7 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + VIX_DAILY_RETURNS + DJI_HL_RATIO', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1249
# Model:                          Logit   Df Residuals:                     1243
# Method:                           MLE   Df Model:                            5
# Date:                Wed, 17 Apr 2024   Pseudo R-squ.:                  0.1506
# Time:                        11:33:51   Log-Likelihood:                -663.97
# converged:                       True   LL-Null:                       -781.69
# Covariance Type:            nonrobust   LLR p-value:                 7.327e-49
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             31.5643      9.412      3.354      0.001      13.118      50.010
# NSEI_DAILY_RETURNS    -0.2495      0.073     -3.421      0.001      -0.392      -0.107
# IXIC_DAILY_RETURNS     0.4389      0.072      6.099      0.000       0.298       0.580
# HSI_DAILY_RETURNS     -0.1480      0.052     -2.874      0.004      -0.249      -0.047
# VIX_DAILY_RETURNS     -0.0558      0.013     -4.409      0.000      -0.081      -0.031
# DJI_HL_RATIO         -30.3397      9.301     -3.262      0.001     -48.569     -12.110
# ======================================================================================
# """



# %% 10 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.229487
# 1  IXIC_DAILY_RETURNS  2.030198
# 2   HSI_DAILY_RETURNS  1.178408
# 3   VIX_DAILY_RETURNS  2.004127
# 4        DJI_HL_RATIO  1.072773




# %% 8 - ROC Curve
train['predicted'] = model.predict(train)

# plt.plot_setup()
# sns.sns_setup()

fpr, tpr, thresholds = roc_curve(train['NSEI_OPEN_DIR'], train['predicted'])

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
auc_roc = roc_auc_score(train['NSEI_OPEN_DIR'], train['predicted'])
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.765442960985893



# %% 10 - Optimal Threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold is : {optimal_threshold}')
# Best Threshold is : 0.654



# %% 11 - Classification Report
train['predicted_class'] = (train['predicted'] > optimal_threshold).astype(int)
print(classification_report(train['NSEI_OPEN_DIR'], train['predicted_class']))
#               precision    recall  f1-score   support
# 
#            0       0.62      0.60      0.61       398
#            1       0.82      0.82      0.82       851
# 
#     accuracy                           0.75      1249
#    macro avg       0.72      0.71      0.71      1249
# weighted avg       0.75      0.75      0.75      1249



# %% 11 - 
table = pd.crosstab(np.where(train['predicted'] <= optimal_threshold,  0, 1), train['NSEI_OPEN_DIR'])
table
# NSEI_OPEN_DIR    0    1
# row_0                  
# 0              239  149
# 1              159  702



# %% 11 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.654 is : 82.49

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.654 is : 60.05



# %% 12 - ROC Curve
test['predicted'] = model.predict(test)

# plt.plot_setup()
# sns.sns_setup()

fpr, tpr, thresholds = roc_curve(test['NSEI_OPEN_DIR'], test['predicted'])

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



# %% 13 - AUC Curve
auc_roc = roc_auc_score(test['NSEI_OPEN_DIR'], test['predicted'])
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7112676056338028



# %% 14 - Classification Report
test['predicted_class'] = (test['predicted'] > optimal_threshold).astype(int)
print(classification_report(test['NSEI_OPEN_DIR'], test['predicted_class']))
#               precision    recall  f1-score   support
# 
#            0       0.54      0.56      0.55       100
#            1       0.79      0.77      0.78       213
# 
#     accuracy                           0.71       313
#    macro avg       0.66      0.67      0.67       313
# weighted avg       0.71      0.71      0.71       313



# %% 11 - 
table = pd.crosstab(np.where(test['predicted'] <= optimal_threshold,  0, 1), test['NSEI_OPEN_DIR'])
table
# NSEI_OPEN_DIR   0    1
# row_0                 
# 0              56   48
# 1              44  165



# %% 11 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.654 is : 77.46

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.654 is : 56.0
