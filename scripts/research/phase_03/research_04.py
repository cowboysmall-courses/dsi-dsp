
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

from statsmodels.formula.api import logit
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

from gsma import COLUMNS
from gsma.data.file import read_master_file
from gsma.feature.indicators import calculate_rsi
from gsma.plots import plt, sns



# %% 2 -
master = read_master_file()

master["NSEI_OPEN_DIR"] = np.where(master["NSEI_OPEN"] > master["NSEI_CLOSE"].shift(), 1, 0)

master["NSEI_HL_RATIO"] = master["NSEI_HIGH"] / master["NSEI_LOW"]
master["DJI_HL_RATIO"]  = master["DJI_HIGH"] / master["DJI_LOW"]

master["NSEI_RSI"]      = calculate_rsi(master["NSEI_CLOSE"])
master["DJI_RSI"]       = calculate_rsi(master["DJI_CLOSE"])

EXTRA_COLS = ["NSEI_HL_RATIO", "DJI_HL_RATIO", "NSEI_RSI", "DJI_RSI"]



# %% 3 -
data = pd.concat([master["NSEI_OPEN_DIR"].shift(-1), master[COLUMNS + EXTRA_COLS]], axis = 1)
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
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1238
# Model:                          Logit   Df Residuals:                     1226
# Method:                           MLE   Df Model:                           11
# Date:                Sat, 20 Apr 2024   Pseudo R-squ.:                  0.1468
# Time:                        14:06:38   Log-Likelihood:                -665.88
# converged:                       True   LL-Null:                       -780.40
# Covariance Type:            nonrobust   LLR p-value:                 6.680e-43
# =======================================================================================
#                           coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# Intercept              29.7373     12.894      2.306      0.021       4.466      55.009
# NSEI_DAILY_RETURNS     -0.2595      0.080     -3.246      0.001      -0.416      -0.103
# DJI_DAILY_RETURNS       0.1806      0.129      1.403      0.161      -0.072       0.433
# IXIC_DAILY_RETURNS      0.4187      0.090      4.647      0.000       0.242       0.595
# HSI_DAILY_RETURNS      -0.1146      0.054     -2.116      0.034      -0.221      -0.008
# N225_DAILY_RETURNS     -0.0336      0.069     -0.486      0.627      -0.169       0.102
# GDAXI_DAILY_RETURNS    -0.0255      0.075     -0.340      0.734      -0.172       0.121
# VIX_DAILY_RETURNS      -0.0415      0.014     -3.074      0.002      -0.068      -0.015
# NSEI_HL_RATIO         -10.8165     11.044     -0.979      0.327     -32.462      10.829
# DJI_HL_RATIO          -17.5415     11.979     -1.464      0.143     -41.019       5.936
# NSEI_RSI               -0.0032      0.004     -0.715      0.475      -0.012       0.006
# DJI_RSI                -0.0002      0.005     -0.037      0.970      -0.010       0.009
# =======================================================================================
# """

# Drop DJI_RSI



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                 Feature       VIF
# 0    NSEI_DAILY_RETURNS  1.473212
# 1     DJI_DAILY_RETURNS  4.361098
# 2    IXIC_DAILY_RETURNS  3.821839
# 3     HSI_DAILY_RETURNS  1.353015
# 4    N225_DAILY_RETURNS  1.319261
# 5   GDAXI_DAILY_RETURNS  1.870333
# 6     VIX_DAILY_RETURNS  2.004505
# 7         NSEI_HL_RATIO  1.711456
# 8          DJI_HL_RATIO  2.036382
# 9              NSEI_RSI  1.439420
# 10              DJI_RSI  1.516552



# %% 7 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + DJI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + N225_DAILY_RETURNS + GDAXI_DAILY_RETURNS + VIX_DAILY_RETURNS + NSEI_HL_RATIO + DJI_HL_RATIO + NSEI_RSI', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1238
# Model:                          Logit   Df Residuals:                     1227
# Method:                           MLE   Df Model:                           10
# Date:                Sat, 20 Apr 2024   Pseudo R-squ.:                  0.1468
# Time:                        14:08:31   Log-Likelihood:                -665.88
# converged:                       True   LL-Null:                       -780.40
# Covariance Type:            nonrobust   LLR p-value:                 1.356e-43
# =======================================================================================
#                           coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------------
# Intercept              29.6261     12.537      2.363      0.018       5.053      54.199
# NSEI_DAILY_RETURNS     -0.2592      0.080     -3.254      0.001      -0.415      -0.103
# DJI_DAILY_RETURNS       0.1798      0.127      1.416      0.157      -0.069       0.429
# IXIC_DAILY_RETURNS      0.4189      0.090      4.654      0.000       0.242       0.595
# HSI_DAILY_RETURNS      -0.1147      0.054     -2.119      0.034      -0.221      -0.009
# N225_DAILY_RETURNS     -0.0337      0.069     -0.487      0.626      -0.169       0.102
# GDAXI_DAILY_RETURNS    -0.0256      0.075     -0.342      0.732      -0.172       0.121
# VIX_DAILY_RETURNS      -0.0416      0.013     -3.078      0.002      -0.068      -0.015
# NSEI_HL_RATIO         -10.8678     10.956     -0.992      0.321     -32.341      10.605
# DJI_HL_RATIO          -17.3865     11.231     -1.548      0.122     -39.398       4.625
# NSEI_RSI               -0.0032      0.004     -0.790      0.430      -0.011       0.005
# =======================================================================================
# """

# Drop GDAXI_DAILY_RETURNS



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#                Feature       VIF
# 0   NSEI_DAILY_RETURNS  1.462241
# 1    DJI_DAILY_RETURNS  4.319660
# 2   IXIC_DAILY_RETURNS  3.815632
# 3    HSI_DAILY_RETURNS  1.350965
# 4   N225_DAILY_RETURNS  1.317453
# 5  GDAXI_DAILY_RETURNS  1.866937
# 6    VIX_DAILY_RETURNS  2.002889
# 7        NSEI_HL_RATIO  1.667773
# 8         DJI_HL_RATIO  1.818337
# 9             NSEI_RSI  1.209332



# %% 7 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + DJI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + N225_DAILY_RETURNS + VIX_DAILY_RETURNS + NSEI_HL_RATIO + DJI_HL_RATIO + NSEI_RSI', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1238
# Model:                          Logit   Df Residuals:                     1228
# Method:                           MLE   Df Model:                            9
# Date:                Sat, 20 Apr 2024   Pseudo R-squ.:                  0.1467
# Time:                        14:10:28   Log-Likelihood:                -665.93
# converged:                       True   LL-Null:                       -780.40
# Covariance Type:            nonrobust   LLR p-value:                 2.755e-44
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             29.4661     12.513      2.355      0.019       4.941      53.991
# NSEI_DAILY_RETURNS    -0.2653      0.078     -3.413      0.001      -0.418      -0.113
# DJI_DAILY_RETURNS      0.1675      0.122      1.376      0.169      -0.071       0.406
# IXIC_DAILY_RETURNS     0.4193      0.090      4.661      0.000       0.243       0.596
# HSI_DAILY_RETURNS     -0.1166      0.054     -2.164      0.030      -0.222      -0.011
# N225_DAILY_RETURNS    -0.0354      0.069     -0.513      0.608      -0.171       0.100
# VIX_DAILY_RETURNS     -0.0411      0.013     -3.058      0.002      -0.067      -0.015
# NSEI_HL_RATIO        -10.5379     10.905     -0.966      0.334     -31.911      10.835
# DJI_HL_RATIO         -17.5594     11.197     -1.568      0.117     -39.505       4.386
# NSEI_RSI              -0.0032      0.004     -0.783      0.434      -0.011       0.005
# ======================================================================================
# """

# Drop N225_DAILY_RETURNS



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.364053
# 1   DJI_DAILY_RETURNS  3.955631
# 2  IXIC_DAILY_RETURNS  3.815354
# 3   HSI_DAILY_RETURNS  1.332406
# 4  N225_DAILY_RETURNS  1.300029
# 5   VIX_DAILY_RETURNS  1.993118
# 6       NSEI_HL_RATIO  1.652285
# 7        DJI_HL_RATIO  1.814822
# 8            NSEI_RSI  1.207621



# %% 7 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + DJI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + VIX_DAILY_RETURNS + NSEI_HL_RATIO + DJI_HL_RATIO + NSEI_RSI', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1238
# Model:                          Logit   Df Residuals:                     1229
# Method:                           MLE   Df Model:                            8
# Date:                Sat, 20 Apr 2024   Pseudo R-squ.:                  0.1465
# Time:                        14:12:05   Log-Likelihood:                -666.07
# converged:                       True   LL-Null:                       -780.40
# Covariance Type:            nonrobust   LLR p-value:                 5.651e-45
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             29.1457     12.344      2.361      0.018       4.952      53.340
# NSEI_DAILY_RETURNS    -0.2733      0.076     -3.578      0.000      -0.423      -0.124
# DJI_DAILY_RETURNS      0.1608      0.121      1.330      0.184      -0.076       0.398
# IXIC_DAILY_RETURNS     0.4213      0.090      4.688      0.000       0.245       0.597
# HSI_DAILY_RETURNS     -0.1253      0.051     -2.449      0.014      -0.226      -0.025
# VIX_DAILY_RETURNS     -0.0413      0.013     -3.070      0.002      -0.068      -0.015
# NSEI_HL_RATIO        -10.5005     10.814     -0.971      0.332     -31.696      10.695
# DJI_HL_RATIO         -17.2818     11.168     -1.547      0.122     -39.170       4.607
# NSEI_RSI              -0.0032      0.004     -0.782      0.434      -0.011       0.005
# ======================================================================================
# """

# Drop NSEI_RSI



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.330029
# 1   DJI_DAILY_RETURNS  3.904254
# 2  IXIC_DAILY_RETURNS  3.802453
# 3   HSI_DAILY_RETURNS  1.190962
# 4   VIX_DAILY_RETURNS  1.992662
# 5       NSEI_HL_RATIO  1.651608
# 6        DJI_HL_RATIO  1.808771
# 7            NSEI_RSI  1.207614


# %% 7 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + DJI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + VIX_DAILY_RETURNS + NSEI_HL_RATIO + DJI_HL_RATIO', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1238
# Model:                          Logit   Df Residuals:                     1230
# Method:                           MLE   Df Model:                            7
# Date:                Sat, 20 Apr 2024   Pseudo R-squ.:                  0.1461
# Time:                        14:13:22   Log-Likelihood:                -666.37
# converged:                       True   LL-Null:                       -780.40
# Covariance Type:            nonrobust   LLR p-value:                 1.281e-45
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             25.8836     11.696      2.213      0.027       2.961      48.807
# NSEI_DAILY_RETURNS    -0.2847      0.075     -3.792      0.000      -0.432      -0.138
# DJI_DAILY_RETURNS      0.1619      0.121      1.338      0.181      -0.075       0.399
# IXIC_DAILY_RETURNS     0.4200      0.090      4.672      0.000       0.244       0.596
# HSI_DAILY_RETURNS     -0.1260      0.051     -2.464      0.014      -0.226      -0.026
# VIX_DAILY_RETURNS     -0.0419      0.013     -3.117      0.002      -0.068      -0.016
# NSEI_HL_RATIO         -8.8957     10.683     -0.833      0.405     -29.835      12.043
# DJI_HL_RATIO         -15.8405     11.023     -1.437      0.151     -37.444       5.763
# ======================================================================================
# """

# Drop NSEI_HL_RATIO



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.282071
# 1   DJI_DAILY_RETURNS  3.902491
# 2  IXIC_DAILY_RETURNS  3.800148
# 3   HSI_DAILY_RETURNS  1.190925
# 4   VIX_DAILY_RETURNS  1.985365
# 5       NSEI_HL_RATIO  1.616615
# 6        DJI_HL_RATIO  1.749106



# %% 7 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + DJI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + VIX_DAILY_RETURNS + DJI_HL_RATIO', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1238
# Model:                          Logit   Df Residuals:                     1231
# Method:                           MLE   Df Model:                            6
# Date:                Sat, 20 Apr 2024   Pseudo R-squ.:                  0.1457
# Time:                        14:14:28   Log-Likelihood:                -666.71
# converged:                       True   LL-Null:                       -780.40
# Covariance Type:            nonrobust   LLR p-value:                 2.778e-46
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             20.8240     10.071      2.068      0.039       1.085      40.563
# NSEI_DAILY_RETURNS    -0.2857      0.075     -3.800      0.000      -0.433      -0.138
# DJI_DAILY_RETURNS      0.1625      0.122      1.336      0.182      -0.076       0.401
# IXIC_DAILY_RETURNS     0.4173      0.090      4.645      0.000       0.241       0.593
# HSI_DAILY_RETURNS     -0.1244      0.051     -2.434      0.015      -0.224      -0.024
# VIX_DAILY_RETURNS     -0.0416      0.013     -3.102      0.002      -0.068      -0.015
# DJI_HL_RATIO         -19.7336      9.958     -1.982      0.048     -39.252      -0.216
# ======================================================================================
# """

# Drop DJI_DAILY_RETURNS



# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.281968
# 1   DJI_DAILY_RETURNS  3.885961
# 2  IXIC_DAILY_RETURNS  3.797006
# 3   HSI_DAILY_RETURNS  1.189345
# 4   VIX_DAILY_RETURNS  1.979783
# 5        DJI_HL_RATIO  1.091161


# %% 7 -
model = logit('NSEI_OPEN_DIR ~ NSEI_DAILY_RETURNS + IXIC_DAILY_RETURNS + HSI_DAILY_RETURNS + VIX_DAILY_RETURNS + DJI_HL_RATIO', data = train).fit()
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                 1238
# Model:                          Logit   Df Residuals:                     1232
# Method:                           MLE   Df Model:                            5
# Date:                Sat, 20 Apr 2024   Pseudo R-squ.:                  0.1445
# Time:                        14:15:48   Log-Likelihood:                -667.63
# converged:                       True   LL-Null:                       -780.40
# Covariance Type:            nonrobust   LLR p-value:                 9.616e-47
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             22.4225      9.636      2.327      0.020       3.537      41.308
# NSEI_DAILY_RETURNS    -0.2686      0.074     -3.648      0.000      -0.413      -0.124
# IXIC_DAILY_RETURNS     0.4871      0.073      6.643      0.000       0.343       0.631
# HSI_DAILY_RETURNS     -0.1266      0.051     -2.486      0.013      -0.226      -0.027
# VIX_DAILY_RETURNS     -0.0475      0.013     -3.762      0.000      -0.072      -0.023
# DJI_HL_RATIO         -21.3146      9.527     -2.237      0.025     -39.986      -2.643
# ======================================================================================
# """


# %% 6 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.222497
# 1  IXIC_DAILY_RETURNS  1.889810
# 2   HSI_DAILY_RETURNS  1.187077
# 3   VIX_DAILY_RETURNS  1.872069
# 4        DJI_HL_RATIO  1.088370



# %% 8 - ROC Curve
train['predicted'] = model.predict(train)

fpr, tpr, thresholds = roc_curve(train['NSEI_OPEN_DIR'], train['predicted'])

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "03_01", "01 - training data", "phase_03")



# %% 9 - Optimal Threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold is : {optimal_threshold}')
# Best Threshold is : 0.639



# %% 10 - AUC Curve
auc_roc = roc_auc_score(train['NSEI_OPEN_DIR'], train['predicted'])
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7481219399202449



# %% 11 - Classification Report
train['predicted_class'] = np.where(train['predicted'] <= optimal_threshold,  0, 1)
print(classification_report(train['NSEI_OPEN_DIR'], train['predicted_class']))
#               precision    recall  f1-score   support
# 
#            0       0.60      0.56      0.58       389
#            1       0.80      0.83      0.82       849
# 
#     accuracy                           0.74      1238
#    macro avg       0.70      0.69      0.70      1238
# weighted avg       0.74      0.74      0.74      1238



# %% 11 - 
table = pd.crosstab(train['predicted_class'], train['NSEI_OPEN_DIR'])
table
# NSEI_OPEN_DIR      0    1
# predicted_class          
# 0                217  144
# 1                172  705



# %% 11 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.639 is : 83.04%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.639 is : 55.78%



# %% 12 - ROC Curve
test['predicted'] = model.predict(test)

fpr, tpr, thresholds = roc_curve(test['NSEI_OPEN_DIR'], test['predicted'])

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "03_02", "02 - testing data", "phase_03")



# %% 13 - AUC Curve
auc_roc = roc_auc_score(test['NSEI_OPEN_DIR'], test['predicted'])
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7873808214154748



# %% 14 - Classification Report
test['predicted_class'] = np.where(test['predicted'] <= optimal_threshold,  0, 1)
print(classification_report(test['NSEI_OPEN_DIR'], test['predicted_class']))
#               precision    recall  f1-score   support
# 
#            0       0.72      0.56      0.63       108
#            1       0.79      0.89      0.83       202
# 
#     accuracy                           0.77       310
#    macro avg       0.76      0.72      0.73       310
# weighted avg       0.77      0.77      0.76       310



# %% 11 - 
table = pd.crosstab(test['predicted_class'], test['NSEI_OPEN_DIR'])
table
# NSEI_OPEN_DIR     0    1
# predicted_class         
# 0                60   23
# 1                48  179



# %% 11 - 
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity for cut-off {optimal_threshold} is : {sensitivity}%")
# Sensitivity for cut-off 0.639 is : 88.61%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity for cut-off {optimal_threshold} is : {specificity}%")
# Specificity for cut-off 0.639 is : 55.56%

# %%
