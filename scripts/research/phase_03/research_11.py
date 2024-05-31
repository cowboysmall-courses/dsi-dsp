
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

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler

from cowboysmall.data.file import read_master_file
from cowboysmall.model.logit import pruned_logit
from cowboysmall.feature.indicators import get_indicators, INDICATORS
from cowboysmall.plots import plt, sns



# %% 2 -
INDICES  = ['NSEI', 'DJI', 'IXIC', 'HSI', 'N225', 'GDAXI', 'VIX']
COLUMNS  = [f"{index}_DAILY_RETURNS" for index in INDICES]
RATIOS   = ["NSEI_HL_RATIO", "DJI_HL_RATIO"]

ALL_COLS = COLUMNS + RATIOS + INDICATORS


# %% 2 -
master = read_master_file()



# %% 2 -
master["NSEI_OPEN_DIR"] = np.where(master["NSEI_OPEN"] > master["NSEI_CLOSE"].shift(), 1, 0)



# %% 2 -
master["NSEI_HL_RATIO"] = master["NSEI_HIGH"] / master["NSEI_LOW"]
master["DJI_HL_RATIO"]  = master["DJI_HIGH"] / master["DJI_LOW"]



# %% 2 -
master = get_indicators(master)




# %% 2 -
counts = master['NSEI_OPEN_DIR'].value_counts().reset_index()
counts.columns = ['NSEI_OPEN_DIR', 'Freq']
print(counts)
#    NSEI_OPEN_DIR  Freq
# 0              1  1064
# 1              0   499



# %% 2 -
print((counts["Freq"][0] / (counts["Freq"][0] + counts["Freq"][1])).round(3))
# 0.681



# %% 3 -
data = pd.concat([master["NSEI_OPEN_DIR"].shift(-1), master[ALL_COLS]], axis = 1)
data.dropna(inplace = True)
data.head()



# %% 3 -
X = data[ALL_COLS]
y = data['NSEI_OPEN_DIR']



# %% 3 -
X.insert(loc = 0, column = "Intercept", value = 1)


# %% 3 -
remedy = RandomUnderSampler(random_state = 0)
X, y = remedy.fit_resample(X, y)


# %% 4 -
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1337)



# %% 5 -
model, dropped = pruned_logit(X_train, y_train)
# dropping DJI_AWE with p-value 0.9508950238176997
# dropping NSEI_KAM with p-value 0.9092895302946098
# dropping GDAXI_DAILY_RETURNS with p-value 0.9086505313792627
# dropping NSEI_VPT with p-value 0.8381519905776433
# dropping DJI_KAM with p-value 0.8310405408192362
# dropping NSEI_AWE with p-value 0.7994181747246288
# dropping DJI_DAILY_RETURNS with p-value 0.7103674440670062
# dropping DJI_ULC with p-value 0.45932472376766087
# dropping NSEI_ULC with p-value 0.6088740491222955
# dropping N225_DAILY_RETURNS with p-value 0.2961948000860465
# dropping DJI_ROC with p-value 0.2666053030395842
# dropping NSEI_ROC with p-value 0.37321356589805954
# dropping NSEI_EMA with p-value 0.11390774785109663
# dropping DJI_VPT with p-value 0.08170541426719269
# dropping DJI_HL_RATIO with p-value 0.16695089866822344
# dropping NSEI_SMA with p-value 0.12110210204699845
# dropping NSEI_RSI with p-value 0.06299735530932622
# dropping DJI_SMA with vif 1582.7040130235273
# dropping DJI_EMA with p-value 0.0638474632480096
# dropping NSEI_TSI with p-value 0.0559010907046787
# dropping DJI_TSI with p-value 0.05370899161923685
# dropping DJI_RSI with p-value 0.08817670282144818



# %% 5 -
model.summary()
# """
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:          NSEI_OPEN_DIR   No. Observations:                  780
# Model:                          Logit   Df Residuals:                      774
# Method:                           MLE   Df Model:                            5
# Date:                Fri, 31 May 2024   Pseudo R-squ.:                  0.1552
# Time:                        21:19:37   Log-Likelihood:                -456.48
# converged:                       True   LL-Null:                       -540.34
# Covariance Type:            nonrobust   LLR p-value:                 2.226e-34
# ======================================================================================
#                          coef    std err          z      P>|z|      [0.025      0.975]
# --------------------------------------------------------------------------------------
# Intercept             37.8987      9.578      3.957      0.000      19.127      56.671
# NSEI_DAILY_RETURNS    -0.2820      0.092     -3.059      0.002      -0.463      -0.101
# IXIC_DAILY_RETURNS     0.4529      0.089      5.090      0.000       0.278       0.627
# HSI_DAILY_RETURNS     -0.1233      0.062     -1.980      0.048      -0.245      -0.001
# VIX_DAILY_RETURNS     -0.0744      0.017     -4.481      0.000      -0.107      -0.042
# NSEI_HL_RATIO        -37.3012      9.468     -3.940      0.000     -55.857     -18.745
# ======================================================================================
# """



# %% 10 -
vif_data = pd.DataFrame()
vif_data["Feature"] = model.model.exog_names[1:]
vif_data["VIF"]     = [variance_inflation_factor(model.model.exog, i) for i in range(1, model.model.exog.shape[1])]
vif_data
#               Feature       VIF
# 0  NSEI_DAILY_RETURNS  1.242093
# 1  IXIC_DAILY_RETURNS  2.070864
# 2   HSI_DAILY_RETURNS  1.220971
# 3   VIX_DAILY_RETURNS  2.004488
# 4       NSEI_HL_RATIO  1.047967



# %% 8 - ROC Curve
y_pred = model.predict(X_train.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y_train, y_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "11_01", "01 - training data", "phase_03")



# %% 9 - Optimal Threshold
optimal_threshold = round(thresholds[np.argmax(tpr - fpr)], 3)
print(f'Best Threshold: {optimal_threshold}')
# Best Threshold: 0.517



# %% 10 - AUC Curve
auc_roc = roc_auc_score(y_train, y_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7598352403950546



# %% 11 - Classification Report
y_pred_class = np.where(y_pred <= optimal_threshold,  0, 1)
print(classification_report(y_train, y_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.71      0.66      0.69       379
#          1.0       0.70      0.74      0.72       401
# 
#     accuracy                           0.71       780
#    macro avg       0.71      0.70      0.70       780
# weighted avg       0.71      0.71      0.70       780



# %% 11 - Confusion Matrix
table = pd.crosstab(y_pred_class, y_train)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0              252  103
# 1              127  298



# %% 11 - Sensitivity / Specificity
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity: {sensitivity}%")
# Sensitivity: 74.31%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity: {specificity}%")
# Specificity: 66.49%



# %% 12 - ROC Curve
y_test_pred = model.predict(X_test.drop(dropped, axis = 1))

fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)

plt.plot_setup()
sns.sns_setup()
plt.roc_curve(fpr, tpr, "11_02", "02 - test data", "phase_03")



# %% 13 - AUC Curve
auc_roc = roc_auc_score(y_test, y_test_pred)
print(f'AUC ROC: {auc_roc}')
# AUC ROC: 0.7489191184224402



# %% 14 - Classification Report
y_test_pred_class = np.where(y_test_pred <= optimal_threshold,  0, 1)
print(classification_report(y_test, y_test_pred_class))
#               precision    recall  f1-score   support
# 
#          0.0       0.72      0.68      0.70       109
#          1.0       0.62      0.67      0.64        87
# 
#     accuracy                           0.67       196
#    macro avg       0.67      0.67      0.67       196
# weighted avg       0.68      0.67      0.67       196



# %% 11 - Confusion Matrix
table = pd.crosstab(y_test_pred_class, y_test)
table
# NSEI_OPEN_DIR  0.0  1.0
# row_0                  
# 0               74   29
# 1               35   58



# %% 11 - Sensitivity / Specificity
sensitivity = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
print(f"Sensitivity: {sensitivity}%")
# Sensitivity: 66.67%

specificity = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)
print(f"Specificity: {specificity}%")
# Specificity: 67.89%
