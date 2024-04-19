
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
from statsmodels.api import Logit
from sklearn.linear_model import LogisticRegression
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


X = data.drop(["NSEI_OPEN_DIR"], axis = 1)
y = data['NSEI_OPEN_DIR']



# %% 6 -
model = Logit(y, X).fit()
# model.pvalues.index
insignificant = [p for p in zip(model.pvalues.index, model.pvalues) if p[1] > 0.05]
insignificant
# model = LogisticRegression().fit(X, y)





# %% 7 -
model.get_params()




