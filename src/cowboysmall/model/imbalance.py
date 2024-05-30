
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score


def imbalance_remedy_evaluation(remedy, model, X, y):
    X_os, y_os = remedy.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_os, y_os, test_size = 0.2, random_state = 1337)

    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1])

    results = {}
    results['AUC'] = roc_auc_score(y_test, y_pred[:, 1])
    results['THRESHOLD'] = round(thresholds[np.argmax(tpr - fpr)], 3)

    y_class = np.where(y_pred[:, 1] <= results['THRESHOLD'], 0, 1)
    results['ACCURACY'] = accuracy_score(y_test, y_class)

    table = pd.crosstab(y_class, y_test)
    results['SENSITIVITY'] = round((table.iloc[1, 1] / (table.iloc[0, 1] + table.iloc[1, 1])) * 100, 2)
    results['SPECIFICITY'] = round((table.iloc[0, 0] / (table.iloc[0, 0] + table.iloc[1, 0])) * 100, 2)

    return results
