
from statsmodels.api import Logit


def pruned_logit(X, y):
    dropped = []
    while True:
        model = Logit(y, X).fit()
        insignificant = [p for p in zip(model.pvalues.index, model.pvalues) if p[1] > 0.05]
        insignificant.sort(key = lambda p: -p[1])
        if insignificant:
            print(f"dropping {insignificant[0][0]}")
            X = X.drop([insignificant[0][0]], axis = 1)
            dropped.append(insignificant[0][0])
        else:
            return model, dropped
