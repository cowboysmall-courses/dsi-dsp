
from statsmodels.api import Logit
from statsmodels.stats.outliers_influence import variance_inflation_factor


def pruned_logit(X, y):
    dropped = []
    while True:
        model = Logit(y, X).fit()

        insignificant = [p for p in zip(model.pvalues.index, model.pvalues) if p[1] > 0.05]
        insignificant.sort(key = lambda p: -p[1])

        values   = [variance_inflation_factor(model.model.exog, i) for i in range(model.model.exog.shape[1])]
        colinear = [val for val in zip(model.model.exog_names, values) if val[1] > 5]
        colinear.sort(key = lambda c: -c[1])

        if insignificant and insignificant[0][0] == "Intercept":
            insignificant = insignificant[1:]

        if colinear and colinear[0][0] == "Intercept":
            colinear = colinear[1:]

        if insignificant:
            print(f"dropping {insignificant[0][0]}")
            X = X.drop([insignificant[0][0]], axis = 1)
            dropped.append(insignificant[0][0])

        elif colinear:
            print(f"dropping {colinear[0][0]}")
            X = X.drop([colinear[0][0]], axis = 1)
            dropped.append(colinear[0][0])

        else:
            return model, dropped
