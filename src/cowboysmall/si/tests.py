
import statsmodels.api as sm

from scipy import stats


def test_normality(data, column_name, index_name):
    print()
    print(f"\t Index {index_name}")
    print(f"\tColumn {column_name}")


    result = stats.shapiro(data[column_name].dropna())
    print("\t     Shapiro-Wilks Test:")
    if result[1] < 0.05:
        print(f"\t        reject null hypothesis: p-value = {result[1]}")
    else:
        print(f"\tfail to reject null hypothesis: p-value = {result[1]}")


    result = sm.stats.diagnostic.lilliefors(data[column_name].dropna())
    print("\tKolmogorov-Smirnov Test:")
    if result[1] < 0.05:
        print(f"\t        reject null hypothesis: p-value = {result[1]}")
    else:
        print(f"\tfail to reject null hypothesis: p-value = {result[1]}")


    print()
