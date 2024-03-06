
import statsmodels.api as sm

from scipy import stats


def test_normality(index, data):
    column = '{}_DAILY_RETURNS'.format(index)

    print()
    print('\t Index {}'.format(index))
    print('\tColumn {}'.format(column))

    result = stats.shapiro(data[column].dropna())
    if result[1] < 0.05:
        print('\t     Shapiro-Wilks Test:         reject null hypothesis - with p-value = {}'.format(result[1]))
    else:
        print('\t     Shapiro-Wilks Test: fail to reject null hypothesis - with p-value = {}'.format(result[1]))

    result = sm.stats.diagnostic.lilliefors(data[column].dropna())
    if result[1] < 0.05:
        print('\tKolmogorov-Smirnov Test:         reject null hypothesis - with p-value = {}'.format(result[1]))
    else:
        print('\tKolmogorov-Smirnov Test: fail to reject null hypothesis - with p-value = {}'.format(result[1]))

    print()
