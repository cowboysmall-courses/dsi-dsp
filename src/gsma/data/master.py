
import pandas as pd



def merge_data(data, start_date = '2018-01-02', end_date = '2023-12-29'):
    merged = pd.concat(data, axis = 1)

    merged.fillna(method = 'ffill', inplace = True)

    merged['MONTH']   = pd.PeriodIndex(merged.index, freq = 'M')
    merged['QUARTER'] = pd.PeriodIndex(merged.index, freq = 'Q')
    merged['YEAR']    = pd.PeriodIndex(merged.index, freq = 'Y')

    return merged[start_date:end_date]
