
import pandas as pd



def merge_data(data, start_date = '2018-01-02', end_date = '2023-12-29'):
    merged = pd.concat(data, axis = 1)

    merged.fillna(method = 'ffill', inplace = True)

    merged['MONTH']   = merged.index.month
    merged['QUARTER'] = merged.index.quarter
    merged['YEAR']    = merged.index.year

    return merged[start_date:end_date]
