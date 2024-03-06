
import pandas as pd



def read_index_file(name, indicators = False):
    data = pd.read_csv(f"./data/raw/{name}.csv", index_col = 'Date')

    data.index = pd.to_datetime(data.index)

    if indicators:
        data['MONTH']   = pd.PeriodIndex(data.index, freq = 'M')
        data['QUARTER'] = pd.PeriodIndex(data.index, freq = 'Q')
        data['YEAR']    = pd.PeriodIndex(data.index, freq = 'Y')

    return data


def save_index_file(data, name):
    data.to_csv(f"./data/raw/{name}.csv")



def read_master_file():
    data = pd.read_csv("./data/processed/master_data.csv", index_col = 'Date')

    data.index = pd.to_datetime(data.index)

    data['MONTH']   = pd.PeriodIndex(data['MONTH'],   freq = 'M')
    data['QUARTER'] = pd.PeriodIndex(data['QUARTER'], freq = 'Q')
    data['YEAR']    = pd.PeriodIndex(data['YEAR'],    freq = 'Y')

    return data


def save_master_file(data):
    data.to_csv("./data/processed/master_data.csv")
