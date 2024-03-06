
import pandas as pd



def read_index_file(name, indicators = False):
    data = pd.read_csv(f"./data/raw/{name}.csv", index_col = 'Date')

    data.index = pd.to_datetime(data.index)

    if indicators:
        data['MONTH']   = data.index.month
        data['QUARTER'] = data.index.quarter
        data['YEAR']    = data.index.year

    return data


def save_index_file(data, name):
    data.to_csv(f"./data/raw/{name}.csv")



def read_master_file():
    data = pd.read_csv("./data/processed/master_data.csv", index_col = 'Date')

    data.index = pd.to_datetime(data.index)

    return data


def save_master_file(data):
    data.to_csv("./data/processed/master_data.csv")
