
import pandas as pd
import yfinance as yf


def retrieve_data(index, start_date = '2017-12-1', end_date = '2024-1-31'):
    data = yf.download(f'^{index}', start_date, end_date)
    data['Daily Returns'] = data.Close.pct_change() * 100
    return data


def rename_columns(data, index):
    data.columns = [f'{index}_{'_'.join(c.upper() for c in column.split())}' for column in data.columns]
    return data


def read_data(index):
    data = pd.read_csv("./data/raw/{}.csv".format(index), index_col = 'Date')
    data.index = pd.to_datetime(data.index)
    return data


def read_master():
    data = pd.read_csv("./data/processed/master_data.csv", index_col = 'Date')
    data.index = pd.to_datetime(data.index)
    data['MONTH']   = pd.PeriodIndex(data['MONTH'],   freq = 'M')
    data['QUARTER'] = pd.PeriodIndex(data['QUARTER'], freq = 'Q')
    data['YEAR']    = pd.PeriodIndex(data['YEAR'],    freq = 'Y')
    return data


def merge_data(indices, start_date = '2018-01-02', end_date = '2023-12-29'):
    merged = pd.concat(indices, axis = 1)
    merged.fillna(method = 'ffill', inplace = True)
    return merged[start_date:end_date]


def add_indicators(data):
    data['MONTH']   = pd.PeriodIndex(data.index, freq = 'M')
    data['QUARTER'] = pd.PeriodIndex(data.index, freq = 'Q')
    data['YEAR']    = pd.PeriodIndex(data.index, freq = 'Y')
    return data


def save_data(data, index):
    data.to_csv(f'./data/raw/{index}.csv')
    return data
