import sqlite3
import numpy as np
import pandas as pd
import configparser
from sklearn.preprocessing import MinMaxScaler


def get_data(limit=100):
    config = configparser.ConfigParser()
    # DBに接続する。
    config.read('config.ini')
    file_pass = config['DATABASE']['OANDA']
    conn = sqlite3.connect(file_pass)

    df = pd.read_sql_query(sql=u"SELECT Time, Volume, Open, High, Low, Close FROM candles\
     ORDER BY Time DESC LIMIT :limit", con=conn, params={"limit": limit})
    # コネクションをクローズ
    conn.close()

    return df[::-1]


def prepare_data(window_len=10, train_rate=0.8, limit=10000):
    if 'window_len' not in locals():
        window_len = 10
        train_rate = 0.8
        limit = 10000

    df = get_data(limit)
    x_df = df[['Open', 'High', 'Low', 'Close']]
    y_df = df[['Close']]

    mms = MinMaxScaler()
    x_df = mms.fit_transform(x_df)
    y_df = mms.fit_transform(y_df)

    x, y = [], []
    for i in range(len(x_df) - window_len):
        x.append(x_df[i:i + window_len])
        y.append(y_df[i + window_len])

    train_size = int(train_rate * (len(x_df) - window_len))
    x_train = np.array(x[:train_size])
    y_train = np.array(y[:train_size])
    x_test = np.array(x[train_size:])
    y_test = np.array(y[train_size:])

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    print(prepare_data(window_len=10, train_rate=0.8, limit=10000))
