import sqlite3
import pandas as pd
import configparser
import sys
import os
sys.path.append(os.pardir)


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
    return df


if __name__ == '__main__':
    print(get_data())
