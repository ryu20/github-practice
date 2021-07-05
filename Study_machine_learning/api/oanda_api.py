from oandapyV20 import API
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import configparser
import sqlite3


# 日時による取得
def fetch_data_by_date(params, instr='USD_JPY'):
    data = []
    # APIに渡すパラメーターの設定
    # APIへ過去為替レートをリクエスト
    for r in InstrumentsCandlesFactory(instrument=instr, params=params):
        api.request(r)
        for raw in r.response['candles']:
            data.append(
                [raw['time'], raw['volume'], raw['mid']['o'], raw['mid']['h'], raw['mid']['l'], raw['mid']['c']])

    api_df = pd.DataFrame(data)
    api_df.columns = ['Time', 'Volume', 'Open', 'High', 'Low', 'Close']
    api_df['Time'] = pd.to_datetime(api_df['Time'], format='%Y%m%d %H:%M:%S')

    save_data(api_df)

    return


def save_data(ins_df):
    # DBに接続する。
    conn = sqlite3.connect('../db/sqlite3s/oanda_api.sqlite3')
    # カーソルを取得する
    c = conn.cursor()
    ins_df.to_sql(u'candles', conn, if_exists='append', index=None)


def get_latest_time():
    # DBに接続する。
    conn = sqlite3.connect('../db/sqlite3s/oanda_api.sqlite3')
    # カーソルを取得する
    c = conn.cursor()
    # 1. カーソルをイテレータ (iterator) として扱う
    c.execute('select TIME from candles order by TIME DESC LIMIT 0, 1')

    if c.fetchone()[0] is not None:
        return datetime.strptime(c.fetchone()[0], '%Y-%m-%d %H:%M:%S+00:00') + timedelta(minutes=5)
    else:
        today = datetime.today()
        return today - relativedelta(years=1)


config = configparser.ConfigParser()
config.read('../../config/oanda_conf.ini')
access_token = config['LIVE']['TOKEN']
api = API(access_token=access_token, environment="live")

iso_format = '%Y-%m-%dT%H:%M:%SZ'

time_from = get_latest_time()
api_from = time_from.strftime(iso_format)
time_to = datetime.now()
api_to = time_to.strftime(iso_format)

gran = 'M5'
params = {
    'from': api_from,
    'to': api_to,
    'granularity': gran
}
fetch_data_by_date(params)
