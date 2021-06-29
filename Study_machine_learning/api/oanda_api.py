from oandapyV20 import API
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import configparser

config = configparser.ConfigParser()
config.read('../../config/oanda_conf.ini')

access_token = config['LIVE']['TOKEN']

api = API(access_token=access_token, environment="live")

today = datetime.today()
one_years_ago = today - relativedelta(years=1)
one_month_ago = today - relativedelta(months=1)

iso_format = '%Y-%m-%dT%H:%M:%SZ'
api_from = one_years_ago.strftime(iso_format)
api_to = today.strftime(iso_format)

gran = 'M5'
params = {
    'from': api_from,
    'to': api_to,
    'granularity': gran
}


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
    api_df = api_df.set_index('Time')
    api_df.index = pd.to_datetime(api_df.index, format='%Y%m%d %H:%M:%S')

    today = datetime.today().strftime("%m%d")
    f_name = f'../data/{instr}-{gran}-{today}-{len(api_df)}.csv'

    api_df.to_csv(f_name)
    return


fetch_data_by_date(params)
