from oandapyV20 import API
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform


def prepare_data():
    window_len = 10
    train_rate = 0.8

    f_name = 'USD_JPY-M5-0621-74286.csv'
    df = pd.read_csv(f_name, encoding="utf_8", index_col=0)

    x_df = df[['Open', 'High', 'Low', 'Close']]
    y_df = df[['Close']]

    mms = MinMaxScaler()
    x_df = mms.fit_transform(x_df)
    y_df = np.array(y_df)

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


def create_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(LSTM({{choice([256, 512])}}, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(1))
    model.add(Activation("linear"))

    model.compile(loss="mae", optimizer="adam")
    # early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=0)
    model.fit(x_train, y_train,
              batch_size=100,
              epochs=10,
              verbose=1,
              validation_data=(x_test, y_test))

    val_loss, val_acc = model.evaluate(x_test, y_test, verbose=0)
    return {'loss': -val_acc, 'status': STATUS_OK, 'model': model}


if __name__ == "__main__":
    best_run, best_model = optim.minimize(model=create_model,
                                          data=prepare_data,
                                          algo=tpe.suggest,
                                          max_evals=6,
                                          trials=Trials())

    print(best_model.summary())
    print(best_run)

    _, _, x_test, y_test = prepare_data()
    val_loss, val_acc = best_model.evaluate(x_test, y_test)
    print("val_loss: ", val_loss)
    print("val_acc: ", val_acc)
