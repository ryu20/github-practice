from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from getdata import prepare_data, get_data
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def build_model(x_train, y_train):
    model = Sequential()
    model.add(LSTM({{choice([64, 128, 256, 512])}}, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(1))
    model.add(Activation("linear"))

    model.compile(loss="mean_squared_error", optimizer={{choice(['sgd', 'adam'])}}, metrics='mae')

    early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=0)
    model.fit(x_train, y_train,
              batch_size=200,
              epochs=10,
              verbose=1,
              validation_split=0.2,
              callbacks=[early_stopping]
              )
    val_loss, val_acc = model.evaluate(x_test, y_test, verbose=0)
    return {'loss': val_acc, 'status': STATUS_OK, 'model': model}


if __name__ == "__main__":
    best_run, best_model = optim.minimize(model=build_model,
                                          data=prepare_data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials(),
                                          eval_space=True
                                          )

    _, _, x_test, y_test = prepare_data()

    print(best_model.summary())
    print("Evalutation of best performing model:")
    print(best_model.evaluate(x_test, y_test))
    print('--- sorted ---')
    sorted_best_run = sorted(best_run.items(), key=lambda x: x[0])
    for i, k in sorted_best_run:
        print(i + ' : ' + str(k))
