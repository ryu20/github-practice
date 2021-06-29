import numpy as np

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical


def prepare_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense({{choice([256, 512])}}, input_shape=(784,)))
    model.add(Dense({{choice([16, 32, 64])}}))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=100,
              epochs=10,
              verbose=1,
              validation_data=(x_test, y_test))

    val_loss, val_acc = model.evaluate(x_test, y_test, verbose=0)
    return {'loss': -val_acc, 'status': STATUS_OK, 'model': model}


def create_model2(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense({{choice([256, 512])}}, input_shape=(784,)))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([16, 32, 64])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=100,
              epochs=10,
              verbose=1,
              validation_data=(x_test, y_test))

    val_loss, val_acc = model.evaluate(x_test, y_test, verbose=0)
    return {'loss': -val_acc, 'status': STATUS_OK, 'model': model}


if __name__ == "__main__":

    best_run, best_model = optim.minimize(model=create_model2,
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
