from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
from getdata import prepare_data


def build_model(x_train):
    model = Sequential()
    model.add(LSTM(256, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.288662535902546))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer='adam', metrics='mae')
    return model


def train_model():
    x_train, y_train, x_test, y_test = prepare_data()
    model = build_model(x_train)
    history = model.fit(x_train, y_train,
                        batch_size=200,
                        epochs=10,
                        verbose=1,
                        validation_split=0.2,
                        )
    val_loss, val_acc = model.evaluate(x_test, y_test, verbose=0)
    print(model.summary())
    print("val_loss: ", val_loss)
    print("val_acc: ", val_acc)
    return history


if __name__ == "__main__":
    history = train_model()
    # MAEをプロットしてみよう
    fig, ax1 = plt.subplots(1, 1)

    ax1.plot(history.epoch, history.history['loss'])
    ax1.set_title('TrainingError')
    ax1.set_ylabel('Model Loss', fontsize=12)
    ax1.set_xlabel('# Epochs', fontsize=12)
    plt.show()
