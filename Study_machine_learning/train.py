from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from getdata import prepare_data
from functions import history_plot


def build_model(x_train):
    model = Sequential()
    model.add(LSTM(256, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer='adam', metrics='mae')
    return model


def train_model(plt_display=False):
    x_train, y_train, x_test, y_test = prepare_data(window_len=10, train_rate=0.8, limit=10000)
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
    model.save('saved_model/my_model3')

    if plt_display:
        history_plot(history)

    return


if __name__ == "__main__":
    train_model()
