from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from getdata import prepare_data


def predict_plt(model_name):
    x_train, y_train, x_test, y_test = prepare_data()
    model = load_model(f'saved_model/{model_name}')
    pred_data = model.predict(x_train)
    plt.plot(y_train.reshape(pred_data.shape[0]), label='train')
    plt.plot(pred_data, label='pred')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == "__main__":
    predict_plt('hyperas_model')
