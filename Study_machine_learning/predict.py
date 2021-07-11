from tensorflow.keras.models import load_model
from getdata import prepare_data
from functions import predict_plot


def predict(model_name):
    _, _, x_test, y_test = prepare_data(window_len=10, train_rate=0.8, limit=10000)
    model = load_model(f'saved_model/{model_name}')
    predicted = model.predict(x_test)
    predict_plot(label=y_test, predicted=predicted)

    return


if __name__ == "__main__":
    predict('my_model2')
