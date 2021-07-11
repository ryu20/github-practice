import matplotlib.pyplot as plt


def predict_plot(label, predicted):
    plt.plot(label.reshape(predicted.shape[0]), label='train')
    plt.plot(predicted, label='pred')
    plt.legend(loc='upper left')
    plt.show()


def history_plot(history):
    # MAEをプロットしてみよう
    fig, ax1 = plt.subplots(1, 1)

    ax1.plot(history.epoch, history.history['loss'])
    ax1.set_title('TrainingError')
    ax1.set_ylabel('Model Loss', fontsize=12)
    ax1.set_xlabel('# Epochs', fontsize=12)
    plt.show()
