from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
import time
import matplotlib.pyplot as plt


def create_lstm_model(nn_input, n_pitch, params):

    lstm_model = Sequential()
    lstm_model.add(LSTM(
        256,
        input_shape=(nn_input.shape[1], nn_input.shape[2]),
        return_sequences=True,
    ))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(512, return_sequences=True))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(256))
    lstm_model.add(Dense(256))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(n_pitch))
    lstm_model.add(Activation(params[2]))
    lstm_model.compile(optimizer=params[0], loss=params[1])

    return lstm_model


def train_lstm(nn, nn_input, nn_output):

    data = nn.fit(nn_input, nn_output, epochs=250, batch_size=64, verbose=0)
    return nn, data


def init(lstm_input, lstm_output, pitch_names):

    # params = [['rmsprop', 'categorical_crossentropy', 'tanh'],
    #           ['rmsprop', 'categorical_crossentropy', 'softmax'],
    #           ['rmsprop', 'mean_squared_error', 'tanh'],
    #           ['rmsprop', 'mean_squared_error', 'softmax'],
    #           ['adam', 'categorical_crossentropy', 'tanh'],
    #           ['adam', 'categorical_crossentropy', 'softmax'],
    #           ['adam', 'mean_squared_error', 'tanh'],
    #           ['adam', 'mean_squared_error', 'softmax']]

    params2 = [['rmsprop', 'categorical_crossentropy', 'tanh'],     # 8 - 10
               ['adam', 'categorical_crossentropy', 'tanh']]

    params3 = [['rmsprop', 'categorical_crossentropy', 'softmax'],  # 4 - 6
               ['adam', 'categorical_crossentropy', 'softmax']]

    params4 = [['rmsprop', 'mean_squared_error', 'tanh'],           # ~0
               ['rmsprop', 'mean_squared_error', 'softmax'],
               ['adam', 'mean_squared_error', 'tanh'],
               ['adam', 'mean_squared_error', 'softmax']]

    separated_params = [params2, params3, params4]

    # plt.savefig('tests\\opimizer and loss\\test1.pdf')
    count = 0
    for par in separated_params:
        count = count + 1

        for i in range(len(par)):
            empty_model = create_lstm_model(lstm_input, len(pitch_names), par[i])

            start_time = time.time()
            model, data = train_lstm(empty_model, lstm_input, lstm_output)  # train NN
            stop_time = round((time.time() - start_time), 1)

            plt.plot(data.history['loss'], label=f'{par[i][0]}, {par[i][1]}, {par[i][2]}, {stop_time}')
            plt.title('Optimizer, Loss, Activation, Dĺžka trénovania [s]')
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.draw()
        plt.savefig(f'tests\\opimizer and loss\\test-separated{count}.pdf')

        plt.clf()

    print('\nEnd of testing')

    return False
