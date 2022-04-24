import sklearn
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
import matplotlib.pyplot as plt
import parse_MIDI

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def create_lstm_model(nn_input, n_pitch):

    lstm_model = Sequential()
    lstm_model.add(LSTM(
        256,
        input_shape=(nn_input.shape[1], nn_input.shape[2]),
        return_sequences=True,
    ))
    lstm_model.add(Dropout(0.6))
    lstm_model.add(LSTM(256, return_sequences=True))
    lstm_model.add(Dropout(0.6))
    lstm_model.add(LSTM(256))
    lstm_model.add(Dropout(0.6))

    lstm_model.add(Dense(256))
    lstm_model.add(Dropout(0.6))

    lstm_model.add(Dense(n_pitch))
    lstm_model.add(Activation('sigmoid'))
    lstm_model.compile(optimizer='Adam', loss='categorical_crossentropy')

    return lstm_model


def train_lstm(nn, nn_input, nn_output, bsize):

    data = nn.fit(nn_input, nn_output, epochs=250, batch_size=bsize, verbose=0)
    return nn, data


def model_exp_init(lstm_input, lstm_output, pitch_names):

    fig = plt.figure()
    ax = plt.subplot(111)

    params = [16, 32, 64, 128, 256]

    for par in params:
        lstm_input_shuffled, lstm_output_shuffled = sklearn.utils.shuffle(lstm_input, lstm_output)
        empty_model = create_lstm_model(lstm_input_shuffled, len(pitch_names))
        model_t, data = train_lstm(empty_model, lstm_input_shuffled, lstm_output_shuffled, par)  # train NN
        plt.plot(data.history['loss'], label=f'{par}', lw=2)
        print(par, ' dotrenovane')

    plt.title('Loss hodnota počas trénovania')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.135),
              fancybox=True, shadow=True, ncol=5)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.draw()
    plt.show()
    fig.savefig('tests\\loss\\test-batch_size_l.pdf')
    plt.clf()

    print('\nEnd of testing')
    return model_t


if __name__ == '__main__':

    print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))

    file_name = 'elise'

    notes_and_chords, lstm_input, lstm_output, notes_to_int, pitch_names = parse_MIDI.init(file_name)

    model = model_exp_init(lstm_input, lstm_output, pitch_names)

    # model = train_model.init(lstm_input, lstm_output, pitch_names)
    # generate_music.init(model, lstm_input, notes_to_int, file_name)

    print('Total end')
