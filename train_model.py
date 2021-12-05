from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.callbacks import ModelCheckpoint


# from keras.preprocessing.sequence import TimeseriesGenerator

import time


def create_lstm_model(nn_input, n_pitch):
    # hidden_nodes = int(2/3 * (len(nn_input[0]) * len(nn_output[0])))
    # print('hidden nodes: ', hidden_nodes)

    # # jedna vrstva, dence nakonci
    # lstm_model = Sequential()
    # lstm_model.add(LSTM(
    #     20,
    #     activation='tanh',
    #     input_shape=(nn_input.shape[1], nn_input.shape[2]),
    #     # return_sequences=True,
    # ))
    # lstm_model.add(Dropout(0.2))
    # lstm_model.add(Dense(n_pitch))
    # lstm_model.compile(optimizer='rmsprop', loss='mse')    # loss='mean_squared_error', loss='categorical_crossentropy'
    # lstm_model.fit(nn_input, nn_output, epochs=150, batch_size=8)
    # return lstm_model

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
    lstm_model.add(Activation('relu'))
    lstm_model.add(Dense(256))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(n_pitch))
    lstm_model.add(Activation('softmax'))

    # Activation = tanh, softmax, relu
    # optimizer = adam, rmsprop
    # loss = mean_squared_error, categorical_crossentropy, sparse_categorical_crossentropy
    lstm_model.compile(optimizer='adam', loss='categorical_crossentropy')

    return lstm_model


def load_weight_to_model(empt_model):
    filepath = 'weights\\toLoadWeights\\6 ffvii_battle weights-improvement-247-0.0671-bigger.hdf5'
    try:
        empt_model.load_weights(filepath)
    except OSError as e:
        print('\nERROR loading weights file in load_weight_to_model()')
        print(e)
        quit()
    return empt_model


def train_lstm(nn, nn_input, nn_output):

    filepath = "weights\\weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    nn.fit(nn_input, nn_output, epochs=250, batch_size=64, callbacks=callbacks_list)
    return nn


def init(lstm_input, lstm_output, pitch_names):
    # start_time = time.time()
    empty_model = create_lstm_model(lstm_input, len(pitch_names))           # load layers of NN to model

    model = train_lstm(empty_model, lstm_input, lstm_output)                            # train NN
    # model = load_weight_to_model(empty_model)  # load weights to model

    return model

