from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.callbacks import ModelCheckpoint
import numpy as np
# import time

# from keras.preprocessing.sequence import TimeseriesGenerator


def create_lstm_model(nn_input, n_pitch):




    #
    # lstm_model = Sequential()
    # lstm_model.add(LSTM(
    #     256,
    #     input_shape=(nn_input.shape[1], nn_input.shape[2]),
    #     return_sequences=True,
    # ))
    #
    # lstm_model.add(Dropout(0.2))
    # lstm_model.add(LSTM(512, return_sequences=True))
    # lstm_model.add(Dropout(0.2))
    # lstm_model.add(LSTM(256))
    # lstm_model.add(Activation('relu'))
    # lstm_model.add(Dense(256))
    # lstm_model.add(Dropout(0.2))
    # lstm_model.add(Dense(n_pitch))
    # lstm_model.add(Activation('softmax'))
    #
    # # Activation = tanh, softmax, relu
    # # optimizer = adam, rmsprop
    # # loss = mean_squared_error, categorical_crossentropy, sparse_categorical_crossentropy
    # lstm_model.compile(optimizer='adam', loss='categorical_crossentropy')


    # ---------------------------------------

    # lstm_model = Sequential()
    # lstm_model.add(LSTM(
    #     512,
    #     input_shape=(nn_input.shape[1], nn_input.shape[2]),
    #     return_sequences=True,
    # ))
    #
    # lstm_model.add(Dropout(0.6))
    # lstm_model.add(LSTM(512))   # return_sequences=True
    # lstm_model.add(Dropout(0.1))
    # # lstm_model.add(LSTM(256))
    # # lstm_model.add(Dropout(0.6))
    #
    # lstm_model.add(Dense(256))
    # lstm_model.add(Dropout(0.1))
    # lstm_model.add(Dense(128))
    # lstm_model.add(Dropout(0.6))
    # lstm_model.add(Activation('sigmoid'))
    #
    # print(n_pitch)
    # lstm_model.add(Dense(n_pitch, activation='sigmoid'))    # n_pitch == nn_output.shape[1]
    #
    # # Activation = tanh, softmax, relu
    # # optimizer = adam, rmsprop
    # # loss = mean_squared_error, categorical_crossentropy, sparse_categorical_crossentropy
    # lstm_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # z experimentov

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


def load_weight_to_model(empt_model):
    filepath = 'weights\\toLoadWeights\\tesAB2-1-0.3291.hdf5'
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

    # print('Input shape: ' + nn_input.shape)
    # print('Output shape: ' + nn_output.shape)

    # print('X[0]: ', nn_input[0])
    # print('argmax y[0]: ', np.argmax(nn_output[0]))
    # print('argmax y[0] / 182: ', np.argmax(nn_output[0]) / float(182))

    data = nn.fit(nn_input, nn_output, epochs=250, batch_size=64, callbacks=callbacks_list)
    return nn, data


def init(lstm_input, lstm_output, pitch_names):
    # start_time = time.time()

    empty_model = create_lstm_model(lstm_input, len(pitch_names))           # load layers of NN to model

    model, data = train_lstm(empty_model, lstm_input, lstm_output)                            # train NN
    # model = load_weight_to_model(empty_model)  # load weights to model

    return model
