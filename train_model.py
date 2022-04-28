from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf


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


def load_weight_to_model(empt_model):
    filepath = 'weights\\toLoadWeights\\maestro2018_cut-109-5.8109.hdf5'
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

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #         data = nn.fit(nn_input, nn_output, epochs=250, batch_size=16, callbacks=callbacks_list)  # batch_size=64
    #         return nn, data
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)

    # with tf.device('/GPU:0'):
    data = nn.fit(nn_input, nn_output, epochs=250, batch_size=64, callbacks=callbacks_list)     # batch_size=64

    fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(data.history['loss'], label=f'Loss hodnota', lw=2)
    plt.title('Trénovanie celého datasetu')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.135),
              fancybox=True, shadow=True, ncol=5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.draw()
    plt.show()
    fig.savefig('tests\\cely-dataset-bs64_nieco.pdf')
    plt.clf()
    return nn


def init(lstm_input, lstm_output, pitch_names_len):

    empty_model = create_lstm_model(lstm_input, pitch_names_len)            # load layers of NN to model

    model = train_lstm(empty_model, lstm_input, lstm_output)                # train NN
    # model = load_weight_to_model(empty_model)                             # load weights to model

    return model
