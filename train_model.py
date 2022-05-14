from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


def create_lstm_model(nn_input, enique_pitchese_num):
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

    lstm_model.add(Dense(enique_pitchese_num))
    lstm_model.add(Activation('sigmoid'))
    lstm_model.compile(optimizer='Adam', loss='categorical_crossentropy')

    return lstm_model


def load_weight_to_model(empt_model, weight):
    filepath = f'weights\\toLoadWeights\\{weight}.hdf5'
    try:
        empt_model.load_weights(filepath)
    except OSError as e:
        print('\nERROR loading weights file in load_weight_to_model()')
        print(e)
        quit()
    return empt_model


def train_lstm(nn, nn_input, nn_output, epochs, batch_size):

    filepath = "weights\\weights-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    data = nn.fit(nn_input, nn_output, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)

    # draw a graph of loss during training
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(data.history['loss'], label=f'Loss hodnota', lw=2)
    plt.title('Trénovanie celého súboru údajov')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.135),
              fancybox=True, shadow=True, ncol=5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.draw()
    plt.show()
    fig.savefig('krivka_ucenia(Loss).pdf')
    plt.clf()

    return nn


def init(lstm_input, lstm_output, pitch_names_len, epochs, batch_size, model_training):

    empty_model = create_lstm_model(lstm_input, pitch_names_len)                        # load layers of NN to model

    if model_training["bool"]:
        print('\nTRAINING LSTM MODEL...\n\n')
        model = train_lstm(empty_model, lstm_input, lstm_output, epochs, batch_size)    # train NN
    else:
        model = load_weight_to_model(empty_model, model_training["weight"])             # load weights to model

    return model
