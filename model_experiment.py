import sklearn
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

import generate_music
import parse_MIDI
import json
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


def create_lstm_model_t(nn_input, n_pitch):

    lstm_model = Sequential()
    lstm_model.add(LSTM(
        256,
        batch_input_shape=(41, nn_input.shape[1], nn_input.shape[2]),
        return_sequences=True,
        stateful=True,
    ))
    lstm_model.add(Dropout(0.6))
    lstm_model.add(LSTM(256, return_sequences=True, stateful=True))
    lstm_model.add(Dropout(0.6))
    lstm_model.add(LSTM(256, stateful=True))
    lstm_model.add(Dropout(0.6))

    lstm_model.add(Dense(256))
    lstm_model.add(Dropout(0.6))

    lstm_model.add(Dense(n_pitch))
    lstm_model.add(Activation('sigmoid'))
    lstm_model.compile(optimizer='Adam', loss='categorical_crossentropy')

    return lstm_model


def train_lstm(nn, nn_input, nn_output, text):

    filepath = "weights\\weights-improvement_" + text + "_-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    data = nn.fit(nn_input, nn_output, epochs=250, batch_size=64, verbose=1, callbacks=callbacks_list)

    return nn, data


def model_exp_init(lstm_input_2, lstm_output_2, pitch_names_length):

    # fig = plt.figure()
    # ax = plt.subplot(111)

    # empty_model = create_lstm_model_t(lstm_input_2, pitch_names_length)
    # model_1, data1 = train_lstm(empty_model, lstm_input_2, lstm_output_2, "_1_")  # train NN
    # # plt.plot(data.history['loss'], label='Nemiešaný, True', lw=2)
    # print('Nemiešaný, True - dotrenovane')

    # -------------------------

    empty_model = create_lstm_model(lstm_input_2, pitch_names_length)
    model_2, data2 = train_lstm(empty_model, lstm_input_2, lstm_output_2, "_2_")  # train NN
    # plt.plot(data.history['loss'], label='Nemiešaný, False', lw=2)
    print('Nemiešaný, False -  dotrenovane')

    # -------------------------

    lstm_input_shuffled, lstm_output_shuffled = sklearn.utils.shuffle(lstm_input_2, lstm_output_2)
    empty_model = create_lstm_model(lstm_input_shuffled, pitch_names_length)
    model_3, data3 = train_lstm(empty_model, lstm_input_shuffled, lstm_output_shuffled, "_3_")  # train NN
    # plt.plot(data.history['loss'], label='Pomiešaný, False', lw=2)
    print('Pomiešaný, False -  dotrenovane')

    # -------------------------

    # plt.title('Loss hodnota počas trénovania')
    #
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                  box.width, box.height * 0.9])
    #
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.135),
    #           fancybox=True, shadow=True, ncol=5)
    #
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.draw()
    # plt.show()
    # fig.savefig('tests\\loss\\test-batch_size_l.pdf')
    # plt.clf()

    print('\nEnd of testing')
    # return model_1, model_2, model_3
    return model_2, model_3


if __name__ == '__main__':

    print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))

    with open('config.json', 'r') as f:
        config = json.load(f)

    sequence_length = int(config["sequence_length"])
    midi_files_folder = 'midi_samples\\' + config["dataset_folder"] + '\\'
    cuts = int(config["cuts"])
    epochs = int(config["epochs"])
    batch_size = int(config["batch_size"])
    new_music_length = int(config["new_music_length"])
    tracks_to_generate = int(config["tracks_to_generate"])
    new_music_file_name = config["new_music_file_name"]
    model_training = config["training"]

    lstm_input, lstm_output, notes_to_int, pitch_names_len = parse_MIDI.init(midi_files_folder, sequence_length, cuts)

    model2, model3 = model_exp_init(lstm_input, lstm_output, pitch_names_len)
    models = [model2, model3]

    for i, model in enumerate(models):
        new_music_file_name_ag = new_music_file_name + "model" + str(i+2) + "_"
        generate_music.init(model, lstm_input, notes_to_int, 150, 1, new_music_file_name_ag)
        generate_music.init(model, lstm_input, notes_to_int, 150, 2, new_music_file_name_ag)
        generate_music.init(model, lstm_input, notes_to_int, 150, 3, new_music_file_name_ag)

    print('Total end')
