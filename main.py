from music21 import converter, instrument, note, chord, stream #, pitch
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# from keras.preprocessing.sequence import TimeseriesGenerator

import os
from fractions import Fraction

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# source of this function : https://stackoverflow.com/questions/1806278/convert-fraction-to-float
def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac


def parse_midi_file(file):

    notes_piano = []
    metadata_piano = {
        "note_count": 0,
        "chord_count": 0,
        "rest": 0,
        "else_count": 0
    }
    else_arr = []

    midi_sample = converter.parse(file)
    instruments = instrument.partitionByInstrument(midi_sample)

    for part in instruments.parts:

        if 'Piano' in str(part):

            notes_to_parse = part.recurse()
            # print(len(notes_to_parse))            # show Piano Template len
            last_offset = 0

            # note  -> n_pitch_quarterLength_deltaOffset
            # chord -> c_pitch_quarterLength_deltaOffset
            # rest  -> r_quarterLength_deltaOffset

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    delta_offset = Fraction(element.offset) - last_offset
                    last_offset = Fraction(element.offset)

                    notes_piano.append('n_'+str(element.pitch)+'_'+str(element.duration.quarterLength)+'_'+str(delta_offset))
                    # notes_piano.append('n_'+str(element.pitch)+'_'+str(element.duration.quarterLength))
                    metadata_piano["note_count"] += 1

                elif isinstance(element, chord.Chord):
                    delta_offset = Fraction(element.offset) - last_offset
                    last_offset = Fraction(element.offset)

                    chord_ = '.'.join(str(n) for n in element.pitches)
                    notes_piano.append('c_' + chord_ + '_' + str(element.duration.quarterLength)+'_'+str(delta_offset))
                    # notes_piano.append('c_' + chord_ + '_' + str(element.duration.quarterLength))
                    metadata_piano["chord_count"] += 1

                elif isinstance(element, note.Rest):
                    delta_offset = Fraction(element.offset) - last_offset
                    last_offset = Fraction(element.offset)

                    notes_piano.append('r_' + str(element.duration.quarterLength)+'_'+str(delta_offset))
                    # notes_piano.append('r_' + str(element.duration.quarterLength))
                    metadata_piano["rest"] += 1
                else:
                    metadata_piano["else_count"] += 1
                    else_arr.append(element)
        # print('m')
    print('koniec parsovania')
    return notes_piano, metadata_piano, else_arr


# def list_instruments(midi):
#     part_stream = midi.parts.stream()
#     print("List of instruments found on MIDI file:")
#     for p in part_stream:
#         aux = p
#         print (p.partName)


def mapping(notes):
    # original
    sequence_length = 5        # 100

    pitchnames = set(notes)

    note_to_int = dict((note_var, number) for number, note_var in enumerate(pitchnames))

    nn_input = []
    nn_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]

        nn_input.append([note_to_int[char] for char in sequence_in])
        nn_output.append(note_to_int[sequence_out])

    input_count = len(nn_input)
    mapped_n_count = float(len(note_to_int))

    print('---')

    # reshape the input into a format compatible with LSTM layers
    nn_input = np.reshape(nn_input, (input_count, sequence_length, 1))
    # normalize input and values for mapped notes
    nn_input = nn_input / mapped_n_count
    # for i in note_to_int:
    #     note_to_int[i] = note_to_int[i] / mapped_n_count

    nn_output = np_utils.to_categorical(nn_output)
    print('---')
    return nn_input, nn_output, note_to_int, pitchnames


# def mapping(notes):
#     # skuska implementovat TimeseriesGenerator
#     sequence_length = 50        # 100
#
#     pitchnames = sorted(set(item for item in notes))
#     note_to_int = dict((note_var, number) for number, note_var in enumerate(pitchnames))
#
#     notes_as_int = list(note_to_int.values())
#     nn_input = []
#     nn_output = []
#
#     generator = TimeseriesGenerator(note_to_int, note_to_int, length=sequence_length, batch_size=1)
#
#     # create input sequences and the corresponding outputs
#     for i in range(0, len(notes) - sequence_length, 1):
#         sequence_in = notes[i:i + sequence_length]
#         sequence_out = notes[i + sequence_length]
#
#         nn_input.append([note_to_int[char] for char in sequence_in])
#         nn_output.append(note_to_int[sequence_out])
#     n_patterns = len(nn_input)
#
#     print(generator)
#     print('r')
#
#     # reshape the input into a format compatible with LSTM layers
#     nn_input = np.reshape(nn_input, (n_patterns, sequence_length, 1))
#     # normalize input
#     nn_input = nn_input / float(n_patterns)
#
#     nn_output = np_utils.to_categorical(nn_output)
#
#     return nn_input, nn_output, note_to_int, pitchnames


def save_midi_file(output, mapping_keys):

    unmapped_from_int = []
    converted = []
    notes = []
    offset = 0

    # # unmapping notes, chores and rests from output integers
    # # v2
    for element in output:
        for key in mapping_keys:
            if int(mapping_keys.get(key)) == element:
                unmapped_from_int.append(key)
                break

    # creating note, chores and rest objects
    for element in unmapped_from_int:
        if 'n_' in element:                         # note
            element = element[2:]                                                       # cut the 'n_' mark
            offset += Fraction(element.split('_')[2])
            note_ = note.Note(element.split('_')[0])                                    # creating note
            note_.duration.quarterLength = convert_to_float(element.split('_')[1])      # adding duration quarterLength
            note_.offset = offset                                                       # adding offset
            note_.storedInstrument = instrument.Piano()
            converted.append(note_)                                                     # appending final array
            # print('f')

        elif 'c_' in element:                       # chord
            element = element[2:]                                                       # cut the 'c_' mark
            offset += Fraction(element.split('_')[2])
            notes_in_chord = element.split('_')[0].split('.')                           # gettig notes as string from element

            notes.clear()
            for current_note in notes_in_chord:
                new_note = note.Note(current_note)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)

            chord_ = chord.Chord(notes)                                                 # creating chord form notes
            chord_.duration.quarterLength = convert_to_float(element.split('_')[1])     # adding duration quarterLength
            chord_.offset = offset                                                      # adding offset
            converted.append(chord_)                                                    # appending final array
            # print('f')

        elif 'r_' in element:                       # rest
            element = element[2:]                                                       # cut the 'r_' mark
            offset += Fraction(element.split('_')[1])
            rest_ = note.Rest()                                                         # creating rest
            rest_.duration.quarterLength = convert_to_float(element.split('_')[0])      # adding duration quarterLength
            rest_.offset = offset                                                       # adding offset
            rest_.storedInstrument = instrument.Piano()
            converted.append(rest_)                                                     # appending final array

    # print(converted)

    try:
        midi_stream = stream.Stream(converted)
        # midi_stream.append(converted)

        midi_stream.write('midi', fp='midi_samples\\outputs\\67_tests.mid')
        print('created new MIDI file')
    except:
        print('save_midi_file ERROR')


def lstm(nn_input, nn_output, n_pitch):
    lstm_model = Sequential()

    # # recurrent layer
    # lstm_model.add(LSTM(128, return_sequences=True))
    # lstm_model.add(Dropout(0.2))
    # # lstm_model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
    #
    # lstm_model.add(LSTM(128, return_sequences=True))
    # lstm_model.add(Dropout(0.2))
    #
    # # fully connected layer
    # lstm_model.add(Dense(32, activation='relu'))
    # # fropout for regularization
    # lstm_model.add(Dropout(0.2))
    #
    # # output layer
    # lstm_model.add(Dense(diff_notes, activation='softmax'))
    # lstm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    # return lstm_model

    lstm_model.add(LSTM(
        256,
        input_shape=(nn_input.shape[1], nn_input.shape[2]),
        return_sequences=True
    ))
    lstm_model.add(Dropout(0.3))
    lstm_model.add(LSTM(512, return_sequences=True))
    lstm_model.add(Dropout(0.3))
    lstm_model.add(LSTM(256))
    lstm_model.add(Dense(256))
    lstm_model.add(Dropout(0.3))
    lstm_model.add(Dense(n_pitch, activation='softmax'))
    lstm_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    filepath = "weights\\weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    lstm_model.fit(nn_input, nn_output, epochs=10, batch_size=64, callbacks=callbacks_list)


def simple_lstm(nn_input, nn_output, n_pitch):
    # hidden_nodes = int(2/3 * (len(nn_input[0]) * len(nn_output[0])))
    # print('hidden nodes: ', hidden_nodes)

    lstm_model = Sequential()
    lstm_model.add(LSTM(
        20,
        activation='tanh',
        input_shape=(nn_input.shape[1], nn_input.shape[2]),
        # return_sequences=True,
    ))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(n_pitch))
    lstm_model.compile(optimizer='rmsprop', loss='mse')    # loss='mean_squared_error', loss='categorical_crossentropy'
    lstm_model.fit(nn_input, nn_output, epochs=150, batch_size=8)
    return lstm_model


if __name__ == '__main__':

    print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))
    midi_file = None
    gen_music = []

    try:
        midi_file = 'midi_samples\\67notes.mid'
    except OSError as e:
        print('ERROR loading MIDI file')
        exit(1)

    # parsovanie midi suboru
    notes_and_chords, metadata_p, else_array = parse_midi_file(midi_file)

    # mapovanie midi suboru
    lstm_input, lstm_output, notes_to_int, pitch_names = mapping(notes_and_chords)
    mapped_notes_count = len(notes_to_int)

    print('---')

    # trenovanie NN
    model = simple_lstm(lstm_input, lstm_output, len(pitch_names))

    sequence_len = len(lstm_input[0])
    last_input_seq = lstm_input[len(lstm_input) - 1]
    first_input_seq = lstm_input[0]

    # generovanie pitch1
    x_in = np.array(first_input_seq).reshape((1, sequence_len, 1))
    x_in = x_in / float(mapped_notes_count)
    out = model.predict(x_in, verbose=1)
    # print('out: ', out)
    out_ee = np.argmax(out)
    # print('out_ee: ', out_ee)
    gen_music.append(out_ee)

    # generovanie pitch2
    x_in = np.array(first_input_seq).reshape((1, sequence_len, 1))
    x_in = x_in / float(mapped_notes_count)
    out = model.predict(x_in, verbose=1)
    out_ee = np.argmax(out)
    gen_music.append(out_ee)

    # ukladanie vygenerovanych pitch do midi suboru
    save_midi_file(gen_music, notes_to_int)

    # lstm_model.summary()
