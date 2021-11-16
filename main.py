from music21 import converter, instrument, note, chord, stream  #, pitch
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# from keras.preprocessing.sequence import TimeseriesGenerator

import os
from fractions import Fraction
import time

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


def load_midi_file(file_name):

    midi_sample = None
    try:
        midi_sample = converter.parse(file_name)
    except OSError as e:
        print('\nERROR loading MIDI file in load_midi_file()')
        print(e)
        quit()
    return midi_sample


def parse_midi_file(file):

    notes_piano = []
    metadata_piano = {
        "note_count": 0,
        "chord_count": 0,
        "rest": 0,
        "else_count": 0
    }
    else_arr = []

    midi_sample = load_midi_file(file)
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
    # print('koniec parsovania')
    return notes_piano, metadata_piano, else_arr


# def list_instruments(midi):
#     part_stream = midi.parts.stream()
#     print("List of instruments found on MIDI file:")
#     for p in part_stream:
#         aux = p
#         print (p.partName)


def mapping(notes):
    # original
    sequence_length = 10        # 100

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

    # print('---')

    # reshape the input into a format compatible with LSTM layers
    nn_input = np.reshape(nn_input, (input_count, sequence_length, 1))
    # normalize input and values for mapped notes
    nn_input = nn_input / mapped_n_count
    # for i in note_to_int:
    #     note_to_int[i] = note_to_int[i] / mapped_n_count

    nn_output = np_utils.to_categorical(nn_output)
    # print('---')
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


def create_midi_file(output, mapping_keys):

    unmapped_from_int = []
    converted = []
    notes = []
    offset = 0

    metadata = {
        "note": 0,
        "chord": 0,
        "rest": 0
    }

    # # unmapping notes, chores and rests from output integers
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
            metadata['note'] = metadata['note'] + 1
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
            metadata['chord'] = metadata['chord'] + 1
            # print('f')

        elif 'r_' in element:                       # rest
            element = element[2:]                                                       # cut the 'r_' mark
            offset += Fraction(element.split('_')[1])
            rest_ = note.Rest()                                                         # creating rest
            rest_.duration.quarterLength = convert_to_float(element.split('_')[0])      # adding duration quarterLength
            rest_.offset = offset                                                       # adding offset
            rest_.storedInstrument = instrument.Piano()
            converted.append(rest_)                                                     # appending final array
            metadata['rest'] = metadata['rest'] + 1

    # print(converted)
    print('elements of newly generated music', metadata)

    try:
        midi_stream = stream.Stream(converted)
        # midi_stream.append(converted)

        midi_stream.write('midi', fp='midi_samples\\outputs\\beethoven-prediction.mid')
        print('created new MIDI file')
    except OSError as e:
        print('\nERROR creating MIDI file in create_midi_file()')
        print(e)


def info_print_out(metadata, unique_elements_count):
    all_count = metadata['note_count'] + metadata['chord_count'] + metadata['rest']
    print('\n')
    print('_________________________________________________')
    print('\n')
    print('number of all elements:      ', all_count)
    print('notes:    ', metadata['note_count'])
    print('chords:   ', metadata['chord_count'])
    print('rests:    ', metadata['rest'])
    print('number of unique elements:   ', unique_elements_count)
    print('\n')
    print('_________________________________________________')
    print('\n')


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


def generate_music(nn_model, nn_input, mapped_notes):

    generated_music = []
    sequence_len = len(nn_input[0])
    mapped_notes_count = len(mapped_notes)
    start = np.random.randint(0, len(nn_input) - 1)
    # int_to_notes = {v: k for k, v in mapped_notes.items()}
    pattern = nn_input[start]

    note_input = np.array(pattern).reshape((1, sequence_len, 1))
    note_input = note_input / float(mapped_notes_count)

    for new_note in range(100):

        note_output = nn_model.predict(note_input, verbose=0)
        note_output_max = np.argmax(note_output)
        generated_music.append(note_output_max)

        pattern = np.append(pattern, note_output_max / float(mapped_notes_count))
        pattern = pattern[1:sequence_len + 1]

        note_input = np.array(pattern).reshape((1, sequence_len, 1))

    return generated_music


def mapping_for_debug(notes2):      # maping pre simulovanie outputu z NN
    pitchnames2 = set(notes2)
    note_to_int2 = dict((note_var, number) for number, note_var in enumerate(pitchnames2))

    print('chodopice')
    provi = []
    for el in notes2:
        for key in note_to_int2:
            if key == el:
                provi.append(note_to_int2[key])
                # print(" c: ")
                break
    return provi, note_to_int2


# if __name__ == '__main__':
#
#     print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))
#
#     midi_file = 'midi_samples\\FFVII_BATTLE.mid'
#
#     notes_and_chords, metadata_p, else_array = parse_midi_file(midi_file)               # parse MIDI file
#     lstm_input, lstm_output, notes_to_int, pitch_names = mapping(notes_and_chords)      # mapping MIDI file parts
#     info_print_out(metadata_p, len(notes_to_int))
#
#     # print('---')
#
#     # start_time = time.time()
#     empty_model = create_lstm_model(lstm_input, len(pitch_names))                       # load layers of NN to model
#
#     # model = train_lstm(empty_model, lstm_input, lstm_output)                            # train NN
#     model = load_weight_to_model(empty_model)                                           # load weights to model
#
#     new_music = generate_music(model, lstm_input, notes_to_int)                    # predict new music
#     create_midi_file(new_music, notes_to_int)                                           # save new music to MIDI file
#
#     # print("\n\n\n\n%s seconds" % (round((time.time() - start_time), 1)))
#
#     # model.summary()


if __name__ == '__main__':        # debig main for MIDI parsing and creating

    midi_file = 'midi_samples\\classical-piano_beethoven_opus10_2.mid'

    notes_and_chords, metadata_p, else_array = parse_midi_file(midi_file)
    new_music, notes_to_int = mapping_for_debug(notes_and_chords)

    create_midi_file(new_music, notes_to_int)

    # model.summary()
