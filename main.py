from music21 import converter, instrument, note, chord, stream, pitch
import numpy as np
# import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM    #CuDNNLSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
            print(len(notes_to_parse))            # show Piano Template len
            print('m')

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    # if str(element.duration.type) == 'complex':
                    #     notes_piano.append('n_'+str(element.pitch)+'_'+str(element.duration.components[0].type))
                    #     print(element.pitch)
                    #     print(element.duration.components[0].type)
                    #     print(element.duration.components[1].type)
                    #
                    # else:
                    #     notes_piano.append('n_'+str(element.pitch)+'_'+str(element.duration.type))
                    notes_piano.append('n_'+str(element.pitch)+'_'+str(element.duration.quarterLength))
                    metadata_piano["note_count"] += 1

                elif isinstance(element, chord.Chord):
                    chord_ = '.'.join(str(n) for n in element.pitches)
                    # if str(element.duration.type) == 'complex':
                    #     notes_piano.append('c_' + chord_ + '_' + str(element.duration.components[0].type))
                    # else:
                    #     notes_piano.append('c_'+chord_+'_'+str(element.duration.type))
                    notes_piano.append('c_' + chord_ + '_' + str(element.duration.quarterLength))
                    metadata_piano["chord_count"] += 1

                elif isinstance(element, note.Rest):
                    # if str(element.duration.type) == 'complex':
                    #     # print(element.duration.components[0].type)
                    #     notes_piano.append('r_' + str(element.duration.components[0].type))
                    #     # print('v')
                    # else:
                    #     notes_piano.append('r_'+str(element.duration.type))
                    #     # print('m')
                    notes_piano.append('r_' + str(element.duration.quarterLength))
                    metadata_piano["rest"] += 1
                else:
                    metadata_piano["else_count"] += 1
                    else_arr.append(element)

        print('m')
    print('m')
    return notes_piano, metadata_piano, else_arr


# def list_instruments(midi):
#     part_stream = midi.parts.stream()
#     print("List of instruments found on MIDI file:")
#     for p in part_stream:
#         aux = p
#         print (p.partName)


def mapping(notes):
    sequence_length = 20        # 100

    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note_var, number) for number, note_var in enumerate(pitchnames))

    nn_input = []
    nn_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]

        nn_input.append([note_to_int[char] for char in sequence_in])
        nn_output.append(note_to_int[sequence_out])
    n_patterns = len(nn_input)

    # reshape the input into a format compatible with LSTM layers
    nn_input = np.reshape(nn_input, (n_patterns, sequence_length, 1))
    # normalize input
    nn_input = nn_input / float(n_patterns)

    nn_output = np_utils.to_categorical(nn_output)

    return nn_input, nn_output, note_to_int, pitchnames


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


def save_midi_file(output, mapping_keys):

    unmapped_from_int = []
    converted = []

    # unmapping notes, chores and rests from output integers
    for element in output:
        index = np.where(element)[0][0]
        for key in mapping_keys:
            if int(mapping_keys[key]) == index:
                unmapped_from_int.append(key)

    # creating note, chores and rest objects
    for element in unmapped_from_int:
        if 'n_' in element:                         # note
            element = element[2:]                                                       # cut the 'n_' mark
            note_ = note.Note(element.split('_')[0])                                    # creating note
            note_.duration.quarterLength = convert_to_float(element.split('_')[1])      # adding duration quarterLength
            converted.append(note_)                                                     # appending final array
            # print('f')

        elif 'c_' in element:                       # chord
            element = element[2:]                                                       # cut the 'c_' mark
            notes_in_chord = element.split('_')[0].split('.')                           # gettig notes in str from element
            chord_obj = chord.Chord(notes_in_chord)                                     # creating chord form notes
            chord_obj.duration.quarterLength = convert_to_float(element.split('_')[1])  # adding duration quarterLength
            converted.append(chord_obj)                                                 # appending final array
            # print('f')

        elif 'r_' in element:                       # rest
            element = element[2:]                                                       # cut the 'r_' mark
            rest = note.Rest()                                                          # creating rest
            rest.duration.quarterLength = convert_to_float(element)                     # adding duration quarterLength
            converted.append(rest)                                                      # appending final array
            # converted.append(note.Rest(type=element))

    print(converted)

    try:
        midi_stream = stream.Stream()
        midi_stream.append(converted)
        midi_stream.write('midi', fp='midi_samples\\outputs\\test_output5.mid')
        print('created new MIDI file')
    except:
        print('save_midi_file ERROR')


def lstm(nn_input, nn_output, n_pitch):
    model = Sequential()

    # # recurrent layer
    # model.add(LSTM(128, return_sequences=True))
    # model.add(Dropout(0.2))
    # # model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))
    #
    # model.add(LSTM(128, return_sequences=True))
    # model.add(Dropout(0.2))
    #
    # # fully connected layer
    # model.add(Dense(32, activation='relu'))
    # # fropout for regularization
    # model.add(Dropout(0.2))
    #
    # # output layer
    # model.add(Dense(diff_notes, activation='softmax'))
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    # return model

    model.add(LSTM(
        256,
        input_shape=(nn_input.shape[1], nn_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(256))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_pitch, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    filepath = "weights\\weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath, monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    model.fit(nn_input, nn_output, epochs=10, batch_size=64, callbacks=callbacks_list)


if __name__ == '__main__':

    midi_file = 'midi_samples\\the_nightmare_begins-cut1_new.mid'
    notes_and_chords, metadata_p, else_array = parse_midi_file(midi_file)

    print('main')
    print(notes_and_chords)
    print(metadata_p)
    print(else_array)
    print('koniec mojho')

    lstm_input, lstm_output, notes_to_int, pitch_names = mapping(notes_and_chords)
    print('po mapping')

    save_midi_file(lstm_output, notes_to_int)

    # print(lstm_input)
    # print(lstm_output)
    # print(notes_to_int)
    # print(pitch_names)

    # lstm_model =
    # lstm(lstm_input, lstm_output, len(pitch_names))
    # lstm_model.summary()


