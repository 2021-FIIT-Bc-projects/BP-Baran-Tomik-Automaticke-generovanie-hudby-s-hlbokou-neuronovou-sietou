# from music21 import converter, instrument, note, chord  #, stream, pitch
from music21 import *
import numpy as np
import tensorflow

# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Activation, Dropout
#
from keras.callbacks import ModelCheckpoint
# from keras.utils import np_utils
from keras.layers import *
from keras.models import *
from keras.callbacks import *


notes_count = 0


def parse_midi_file(file):

    # notes_to_parse = None
    notes_piano = []
    metadata_piano = {
        "note_count": 0,
        "chord_count": 0,
        "else_count": 0
    }

    midi_sample = converter.parse(file)
    instruments = instrument.partitionByInstrument(midi_sample)

    for part in instruments.parts:
        # print(str(part))                          #show Piano Templat

        if 'Piano' in str(part):

            notes_to_parse = part.recurse()
            # print(len(notes_to_parse))            #show Piano Template len (7)

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes_piano.append(str(element.pitch))
                    metadata_piano["note_count"] += 1
                    # print(str(element.pitches).split(' ')[-1][:-3])
                    # print(element.duration.type)
                    # print(element)

                elif isinstance(element, chord.Chord):
                    notes_piano.append('.'.join(str(n) for n in element.normalOrder))
                    metadata_piano["chord_count"] += 1
                    # print(str(element.pitches).split(' ')[-1][:-2])
                    # print(element.notes)

                else:
                    metadata_piano["else_count"] += 1
    return np.array(notes_piano), metadata_piano


# def list_instruments(midi):
#     part_stream = midi.parts.stream()
#     print("List of instruments found on MIDI file:")
#     for p in part_stream:
#         aux = p
#         print (p.partName)


def mapping(notes):
    sequence_length = 100

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

    # # reshape the input into a format compatible with LSTM layers
    nn_input = np.reshape(nn_input, (n_patterns, sequence_length, 1))
    # # normalize input
    # nn_input = nn_input / float(n_vocab)
    #
    # nn_output = np_utils.to_categorical(nn_output)

    return nn_input, nn_output, note_to_int, pitchnames


def save_midi_file(notes):

    # n = note.Note(str(element.pitches).split(' ')[-1][:-3], type=element.duration.type)
    # s.append(n)
    # s.show('text')print('text')
    converted = []

    for element in notes:
        if '.' in element:      #chors
            converted.append(chord.Chord(element))
            print(converted)

        else:                   #note
            converted.append(note.Note(element))
            print(converted)

        print(converted)

    # print(np.array(converted))
    # print(ne)

    # midi_stream = stream.Stream()
    # midi_stream.append(notes)
    # midi_stream.write('midi', fp='midi_samples\\test_output.mid')

    # try:
    #     midi_stream = stream.Stream(notes[0])
    #     # midi_stream.append(notes)
    #     midi_stream.write('midi', fp='midi_samples\\test_output.mid')
    # except:
    #     print('save_midi_file ERROR')


def hluposti():
    # notes_ = [element for note_ in notes_and_chords for element in note_]
    #
    # # No. of unique notes
    # unique_notes = list(set(notes_))
    # print(len(unique_notes))
    #
    # freq = dict(Counter(notes_))
    #
    # # consider only the frequencies
    # no = [count for _, count in freq.items()]
    #
    # # set the figure size
    # plt.figure(figsize=(5, 5))
    #
    # # plot
    # plt.hist(no)
    # # plt.show()
    #
    # frequent_notes = [note_ for note_, count in freq.items() if count >= 50]
    # print(len(frequent_notes))
    #
    # new_music = []
    #
    # for notes in notes_and_chords:
    #     temp = []
    #     for note_ in notes:
    #         if note_ in frequent_notes:
    #             temp.append(note_)
    #     new_music.append(temp)
    #
    # new_music = np.array(new_music)
    #
    # no_of_timesteps = 32
    # x = []
    # y = []
    #
    # for note_ in new_music:
    #     print(len(note_))
    #     for i in range(0, len(note_) - no_of_timesteps, 1):
    #         # preparing input and output sequences
    #         input_ = note_[i:i + no_of_timesteps]
    #         output = note_[i + no_of_timesteps]
    #
    #         x.append(input_)
    #         y.append(output)
    #
    # x = np.array(x)
    # y = np.array(y)
    # print(x)
    # print(y)

    # unique_x = list(set(x.ravel()))
    # x_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_x))
    #
    # x_seq = []
    # for i in x:
    #     temp = []
    #     for j in i:
    #         # assigning unique integer to every note
    #         temp.append(x_note_to_int[j])
    #     x_seq.append(temp)
    #
    # x_seq = np.array(x_seq)
    # print(x_seq)
    pass


def lstm():
    model = Sequential()

    # recurrent layer
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))
    # model.add(LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1))

    # fully connected layer
    model.add(Dense(256, activation='relu'))

    # fropout for regularization
    model.add(Dropout(0.5))

    # output layer
    model.add(Dense(notes_count, activation='softmax'))               # n_vocab
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model


if __name__ == '__main__':

    midi_file = 'midi_samples\\FFVII_BATTLE.mid'
    notes_and_chords, metadata_p = parse_midi_file(midi_file)
    notes_count = len(notes_and_chords)

    print('main')
    print(notes_and_chords)
    print(metadata_p)
    print('koniec mojho')

    lstm_input, lstm_output, notes_to_int, pitch_names = mapping(notes_and_chords)
    # print(lstm_input)
    # print(lstm_output)
    # print(notes_to_int)
    # print(pitch_names)

    # lstm_model = lstm()
    # lstm_model.summary()

    # save_midi_file(notes_and_chords)
