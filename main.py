from music21 import converter, instrument, note, chord, stream, pitch
import numpy as np

#tieto prec
from collections import Counter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


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

    # if instruments:
    #     notes_to_parse = instruments.parts[0].recurse()
    # else:
    #     notes_to_parse = midi_sample.flat.notes
    #
    # for element in notes_to_parse:
    #     if isinstance(element, note.Note):
    #         notes.append(str(element.pitch))
    #     elif isinstance(element, chord.Chord):
    #         notes.append('.'.join(str(n) for n in element.normalOrder))

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


if __name__ == '__main__':

    midi_file = 'midi_samples\\the_nightmare_begins-cut1_new.mid'
    notes_and_chords, metadata_p = parse_midi_file(midi_file)

    print('main')
    print(notes_and_chords)
    print(metadata_p)

    sequence_length = 100

    pitchnames = sorted(set(item for item in notes_and_chords))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    print(pitchnames)
    print(note_to_int)

    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes_and_chords) - sequence_length, 1):
        sequence_in = notes_and_chords[i:i + sequence_length]
        sequence_out = notes_and_chords[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    print(len(network_input))
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

    print('network_input')
    print(network_input)
    print('network_output')
    print(network_output)
    print('n_patterns')
    print(n_patterns)


    # normalize input
    # network_input = network_input / float(n_vocab)
    # network_output = np_utils.to_categorical(network_output)









    # save_midi_file(notes_and_chords)
