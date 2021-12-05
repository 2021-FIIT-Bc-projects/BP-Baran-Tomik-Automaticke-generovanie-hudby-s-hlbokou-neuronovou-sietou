from music21 import converter, instrument, note, chord
import numpy as np
from keras.utils import np_utils
from fractions import Fraction

# from keras.preprocessing.sequence import TimeseriesGenerator


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


# def list_instruments(midi):
#     part_stream = midi.parts.stream()
#     print("List of instruments found on MIDI file:")
#     for p in part_stream:
#         aux = p
#         print (p.partName)


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


def init(midi_file):

    notes_and_chords, metadata_p, else_array = parse_midi_file(midi_file)               # parse MIDI file
    lstm_input, lstm_output, notes_to_int, pitch_names = mapping(notes_and_chords)      # mapping MIDI file parts
    info_print_out(metadata_p, len(notes_to_int))

    return notes_and_chords, lstm_input, lstm_output, notes_to_int, pitch_names
