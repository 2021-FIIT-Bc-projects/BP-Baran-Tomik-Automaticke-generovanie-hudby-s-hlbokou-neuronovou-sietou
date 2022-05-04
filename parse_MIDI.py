from music21 import converter, instrument, note, chord
import numpy as np
from keras.utils import np_utils
from fractions import Fraction
import sklearn
import os
import gc
from collections import Counter


def load_midi_file(file_name):

    midi_sample = None
    try:
        midi_sample = converter.parse(file_name)
    except OSError as e:
        print('\nERROR loading MIDI file in load_midi_file()')
        print(e)
        quit()
    return midi_sample


def parse_midi_file(folder_path):

    notes_piano = []
    metadata_piano = {
        "note_count": 0,
        "chord_count": 0,
        "rest": 0,
        "else_count": 0
    }

    all_names = []

    for oneFile in os.listdir(folder_path):
        if oneFile.endswith(".mid"):
            all_names.append(oneFile)
            midi_file = folder_path+oneFile
            midi_sample = load_midi_file(midi_file)
            instruments = instrument.partitionByInstrument(midi_sample)

            for part in instruments.parts:

                if 'Piano' in str(part):
                    print('parsing PIANO', part)

                    notes_to_parse = part.recurse()
                    last_offset = 0

                    # note  -> n_pitch_quarterLength_deltaOffset
                    # chord -> c_pitch_quarterLength_deltaOffset
                    # rest  -> r_quarterLength_deltaOffset

                    elems_count = 0
                    for element in notes_to_parse:
                        elems_count += 1
                        if isinstance(element, note.Note):
                            delta_offset = Fraction(element.offset) - last_offset
                            last_offset = Fraction(element.offset)

                            notes_piano.append('n_'+str(element.pitch)+'_'+str(element.duration.quarterLength)+'_'+str(delta_offset))
                            metadata_piano["note_count"] += 1

                        elif isinstance(element, chord.Chord):
                            delta_offset = Fraction(element.offset) - last_offset
                            last_offset = Fraction(element.offset)

                            chord_ = '.'.join(str(n) for n in element.pitches)
                            notes_piano.append('c_' + chord_ + '_' + str(element.duration.quarterLength)+'_'+str(delta_offset))
                            metadata_piano["chord_count"] += 1

                        elif isinstance(element, note.Rest):
                            delta_offset = Fraction(element.offset) - last_offset
                            last_offset = Fraction(element.offset)

                            notes_piano.append('r_' + str(element.duration.quarterLength)+'_'+str(delta_offset))
                            metadata_piano["rest"] += 1
                        else:
                            metadata_piano["else_count"] += 1
                    print('^elems_count:', elems_count)
    print('MIDI dataset:\n', all_names)
    return notes_piano, metadata_piano


def average_f(lst):
    return sum(lst) / len(lst)


def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]


def cut_notes(uncut_notes, metadata, cuts):
    notes = uncut_notes
    stop_flag = False

    for i in range(cuts):
        count_num = Counter(notes)
        Recurrence = list(count_num.values())

        pitchnames = set(notes)
        note_to_int_before = dict((note_var, number) for number, note_var in enumerate(pitchnames))
        avg_r = round(average_f(Recurrence), 2)

        print('\nnumber of notes before:', len(notes))
        print('number of unique notes before:', len(note_to_int_before))
        print("average recurrence for a note in notes:", avg_r)
        print("most frequent note in notes appeared:", max(Recurrence), "times")
        print("least frequent note in notes appeared:", min(Recurrence), "time/s")

        # if average recurrence is more then a 100, @param avg_r is set to 100 and this will be final eliminating
        if round(avg_r) >= 100:
            avg_r = 100
            stop_flag = True

        # getting a list of elements that appear less then avarage element does
        rare_note = []
        cn_items = count_num.items()
        for index, (key, value) in enumerate(cn_items):
            if value < round(avg_r):
                m = key
                rare_note.append(m)
        print(f'number of notes occuring less than {round(avg_r)} times:', len(rare_note))

        # eleminating those elements
        for element in notes:
            if element in rare_note:
                len_before = len(notes)
                notes = remove_values_from_list(notes, element)
                elements_removed = len_before - len(notes)
                if element[:1] == 'n':
                    metadata['note_count'] -= elements_removed
                elif element[:1] == 'c':
                    metadata['chord_count'] -= elements_removed
                elif element[:1] == 'r':
                    metadata['rest'] -= elements_removed

        print("length of notes after the elemination :", len(notes))
        if stop_flag:
            break

    count_num = Counter(notes)
    Recurrence = list(count_num.values())

    avg_r = round(average_f(Recurrence), 2)

    print("\naverage recurrence for a note in notes:", avg_r)
    print("most frequent note in notes appeared:", max(Recurrence), "times")
    print("least frequent note in notes appeared:", min(Recurrence), "time/s")

    del count_num, Recurrence, pitchnames, note_to_int_before, rare_note
    gc.collect()

    return notes, metadata


def mapping(uncut_notes, metadata, sequence_len, cuts):

    notes, new_metadata = cut_notes(uncut_notes, metadata, cuts)

    sequence_length = sequence_len
    pitchnames = set(notes)
    note_to_int = dict((note_var, number) for number, note_var in enumerate(pitchnames))

    nn_input = []
    nn_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]

        nn_input.append([note_to_int[char] for char in sequence_in])
        nn_output.append(note_to_int[sequence_out])

    input_count = len(nn_input)
    mapped_n_count = float(len(note_to_int))

    # reshape the input into a format compatible with LSTM layers
    nn_input = np.reshape(nn_input, (input_count, sequence_length, 1))
    # normalize input and values for mapped notes
    nn_input = nn_input / mapped_n_count
    # for i in note_to_int:
    #     note_to_int[i] = note_to_int[i] / mapped_n_count

    nn_output = np_utils.to_categorical(nn_output)

    # print metadata about notes after removing
    info_print_out(new_metadata, len(note_to_int))

    return nn_input, nn_output, note_to_int, pitchnames


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


def init(folder_path, sequence_length, cuts):
    notes_and_chords, metadata_p = parse_midi_file(folder_path)                                                         # parse MIDI file
    lstm_input, lstm_output, notes_to_int, pitch_names = mapping(notes_and_chords, metadata_p, sequence_length, cuts)   # mapping MIDI file parts

    lstm_input_shuffled, lstm_output_shuffled = sklearn.utils.shuffle(lstm_input, lstm_output)                          # shuffling input and output simultaneously
    pitch_names_len = len(pitch_names)

    del notes_and_chords
    del metadata_p
    del lstm_input
    del lstm_output
    del pitch_names
    gc.collect()

    return lstm_input_shuffled, lstm_output_shuffled, notes_to_int, pitch_names_len
