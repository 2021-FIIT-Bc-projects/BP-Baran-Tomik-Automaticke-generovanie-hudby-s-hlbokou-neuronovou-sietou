from music21 import instrument, note, chord, stream
import numpy as np
from fractions import Fraction


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


def create_midi_file(output, mapping_keys, length, index, new_file_name):

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

        elif 'r_' in element:                       # rest
            element = element[2:]                                                       # cut the 'r_' mark
            offset += Fraction(element.split('_')[1])
            rest_ = note.Rest()                                                         # creating rest
            rest_.duration.quarterLength = convert_to_float(element.split('_')[0])      # adding duration quarterLength
            rest_.offset = offset                                                       # adding offset
            rest_.storedInstrument = instrument.Piano()
            converted.append(rest_)                                                     # appending final array
            metadata['rest'] = metadata['rest'] + 1

    print('\nElements of newly generated music: ', metadata)

    try:
        midi_stream = stream.Stream(converted)
        midi_stream.write('midi', fp='midi_samples\\outputs\\' + f'{new_file_name}' + str(index) + '.mid')
        print('Created new MIDI file')
        print(' ')
    except OSError as e:
        print('\nERROR creating MIDI file in create_midi_file()')
        print(e)


def generate_music(nn_model, nn_input, mapped_notes, length):

    generated_music = []
    sequence_len = len(nn_input[0])
    mapped_notes_count = len(mapped_notes)
    start = np.random.randint(0, len(nn_input) - 1)
    # int_to_notes = {v: k for k, v in mapped_notes.items()}
    pattern = nn_input[start]

    note_input = np.array(pattern).reshape((1, sequence_len, 1))
    note_input = note_input / float(mapped_notes_count)
    # note_input = note_inputQ / float(mapped_notes_count)

    for new_note in range(length):
        note_output = nn_model.predict(note_input, verbose=0)
        note_output_max = np.argmax(note_output)
        generated_music.append(note_output_max)

        pattern = np.append(pattern, note_output_max / float(mapped_notes_count))
        pattern = pattern[1:sequence_len + 1]

        note_input = np.array(pattern).reshape((1, sequence_len, 1))

    return generated_music


def init(model, lstm_input, notes_to_int, length, index, new_file_name):
    new_music = generate_music(model, lstm_input, notes_to_int, length)         # predict new music
    create_midi_file(new_music, notes_to_int, length, index, new_file_name)     # save new music to MIDI file
