import parse_MIDI
from music21 import instrument, note, chord, stream
from fractions import Fraction

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':

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


    def create_midi_file(notes_and_chores, file_name):

        unmapped_from_int = notes_and_chores
        converted = []
        notes = []
        offset = 0

        metadata = {
            "note": 0,
            "chord": 0,
            "rest": 0
        }

        # creating note, chores and rest objects
        for element in unmapped_from_int:
            if 'n_' in element:  # note
                element = element[2:]  # cut the 'n_' mark
                offset += Fraction(element.split('_')[2])
                note_ = note.Note(element.split('_')[0])  # creating note
                note_.duration.quarterLength = convert_to_float(element.split('_')[1])  # adding duration quarterLength
                note_.offset = offset  # adding offset
                note_.storedInstrument = instrument.Piano()
                converted.append(note_)  # appending final array
                metadata['note'] = metadata['note'] + 1
                # print('f')

            elif 'c_' in element:  # chord
                element = element[2:]  # cut the 'c_' mark
                offset += Fraction(element.split('_')[2])
                notes_in_chord = element.split('_')[0].split('.')  # gettig notes as string from element

                notes.clear()
                for current_note in notes_in_chord:
                    new_note = note.Note(current_note)
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)

                chord_ = chord.Chord(notes)  # creating chord form notes
                chord_.duration.quarterLength = convert_to_float(element.split('_')[1])  # adding duration quarterLength
                chord_.offset = offset  # adding offset
                converted.append(chord_)  # appending final array
                metadata['chord'] = metadata['chord'] + 1
                # print('f')

            elif 'r_' in element:  # rest
                element = element[2:]  # cut the 'r_' mark
                offset += Fraction(element.split('_')[1])
                rest_ = note.Rest()  # creating rest
                rest_.duration.quarterLength = convert_to_float(element.split('_')[0])  # adding duration quarterLength
                rest_.offset = offset  # adding offset
                rest_.storedInstrument = instrument.Piano()
                converted.append(rest_)  # appending final array
                metadata['rest'] = metadata['rest'] + 1

        print('\nElements of newly generated music: ', metadata)

        try:
            midi_stream = stream.Stream(converted)

            midi_stream.write('midi', fp='midi_samples\\outputs\\new_' + file_name + '_1.mid')
            print('created new MIDI file')
        except OSError as e:
            print('\nERROR creating MIDI file in create_midi_file()')
            print(e)


    print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))

    file_name = 'FFVII_BATTLE'

    notes_and_chords, lstm_input, lstm_output, notes_to_int, pitch_names = parse_MIDI.init(file_name)
    create_midi_file(notes_and_chords, file_name)  # save new music to MIDI file

    print('end')
