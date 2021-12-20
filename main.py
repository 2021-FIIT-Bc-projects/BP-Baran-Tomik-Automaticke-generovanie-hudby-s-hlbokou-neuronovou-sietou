import parse_MIDI
import train_model
import generate_music
import test_train_model         # docasne

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':

    print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))

    midi_file = 'midi_samples\\FFVII_BATTLE.mid'
    notes_and_chords, lstm_input, lstm_output, notes_to_int, pitch_names = parse_MIDI.init(midi_file)

    model = train_model.init(lstm_input, lstm_output, pitch_names)
    # model = test_train_model.init(lstm_input, lstm_output, pitch_names)         # docasne

    new_music_file_name = 'new_prediction_test_separate'
    generate_music.init(model, lstm_input, notes_to_int, new_music_file_name)

    print('end')

    # model.summary()
