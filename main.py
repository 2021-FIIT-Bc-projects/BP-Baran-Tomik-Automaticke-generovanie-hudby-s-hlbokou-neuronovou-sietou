import parse_MIDI
import train_model
import generate_music

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':

    print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))

    file_name = 'elise'

    notes_and_chords, lstm_input, lstm_output, notes_to_int, pitch_names = parse_MIDI.init(file_name)

    model = train_model.init(lstm_input, lstm_output, pitch_names)

    generate_music.init(model, lstm_input, notes_to_int, file_name)

    print('total end')

    # model.summary()
