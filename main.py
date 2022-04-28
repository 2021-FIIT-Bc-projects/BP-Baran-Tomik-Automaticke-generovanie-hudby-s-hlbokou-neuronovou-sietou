import parse_MIDI
import train_model
import generate_music
import os
import tensorflow as tf
import time


os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
# tf.debugging.set_log_device_placement(True)
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# with tf.device('/GPU:0'):


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':

    print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))

    start_time = time.time()

    midi_files_folder = 'midi_samples\\chopin_cut\\'

    lstm_input, lstm_output, notes_to_int, pitch_names_len = parse_MIDI.init(midi_files_folder)

    model = train_model.init(lstm_input, lstm_output, pitch_names_len)

    generate_music.init(model, lstm_input, notes_to_int, 100)
    # generate_music.init(model, lstm_input, notes_to_int, 150)
    # generate_music.init(model, lstm_input, notes_to_int, 200)
    # generate_music.init(model, lstm_input, notes_to_int, 250)
    generate_music.init(model, lstm_input, notes_to_int, 300)

    end_time = time.time()
    print('Time:', int(end_time-start_time), 's')
    print('\n--- END ---\n')

    # model.summary()
