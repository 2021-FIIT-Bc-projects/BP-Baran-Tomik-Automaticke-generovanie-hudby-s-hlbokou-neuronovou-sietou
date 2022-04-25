import parse_MIDI
import train_model
import generate_music
import os
import tensorflow as tf
import time

# tf.debugging.set_log_device_placement(True)

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# with tf.device('/GPU:0'):
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':

    print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))
    # print(tf.config.list_physical_devices('GPU'))

    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    #     try:
    #         tf.config.set_logical_device_configuration(
    #             gpus[0],
    #             [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])
    #         logical_gpus = tf.config.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Virtual devices must be set before GPUs have been initialized
    #         print(e)

    start_time = time.time()

    # file_name = 'elise'
    midi_files_folder = 'midi_samples\\beeth\\'

    # notes_and_chords, lstm_input, lstm_output, notes_to_int, pitch_names = parse_MIDI.init(midi_files_folder)
    lstm_input, lstm_output, notes_to_int, pitch_names_len = parse_MIDI.init(midi_files_folder)

    model = train_model.init(lstm_input, lstm_output, pitch_names_len)

    generate_music.init(model, lstm_input, notes_to_int, 100)
    generate_music.init(model, lstm_input, notes_to_int, 150)
    generate_music.init(model, lstm_input, notes_to_int, 200)
    generate_music.init(model, lstm_input, notes_to_int, 250)
    generate_music.init(model, lstm_input, notes_to_int, 300)

    end_time = time.time()
    print('Time:', int(end_time-start_time), 's')
    print('Total end')
    print(' ')

    # model.summary()
