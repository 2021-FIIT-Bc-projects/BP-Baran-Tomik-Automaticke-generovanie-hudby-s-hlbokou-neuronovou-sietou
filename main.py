import parse_MIDI
import train_model
import generate_music
import os
import tensorflow as tf
import time
import json


os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':

    try:
        start_time = time.time()
        print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))

        with open('config.json', 'r') as f:
            config = json.load(f)

        sequence_length = int(config["sequence_length"])
        midi_files_folder = 'midi_samples\\' + config["dataset_folder"] + '\\'
        cuts = int(config["cuts"])
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        new_music_length = int(config["new_music_length"])
        tracks_to_generate = int(config["tracks_to_generate"])
        new_music_file_name = config["new_music_file_name"]
        model_training = config["training"]

        lstm_input, lstm_output, notes_to_int, pitch_names_len = parse_MIDI.init(midi_files_folder, sequence_length, cuts)
        model = train_model.init(lstm_input, lstm_output, pitch_names_len, epochs, batch_size, model_training)

        for i in range(tracks_to_generate):
            generate_music.init(model, lstm_input, notes_to_int, new_music_length, i, new_music_file_name)

        end_time = time.time()
        print('Time:', int(end_time - start_time), 's')
        print('\n--- END ---\n')
        # model.summary()

    except OSError as e:
        print('\nERROR loading config file in main.py')
        print(e)
        quit()
