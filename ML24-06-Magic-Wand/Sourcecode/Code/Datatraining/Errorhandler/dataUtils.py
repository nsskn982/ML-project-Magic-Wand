##
# @file dataUtils.py
#
# @brief This module provides utility functions for handling data in the context of the project.

#

import tensorflow as tf
import pathlib
import errorHandler
import numpy as np

def load_dataset(dataset_path, batch_size=64, validation_split=0.2, seed=0, output_sequence_length=16000):
    try:
        dataDir = pathlib.Path(dataset_path)

        if not dataDir.exists():
            # You can customize the dataset download process based on your Magic Wand dataset.
            # If you have a specific dataset structure, modify the code accordingly.
            tf.keras.utils.get_file(
                'magic_wand_dataset.zip',
                origin="your_dataset_url",
                extract=True,
                cache_dir='.',
                cache_subdir='data'
            )
        else:
            print("The dataset already exists")

        commands = np.array(tf.io.gfile.listdir(str(dataDir)))
        commands = commands[(commands != 'README.md') & (commands != '.DS_Store')]
        print('Commands:', commands)

        train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
            directory=dataDir,
            batch_size=batch_size,
            validation_split=validation_split,
            seed=seed,
            output_sequence_length=output_sequence_length,
            subset='both'
        )

        return train_ds, val_ds

    except Exception:
        errorHandler.errorLoadDataset()

def preprocess_audio_dataset(dataset):
    def squeeze(audio, labels):
        try:
            audio = tf.squeeze(audio, axis=-1)
            return audio, labels
        except Exception:
            errorHandler.errorProcessAudio()

    dataset = dataset.map(squeeze, tf.data.AUTOTUNE)
    return dataset

def create_spectrogram_dataset(dataset):
    def getSpectrogram(waveform):
        try:
            spectrogram = tf.signal.stft(
                waveform, frame_length=255, frame_step=128)
            spectrogram = tf.abs(spectrogram)
            spectrogram = spectrogram[..., tf.newaxis]
            return spectrogram
        except Exception:
            errorHandler.errorSpectrogram()

    dataset = dataset.map(
        map_func=lambda audio, label: (getSpectrogram(audio), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    return dataset
