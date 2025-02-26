##
# @mainpage Data Training for Magic Wand Project
#
# @section author Author
# - Created by Deepti Hegde on 30.01.2024.
# - Modified by Deepti Hegde on 02.02.2024.
#
# @section intro_sec Introduction
#
# The provided modular code encompasses various aspects of training a machine learning model. It includes modules for data loading, preparation, augmentation, and splitting. The code is designed to ensure data integrity, perform effective data preparation, and train a robust model. Unit tests are provided to validate each module's correctness, contributing to the reliability of the entire training pipeline. The overarching goal is to enable a comprehensive and reliable approach to training machine learning models on your specific dataset.
#
# @section modules Modules
#
# - dataaugmentation.py
#   Implements data augmentation techniques for enhancing the diversity of your dataset. Augmentation helps in training a more robust model by generating variations of the existing data.
#   Data augmentation that will be used in data_load.py.
#   To perform transformations on input data to create additional training samples.
#
# - dataaugmentationtest.py
#   Test module ensuring the correctness of data augmentation functionalities. It validates that the augmentation techniques produce the expected results.
#
# - dataload.py
#   Module for loading the training dataset, managing file paths, and preparing data for model training. It ensures efficient handling of input data.
#
# - dataloadtest.py
#   Test module validating the correctness of data loading and preparation processes. It ensures that the dataset is correctly loaded and ready for training.
#
# - dataprepare.py
#   Module responsible for preparing the dataset, including preprocessing steps before training. It ensures the data is in the appropriate format for the machine learning model.
#
# - datapreparetest.py
#   Test module ensuring the correctness of dataset preparation functionalities. It validates that the dataset is appropriately preprocessed for training.
#
# - datasplit.py
#   Module for splitting the dataset into training, validation, and testing sets. It helps in organizing data for effective model training and evaluation.
#
# - datasplittest.py
#   Test module validating the dataset splitting functionality. It ensures that the dataset is correctly divided into training, validation, and testing sets.
#
# - traintest.py
#   Test module for validating the model training process. It ensures that the training process produces the expected outcomes.
#
# - dataUtils.py
#   This module provides utility functions for handling data in the context of the Magic Wand project. It includes functions for loading datasets, preprocessing audio data, and creating spectrogram datasets. The module plays a crucial role in preparing data for model training by transforming raw audio data into a format suitable for machine learning models.
#
# - errorHandler.py
#   The module contains a set of custom error functions tailored for the Magic Wand project. These error functions raise specific runtime errors with descriptive messages to assist in debugging and identifying issues during different stages of the project, such as loading datasets, processing audio, creating spectrograms, exporting models, and more. The module aims to provide clear and informative error messages to facilitate effective troubleshooting.
#
# @file train.py
# @brief Main module for training the machine learning model using the prepared dataset. It contains the core logic for model training.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import os
from data_load import DataLoader
import numpy as np
import tensorflow as tf

logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


def reshape_function(data, label):
  reshaped_data = tf.reshape(data, [-1, 3, 1])
  return reshaped_data, label


def calculate_model_size(model):
  print(model.summary())
  var_sizes = [
      np.product(list(map(int, v.shape))) * v.dtype.size
      for v in model.trainable_variables
  ]
  print("Model size:", sum(var_sizes) / 1024, "KB")


def build_cnn(seq_length):
  """Builds a convolutional neural network in Keras."""
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(
          8, (4, 3),
          padding="same",
          activation="relu",
          input_shape=(seq_length, 3, 1)),  # output_shape=(batch, 128, 3, 8)
      tf.keras.layers.MaxPool2D((3, 3)),  # (batch, 42, 1, 8)
      tf.keras.layers.Dropout(0.1),  # (batch, 42, 1, 8)
      tf.keras.layers.Conv2D(16, (4, 1), padding="same",
                             activation="relu"),  # (batch, 42, 1, 16)
      tf.keras.layers.MaxPool2D((3, 1), padding="same"),  # (batch, 14, 1, 16)
      tf.keras.layers.Dropout(0.1),  # (batch, 14, 1, 16)
      tf.keras.layers.Flatten(),  # (batch, 224)
      tf.keras.layers.Dense(16, activation="relu"),  # (batch, 16)
      tf.keras.layers.Dropout(0.1),  # (batch, 16)
      tf.keras.layers.Dense(4, activation="softmax")  # (batch, 4)
  ])
  model_path = os.path.join("./netmodels", "CNN")
  print("Built CNN.")
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  model.load_weights("./netmodels/CNN/weights.h5")
  return model, model_path


def build_lstm(seq_length):
  """Builds an LSTM in Keras."""
  model = tf.keras.Sequential([
      tf.keras.layers.Bidirectional(
          tf.keras.layers.LSTM(22),
          input_shape=(seq_length, 3)),  # output_shape=(batch, 44)
      tf.keras.layers.Dense(4, activation="sigmoid")  # (batch, 4)
  ])
  model_path = os.path.join("./netmodels", "LSTM")
  print("Built LSTM.")
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  return model, model_path


def load_data(train_data_path, valid_data_path, test_data_path, seq_length):
  data_loader = DataLoader(
      train_data_path, valid_data_path, test_data_path, seq_length=seq_length)
  data_loader.format()
  return data_loader.train_len, data_loader.train_data, data_loader.valid_len, \
      data_loader.valid_data, data_loader.test_len, data_loader.test_data


def build_net(args, seq_length):
  if args.model == "CNN":
    model, model_path = build_cnn(seq_length)
  elif args.model == "LSTM":
    model, model_path = build_lstm(seq_length)
  else:
    print("Please input correct model name.(CNN  LSTM)")
  return model, model_path


def train_net(
    model,
    model_path,  # pylint: disable=unused-argument
    train_len,  # pylint: disable=unused-argument
    train_data,
    valid_len,
    valid_data,  # pylint: disable=unused-argument
    test_len,
    test_data,
    kind):
  """Trains the model."""
  calculate_model_size(model)
  epochs = 50
  batch_size = 64
  model.compile(
      optimizer="adam",
      loss="sparse_categorical_crossentropy",
      metrics=["accuracy"])
  if kind == "CNN":
    train_data = train_data.map(reshape_function)
    test_data = test_data.map(reshape_function)
    valid_data = valid_data.map(reshape_function)
  test_labels = np.zeros(test_len)
  idx = 0
  for data, label in test_data:  # pylint: disable=unused-variable
    test_labels[idx] = label.numpy()
    idx += 1
  train_data = train_data.batch(batch_size).repeat()
  valid_data = valid_data.batch(batch_size)
  test_data = test_data.batch(batch_size)
  model.fit(
      train_data,
      epochs=epochs,
      validation_data=valid_data,
      steps_per_epoch=1000,
      validation_steps=int((valid_len - 1) / batch_size + 1),
      callbacks=[tensorboard_callback])
  loss, acc = model.evaluate(test_data)
  pred = np.argmax(model.predict(test_data), axis=1)
  confusion = tf.math.confusion_matrix(
      labels=tf.constant(test_labels),
      predictions=tf.constant(pred),
      num_classes=4)
  print(confusion)
  print("Loss {}, Accuracy {}".format(loss, acc))
  # Convert the model to the TensorFlow Lite format without quantization
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()

  # Save the model to disk
  open("model.tflite", "wb").write(tflite_model)

  # Convert the model to the TensorFlow Lite format with quantization
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
  tflite_model = converter.convert()

  # Save the model to disk
  open("model_quantized.tflite", "wb").write(tflite_model)

  basic_model_size = os.path.getsize("model.tflite")
  print("Basic model is %d bytes" % basic_model_size)
  quantized_model_size = os.path.getsize("model_quantized.tflite")
  print("Quantized model is %d bytes" % quantized_model_size)
  difference = basic_model_size - quantized_model_size
  print("Difference is %d bytes" % difference)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", "-m")
  parser.add_argument("--person", "-p")
  args = parser.parse_args()

  seq_length = 128

  print("Start to load data...")
  if args.person == "true":
    train_len, train_data, valid_len, valid_data, test_len, test_data = \
        load_data("./person_split/train", "./person_split/valid",
                  "./person_split/test", seq_length)
  else:
    train_len, train_data, valid_len, valid_data, test_len, test_data = \
        load_data("./data/train", "./data/valid", "./data/test", seq_length)

  print("Start to build net...")
  model, model_path = build_net(args, seq_length)

  print("Start training...")
  train_net(model, model_path, train_len, train_data, valid_len, valid_data,
            test_len, test_data, args.model)

  print("Training finished!")
