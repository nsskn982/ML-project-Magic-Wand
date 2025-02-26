"""! @brief Python script for image classification using a Convolutional Neural Network (CNN)."""

############################
# Creator of the File: Adhiraj Walse
# Date created: 20.12.2024
# Path: ML23-06-Magic-Wand-with-an-Arduino-Nano-33-BLE-sense\report\Code\CNN\CNNDataMining.py
# Version: 2.0
# Reviewed by: Adhiraj Walse
# Review Date: 06.01.2025
############################

# The following code is adapted from the book:
#       [title = Practical Convolutional Neural Networks:
#                   Implement Advanced Deep Learning Models Using Python
#       Authors   = Mohit Sewak, Md. Rezaul Karim and Pradeep Pujari
#       year      = 2018,
#       publisher = Packt Publishing Ltd,
#       ISBN      = 9781788392303]

##
# @mainpage CIFAR-10 Image Classification with CNN
#
# @section author Author
# - Documented by Adhiraj Walse on 06.01.2025.
# - Original code by Sewak et al in the book "Practical Convolutional Neural Networks: Implement Advanced Deep Learning
# Models Using Python." Packt Publishing Ltd, 2018. isbn: 9781788392303
#
# @section intro_sec Introduction
#
# This documentation provides an overview of a Python script for image classification using a Convolutional Neural
# Network (CNN). The code utilizes the CIFAR-10 dataset and the Keras library for building and training the CNN model.
#
# @section cifar10_dataset CIFAR-10 Dataset
#
# The script loads the CIFAR-10 dataset, performs data preprocessing, and splits it into training, validation, and
# test sets.
# @image html CIFAR10FirstNineImages.png "First 9 images in the CIFAR-10 dataset." width=500px
#
#
# @section image_augmentation Image Augmentation
#
# Data augmentation is applied using the ImageDataGenerator from Keras to increase the diversity of the training
# dataset.
#
# @section model_definition CNN Model Definition
#
# The CNN model is defined using Keras Sequential API. It consists of convolutional layers, max pooling layers, dropout
# layers, nd dense layers for classification. The model summary is provided for reference.
##
# @section model_training Model Training
#
# The model is compiled using categorical crossentropy loss and the RMSprop optimizer. ModelCheckpoint is used as a
# callback to save the best weights during training. The script then trains the model using augmented data.
#
# @section training_results Training Results
#
# The training results, including accuracy and loss, are stored in the 'hist' variable. The script also plots and
# saves the first nine images from the training dataset for visualization.
#
# @section start Getting Started
#
# @warning Before running the script, ensure that you have the required dependencies installed. Additionally,
# verify that the 'model.weights.best.hdf5' file is correctly saved during training for later use.
#
# @note For further details on the code and Keras functionalities, refer to the inline comments within the script.
#

##
# @file CNNDataMining.py
#
# @brief Python script for image classification using a Convolutional Neural Network (CNN) on the CIFAR-10 dataset.
#
#  This code file demonstrates the implementation of a CNN model for image classification using the CIFAR-10 dataset.
#  It includes data preprocessing, model definition, and training with augmented data.
#
# @section author Author
# - Original code by Sewak et al in the book "Practical Convolutional Neural Networks: Implement Advanced Deep Learning
# Models Using Python." Packt Publishing Ltd, 2018. isbn: 9781788392303
# - Modified and documented by Adhiraj Walse on 06.01.2005.
#

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

# Load CIFAR-10 dataset

## @var xTrain
#  @brief Training images from the CIFAR-10 dataset.
#
## @var yTrain
#  @brief Labels corresponding to the training images in xTrain.
#  The labels are one-hot encoded using np_utils.to_categorical.
#
## @var xTest
#  @brief Test images from the CIFAR-10 dataset.
#
## @var yTest
#  @brief Labels corresponding to the test images in xTest.
#  The labels are one-hot encoded using np_utils.to_categorical.

(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()

# Data Preprocessing

## @var numClasses
#  @brief Number of classes in the CIFAR-10 dataset.
#  This variable is determined by the number of unique labels in yTrain.
#
# break training set into training and validation sets
## @var xValid
#  @brief Validation images obtained from the initial training set (xTrain).
#  This subset is used for model validation during training.
#
## @var yValid
#  @brief Labels corresponding to the validation images in xValid.
#  These labels are extracted from the initial training labels (yTrain).

# rescale [0,255] -> [0,1]
xTrain = xTrain.astype('float32') / 255

# one-hot encode the labels
numClasses = len(np.unique(yTrain))
yTrain = np_utils.to_categorical(yTrain, numClasses)
yTest = np_utils.to_categorical(yTest, numClasses)

(xTrain, xValid) = xTrain[5000:], xTrain[:5000]
(yTrain, yValid) = yTrain[5000:], yTrain[:5000]

# print shape of training set
print('xTrain shape:', xTrain.shape)

# printing the number of training, validation, and test images
print(xTrain.shape[0], 'train samples')
print(xTest.shape[0], 'test samples')
print(xValid.shape[0], 'validation samples')

# Augmented image generator for training
## @var datagenTrain
#  @brief ImageDataGenerator for augmenting training images.
#  This generator is used to apply various transformations to the training images,
#  such as width and height shifts and horizontal flips.
#
# Augmented image generator for validation
## @var datagenValid
#  @brief ImageDataGenerator for augmenting validation images.
#  Similar to datagenTrain, this generator applies transformations to validation images.

datagenTrain = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

datagenValid = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Fitting augmented image generator on data
datagenTrain.fit(xTrain)
datagenValid.fit(xValid)

# Plot nine images in the training dataset

# Figure size for plotting images
## @var figsize
#  @brief Size of the figure for plotting images.
#  This variable is used in plt.figure to set the size of the plot.

plt.figure(figsize=(10, 10))
for i in range(10,18):
    plt.subplot(330 + 1 + i)
    plt.imshow(xTrain[i])
plt.tight_layout()  # Adjust layout for better visualization
plt.savefig('CIFAR10NineImages.png')  # Save the figure
plt.show()


# CNN model

# CNN model
## @var model
#  @brief Sequential model for Convolutional Neural Network (CNN) classification.

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.summary()

# Compile the model

## @var loss
#  @brief Loss function used during model compilation.
#  Categorical crossentropy is a common choice for classification tasks.
#
## @var optimizer
#  @brief Optimizer used during model compilation.
#  RMSprop is used as the optimizer for training the model.
#
## @var metrics
#  @brief Metrics to be evaluated during model training.
#  'accuracy' is chosen as the metric to monitor during training.

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# ModelCheckpoint callback

## @var checkpointer
#  @brief ModelCheckpoint callback to save the best weights during training.

checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)

# Train the model with augmented data

## @var hist
#  @brief History object containing training and validation results.
#  The history is obtained from the model.fit_generator method.

hist = model.fit_generator(
    datagenTrain.flow(xTrain, yTrain, batch_size=32),
    steps_per_epoch=len(xTrain) // 32,
    epochs=10,
    validation_data=datagenValid.flow(xValid, yValid, batch_size=32),
    validation_steps=len(xValid) // 32,
    callbacks=[checkpointer],
    verbose=2,
    shuffle=True
)

# Extracting training and validation loss and accuracy from the training history
## @var trainingLoss
#  @brief List containing training loss values for each epoch.
#
## @var trainingAccuracy
#  @brief List containing training accuracy values for each epoch.
#
## @var validationLoss
#  @brief List containing validation loss values for each epoch.
#
## @var validationAccuracy
#  @brief List containing validation accuracy values for each epoch.
## @var label
#  @brief Label for the loss and accuracy curves in the plots.
trainingLoss = hist.history['loss']
trainingAccuracy = hist.history['accuracy']
validationLoss = hist.history['val_loss']
validationAccuracy = hist.history['val_accuracy']

# Plotting and saving the loss curve
plt.plot(trainingLoss, label='Training Loss')
plt.plot(validationLoss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('lossPlot.png')
plt.show()

# Plotting and saving the accuracy curve
plt.plot(trainingAccuracy, label='Training Accuracy')
plt.plot(validationAccuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracyPlot.png')
plt.show()

