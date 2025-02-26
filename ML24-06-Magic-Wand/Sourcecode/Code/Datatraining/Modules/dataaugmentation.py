##
# @file dataaugmentation.py
#
# @brief This is a sample module for demonstrating data augmentation in Python.
#
# @section description_of_file Description
# Implements data augmentation techniques for enhancing the diversity of your dataset. Augmentation helps in training a more robust model by generating variations of the existing data.
# Data augmentation that will be used in data_load.py.
# To perform transformations on input data to create additional training samples.



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np


def time_wrapping(molecule, denominator, data):
  """Generate (molecule/denominator)x speed data."""
  tmp_data = [[0
               for i in range(len(data[0]))]
              for j in range((int(len(data) / molecule) - 1) * denominator)]
  for i in range(int(len(data) / molecule) - 1):
    for j in range(len(data[i])):
      for k in range(denominator):
        tmp_data[denominator * i +
                 k][j] = (data[molecule * i + k][j] * (denominator - k) +
                          data[molecule * i + k + 1][j] * k) / denominator
  return tmp_data


def augment_data(original_data, original_label):
  """Perform data augmentation."""
  new_data = []
  new_label = []
  for idx, (data, label) in enumerate(zip(original_data, original_label)):  # pylint: disable=unused-variable
    # Original data
    new_data.append(data)
    new_label.append(label)
    # Sequence shift
    for num in range(5):  # pylint: disable=unused-variable
      new_data.append((np.array(data, dtype=np.float32) +
                       (random.random() - 0.5) * 200).tolist())
      new_label.append(label)
    # Random noise
    tmp_data = [[0 for i in range(len(data[0]))] for j in range(len(data))]
    for num in range(5):
      for i in range(len(tmp_data)):
        for j in range(len(tmp_data[i])):
          tmp_data[i][j] = data[i][j] + 5 * random.random()
      new_data.append(tmp_data)
      new_label.append(label)
    # Time warping
    fractions = [(3, 2), (5, 3), (2, 3), (3, 4), (9, 5), (6, 5), (4, 5)]
    for molecule, denominator in fractions:
      new_data.append(time_wrapping(molecule, denominator, data))
      new_label.append(label)
    # Movement amplification
    for molecule, denominator in fractions:
      new_data.append(
          (np.array(data, dtype=np.float32) * molecule / denominator).tolist())
      new_label.append(label)
  return new_data, new_label
