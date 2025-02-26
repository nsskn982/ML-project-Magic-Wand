##
# @file dataaugmentationtest.py
#
# @brief Test module for verifying the functionality of the data augmentation module.

"""Test for data_augmentation.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import numpy as np

from data_augmentation import augment_data
from data_augmentation import time_wrapping


class TestAugmentation(unittest.TestCase):

  def test_time_wrapping(self):
    original_data = np.random.rand(10, 3).tolist()
    wrapped_data = time_wrapping(4, 5, original_data)
    self.assertEqual(len(wrapped_data), int(len(original_data) / 4 - 1) * 5)
    self.assertEqual(len(wrapped_data[0]), len(original_data[0]))

  def test_augment_data(self):
    original_data = [
        np.random.rand(128, 3).tolist(),
        np.random.rand(66, 2).tolist(),
        np.random.rand(9, 1).tolist()
    ]
    original_label = ["data", "augmentation", "test"]
    augmented_data, augmented_label = augment_data(original_data,
                                                   original_label)
    self.assertEqual(25 * len(original_data), len(augmented_data))
    self.assertIsInstance(augmented_data, list)
    self.assertEqual(25 * len(original_label), len(augmented_label))
    self.assertIsInstance(augmented_label, list)
    for i in range(len(original_label)):
      self.assertEqual(augmented_label[25 * i], original_label[i])


if __name__ == "__main__":
  unittest.main()
