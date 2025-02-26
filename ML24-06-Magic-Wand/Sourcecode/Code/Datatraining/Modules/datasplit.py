##
# @file datasplit.py
#
# @brief Module for splitting the dataset into training, validation, and testing sets.
#
"""Mix and split data.

Mix different people's data together and randomly split them into train,
validation and test. These data would be saved separately under "/data".
It will generate new files with the following structure:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import random
from data_prepare import write_data


# Read data
def read_data(path):
  data = []  # pylint: disable=redefined-outer-name
  with open(path, "r") as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):  # pylint: disable=unused-variable
      dic = json.loads(line)
      data.append(dic)
  print("data_length:" + str(len(data)))
  return data


def split_data(data, train_ratio, valid_ratio):  # pylint: disable=redefined-outer-name
  """Splits data into train, validation and test according to ratio."""
  train_data = []  # pylint: disable=redefined-outer-name
  valid_data = []  # pylint: disable=redefined-outer-name
  test_data = []  # pylint: disable=redefined-outer-name
  num_dic = {"wing": 0, "ring": 0, "slope": 0, "negative": 0}
  for idx, item in enumerate(data):  # pylint: disable=unused-variable
    for i in num_dic:
      if item["gesture"] == i:
        num_dic[i] += 1
  print(num_dic)
  train_num_dic = {}
  valid_num_dic = {}
  for i in num_dic:
    train_num_dic[i] = int(train_ratio * num_dic[i])
    valid_num_dic[i] = int(valid_ratio * num_dic[i])
  random.seed(30)
  random.shuffle(data)
  for idx, item in enumerate(data):
    for i in num_dic:
      if item["gesture"] == i:
        if train_num_dic[i] > 0:
          train_data.append(item)
          train_num_dic[i] -= 1
        elif valid_num_dic[i] > 0:
          valid_data.append(item)
          valid_num_dic[i] -= 1
        else:
          test_data.append(item)
  print("train_length:" + str(len(train_data)))
  print("test_length:" + str(len(test_data)))
  return train_data, valid_data, test_data


if __name__ == "__main__":
  data = read_data("./data/complete_data")
  train_data, valid_data, test_data = split_data(data, 0.6, 0.2)
  write_data(train_data, "./data/train")
  write_data(valid_data, "./data/valid")
  write_data(test_data, "./data/test")
