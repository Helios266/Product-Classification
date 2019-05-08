
"""
Train model
"""
__author__ = "Ho Tat Huy Cuong"
__email__ = "huycuong.dt3.bkdn@gmail.com"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
import argparse
from model_fn import CNN
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
from input_fn import input_fn

#define input argument
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir","-d", help = "path to the dataset dictionary", type = str)
parser.add_argument("--image_size", "-s", help = "size of the output images", type = int, default=32)
parser.add_argument("--epoch", "-e", help = "number of epoch to train the model", type = int, default=5)
parser.add_argument("--continues", "-c", help = "continue training model or not", type = bool, default=False)

if __name__ == "__main__":
  args = parser.parse_args()

  dataset_dir = args.dataset_dir
  image_size = args.image_size
  continues = args.continues
  epoch = args.epoch

  # Load data
  inputs = input_fn(dataset_dir,image_size)
  X_train = inputs['X_train']
  y_train = inputs['y_train']
  X_test = inputs['X_test']
  y_test = inputs['y_test']
  print('Train data shape: ', X_train.shape)
  print('Train labels shape: ', y_train.shape)
  print('Validation data shape: ', X_test.shape)
  print('Validation labels shape: ', y_test.shape)

  if continues == False:
    model = CNN.my_awesome_model()
    # Train model
    model.fit(X_train, y_train, epochs=epoch)

    # Save entire model to a HDF5 file
    model.save('my_model.h5')

    # Check the accuracy of the model.
    loss, acc = model.evaluate(X_test, y_test)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))
  else:
    # Recreate the exact same model, including weights and optimizer.
    new_model = keras.models.load_model('my_model.h5')# Train new model
    new_model.fit(X_train, y_train, epochs=epoch)

    # Save entire model to a HDF5 file
    new_model.save('my_model.h5')
    # Check the accuracy of the new model.
    loss, acc = new_model.evaluate(X_test, y_test)
    print("Restored model, accuracy: {:5.2f}%".format(100*acc))
