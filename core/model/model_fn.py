"""
Build the NN model.
"""

__author__ = "Ho Tat Huy Cuong"
__email__ = "huycuong.dt3.bkdn@gmail.com"


import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
import cv2
import os
import random
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

class CNN():

  # Returns a short sequential model
  def my_awesome_model():
      model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(
              filters=32, 
              padding='same', 
              kernel_size=3, 
              strides=1, 
              activation=tf.keras.activations.relu),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Conv2D(
              filters=64, 
              padding='same', 
              kernel_size=3, 
              strides=1, 
              activation=tf.keras.activations.relu),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.MaxPool2D(pool_size=2, strides=2),

      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(1024, activation=tf.keras.activations.relu),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Dropout(0.4),
      tf.keras.layers.Dense(200,activation=tf.keras.activations.softmax)  
      ])

      model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

      return model
      


