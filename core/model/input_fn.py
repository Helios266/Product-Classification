# Input pipeline.

import cv2
import numpy as np
import os
import random


def input_fn(dataset_path, output_size):
    """ This function accepts the path of dataset, reads data, preprocesses data 
    and return it to the dictionary
    
    Arguments:
    - dataset_path: str
    - output_size: int

    Returns:
    - dic: list,

    Raise:
    - None.

    """
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    num_of_imgs_1 = 0
    num_of_imgs_2 = 0

    for root, dirs, files in os.walk(dataset_path):
        if root.split('/')[-1:][0] == 'train':
            for file in files:
                labels = file.split('_')[-3:][0]
                images = os.path.join(root, file)
                
                # read data
                images = cv2.imread(images)
                images = cv2.resize(images, (output_size, output_size))
                X_train.append(images)
                y_train.append(labels)
                num_of_imgs_1 +=1
        if root.split('/')[-1:][0] == 'test': 
            for file in files:
                labels = file.split('_')[-3:][0]
                images = os.path.join(root, file)
                
                images = cv2.imread(images)
                images = cv2.resize(images, (output_size, output_size))
                X_test.append(images)
                y_test.append(labels)
                num_of_imgs_2 +=1  

    
    X_train = np.array(X_train)
    y_train = np.array(y_train, dtype = np.uint8)
    X_test = np.array(X_test)
    y_test = np.array(y_test, dtype = np.uint8)

    # make a dictionary that contains all data.
    iter_ = ('X_train', 'y_train', 'X_test', 'y_test')
    dic = dict.fromkeys(iter_)
    dic['X_train'] = X_train
    dic['y_train'] = y_train
    dic['X_test'] = X_test
    dic['y_test'] = y_test

    return dic
