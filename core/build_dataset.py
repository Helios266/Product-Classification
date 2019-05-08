
__author__= 'Ho Tat Huy Cuong'
__email__ = 'huycuong.dt3.bkdn@gmail.com'
__version__= '0.0.1'

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import random
import argparse
import glob

#define input argument
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir","-d", help = "path to the dataset dictionary", type = str)
parser.add_argument("--output_dir","-o", help = "path to where storing the outputs", type = str)
parser.add_argument("--split_ratio","-s", help = "split ratio for take training data", type = float)
parser.add_argument("--output_size", help = "size of the output images", type = int, default=64)

if __name__ == "__main__":
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    split_ratio = args.split_ratio
    output_size = args.output_size

    assert os.path.isdir(dataset_dir), "The {} is not a directory".format(dataset_dir)
    if os.path.isdir(output_dir) == False:
        print("Create new folder at {}".format(output_dir))
        os.mkdir(output_dir)

    image_paths = glob.glob(os.path.join(dataset_dir, '*/*.jpg'))
    # .../xyz/asjdlasd.jpg
    random.seed(123)
    random.shuffle(image_paths)
    train_image_paths = image_paths[:int(split_ratio*len(image_paths))]
    test_image_paths = image_paths[int(split_ratio*len(image_paths)):]

    # train images
    if os.path.isdir(os.path.join(output_dir, 'train')) == False:
        os.mkdir(os.path.join(output_dir, 'train'))
    for image_path in train_image_paths:
        img = cv2.imread(image_path)
        resized_img = cv2.resize(img, (output_size, output_size))
        img_name = image_path.split('/')[-2].zfill(3) + '_' + image_path.split('/')[-1]
        cv2.imwrite(os.path.join(output_dir, 'train', img_name), resized_img)


    # test images
    if os.path.isdir(os.path.join(output_dir, 'test')) == False:
        os.mkdir(os.path.join(output_dir, 'test'))
    for image_path in test_image_paths:
        img = cv2.imread(image_path)
        resized_img = cv2.resize(img, (output_size, output_size))
        img_name = image_path.split('/')[-2].zfill(3) + '_' + image_path.split('/')[-1]
        cv2.imwrite(os.path.join(output_dir, 'test', img_name), resized_img)