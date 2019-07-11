"""[INFO]
This script takes some images from my database and augments it.
However, I really do not like it as I have the option of creating more data from the frames
and that would be inherently be of better quality.
"""


import time
import glob
import os
import psutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import argparse
import tqdm
from matplotlib.image import imread, imsave


def check_ram_use():
    """
    Check and display the current RAM used by the script.
    :return: void
    """
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0]
    # print('RAM use: ', memory_use)
    return memory_use


def main():
    # Get some pictures to augment
    
    inputPos = []
    inputNeg = []
    for folder in FOLDER_NAMES:
        inputPos += [f for f in glob.glob(PATH_IMAGES+folder+'/'+folder+'_NN_crops/'+METHOD+'/*.jpg')]
        
    for folder in FOLDER_NAMES:
        inputNeg += [f for f in glob.glob(PATH_IMAGES+folder+'/'+folder+'_NN_crops/Negatives/*.jpg')]
    paths = inputPos+inputNeg
    # Sub sampling: collect 1 in SKIP images
    reduced_paths = [paths[x] for x in range(len(paths)) if x%SKIP == 0]
    print("[INFO] {} total images, {} selected".format(len(paths), len(reduced_paths)))
    
    print("[INFO] Importing images...")
    X = []
    for img in tqdm.tqdm(reduced_paths):
        X.append(imread(img))
    X = np.array(X)  # Make it 4D array
    
    aug = ImageDataGenerator(
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest")


    # construct the actual Python generator
    print("[INFO] Generating images...")
    imageGen = aug.flow(X, batch_size=BATCH_SIZE, save_to_dir=PATH_OUTPUT,
        save_prefix="aug", save_format="jpg")
        
    total = 0
    # loop over examples from our image data augmentation generator
    for image in tqdm.tqdm(imageGen):
        # increment our counter
        total += 1

        # if we have reached the specified number of examples, break
        # from the loop
        if total == TOTAL:
            break


if __name__ == '__main__':

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--total", type=int, default=128, help="# of samples to generate calculated as total//batch_size (rounded down most of the time)")
    ap.add_argument("-b", "--batch_size", type=int, default=32, help="# of images per batch - default is 32")
    args = vars(ap.parse_args())
    
    PATH_IMAGES = '/home/alex/Desktop/Helico/0_Database/RPi_import/'
    METHOD = 'cropsResizedToNn'
    FOLDER_NAMES = ['190622_201853', '190622_202211', '190622_234007']
    PATH_OUTPUT = 'sample/'
    
    SKIP = 25
    BATCH_SIZE = args["batch_size"]
    TOTAL = args["total"]//BATCH_SIZE
    
    main()
