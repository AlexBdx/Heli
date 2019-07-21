import argparse
import time
import numpy as np
import glob
import os
import psutil
import tensorflow as tf
import tensorflow.keras as k
import matplotlib.pyplot as plt
import video_tools as vt
import tqdm


def main():
    # Load model from file
    loaded_model = vt.transfer_learning.load_model(PATH_ARCHITECTURE, PATH_WEIGHTS)
    
    # Grab all the images
    input_pos = []
    input_neg = []
    for folder in FOLDER_NAME:
        path = os.path.join(ROOT, folder)
        # Test the score of a model on a specific folder
        #print(path)
        accuracy = vt.transfer_learning.test_model_on_folder(loaded_model, path, METHOD, extension=EXT, verbose=True)
        print('[INFO] Accuracy on {} is {:.1f} %'.format(os.path.split(folder)[1], 100*accuracy))
        print()
        
        """[Not needed today]
        # Build the list of images
        pos_path, neg_path = vt.transfer_learning.search_path_for_images(path, method=METHOD)
        input_pos += pos_path
        input_neg += neg_path

    # Create a subsample of images for testing inference speed
    list_benchmark = input_pos+input_neg
    nb_images = 100
    indexes = np.random.permutation(len(list_benchmark))
    random_images = [list_benchmark[i] for i in indexes[:nb_images]]
    vt.transfer_learning.test_inference_speed(loaded_model, random_images)
    """

    


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, help="model architecture", required=True)
    ap.add_argument("-w", "--weights", type=str, help="model weights", required=True)
    args = vars(ap.parse_args())
    PATH_ARCHITECTURE = args["model"]
    PATH_WEIGHTS = args["weights"]
    
        # Get some pictures to benchmark
    ROOT = '/home/alex/Desktop/Helico/0_Database/RPi_import/'
    #METHOD = 'cropsResizedToNn'
    METHOD = 'nnSizeCrops'
    #FOLDER_NAME = ['190622_201853', '190622_202211', '190624_200747', '190622_234007']
    FOLDER_NAME = ['190720_210148']
    #FOLDER_NAME = ['190720_210148', '190720_212410', '190720_213926', '190720_214342']

    EXT = '.jpg'
    
    main()
