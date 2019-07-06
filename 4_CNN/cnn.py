# from imutils.video import video_stream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import numpy as np
import tqdm
import csv
import collections
import tqdm
import glob
import os
import psutil
from sklearn.model_selection import ParameterGrid
import pickle
import tensorflow as tf

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
    
    
def benchmark_model(model, list_path):
    # Use the model to run a few predictions and make sure the accuracy is there
    print("Loading the images in RAM")
    X = []
    for index, img in enumerate(tqdm.tqdm(list_path)):
        if index%250 == 0:
            print(check_ram_use())
        image = imread(img)/127.5 - 1
        X.append(image)
    for img_array in X:
        Y_predict.append(model.predict(img_array))
    return Y_predict


def main():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    
    # Get some pictures to benchmark
    root = 'DATA/RPi_import/'
    method = 'cropsResizedToNn'
    #folder_name = ['190622_201853', '190622_202211', '190624_200747', '190622_234007']
    folder_name = ['190622_201853', '190622_202211', '190622_234007']
    inputPos = []
    inputNeg = []
    for folder in folder_name:
        inputPos += [f for f in glob.glob(root+folder+'/'+folder+'_NN_crops/'+method+'/*.jpg')]

    for folder in folder_name:
        inputNeg += [f for f in glob.glob(root+folder+'/'+folder+'_NN_crops/Negatives/*.jpg')]
    list_benchmark = inputPos+inputNeg
    
    nb_img = 250
    indexes = np.random.permutation(len(list_benchmark))
    random_images = [list_benchmark[i] for i in indexes[:nb_images]]
    t0 = time.perf_counter()
    benchmark_model(loaded_model, random_images)
    t1 = time.perf_counter()
    print("Infered {} images in {:.3f} s ({:.3f}/s)".format(nb_img, t1-t0, nb_img/(t1-t0)))


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, help="model architecture", required=True)
    ap.add_argument("-w", "--weights", type=str, help="model weights", required=True)
    args = vars(ap.parse_args())
    
    main()
