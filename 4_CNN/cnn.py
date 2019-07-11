import argparse
import time
import numpy as np
import glob
import os
import psutil
import tensorflow as tf
import tensorflow.keras as k
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
    
    
def benchmark_model(model, list_path):
    # Use the model to run a few predictions and make sure the accuracy is there
    print("Loading the images in RAM")
    X = []
    Y_predict = []
    for index, img in enumerate(list_path):
        image = imread(img)/127.5 - 1
        X.append(image)
    X = np.array(X)  # Make it 4 D
    return model.predict(X)


def main():
    # load json and create model
    json_file = open(PATH_ARCHITECTURE, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = k.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(PATH_WEIGHTS)
    print("Loaded model from disk")
    
    inputPos = []
    inputNeg = []
    for folder in FOLDER_NAME:
        inputPos += [f for f in glob.glob(PATH_IMAGES+folder+'/'+folder+'_NN_crops/'+METHOD+'/*.jpg')]

    for folder in FOLDER_NAME:
        inputNeg += [f for f in glob.glob(PATH_IMAGES+folder+'/'+folder+'_NN_crops/Negatives/*.jpg')]
    list_benchmark = inputPos+inputNeg
    print(len(list_benchmark))
    nb_images = 5
    indexes = np.random.permutation(len(list_benchmark))
    random_images = [list_benchmark[i] for i in indexes[:nb_images]]
    t0 = time.perf_counter()
    benchmark_model(loaded_model, random_images)
    t1 = time.perf_counter()
    print("Infered {} images in {:.3f} s ({:.3f} images/s)".format(nb_images, t1-t0, nb_images/(t1-t0)))
    for i in range(10):
        index = np.random.randint(len(list_benchmark))
        single_image = np.array(imread(list_benchmark[index]), dtype=np.float64)/127.5 - 1
        single_image = single_image[np.newaxis, ...]
        t2 = time.perf_counter()
        loaded_model.predict(single_image)
        t3 = time.perf_counter()
        print("[np.float64] Infered 1 image in {:.3f} s ({:.3f} images/s)".format(t3-t2, 1/(t3-t2)))
    print()
    for i in range(10):
        index = np.random.randint(len(list_benchmark))
        single_image = np.array(imread(list_benchmark[index]), dtype=np.float32)/127.5 - 1
        single_image = single_image[np.newaxis, ...]
        t2 = time.perf_counter()
        loaded_model.predict(single_image)
        t3 = time.perf_counter()
        print("[np.float32] Infered 1 image in {:.3f} s ({:.3f} images/s)".format(t3-t2, 1/(t3-t2)))
    print()
    for i in range(10):
        index = np.random.randint(len(list_benchmark))
        single_image = np.array(imread(list_benchmark[index]), dtype=np.float16)/127.5 - 1
        single_image = single_image[np.newaxis, ...]
        t2 = time.perf_counter()
        loaded_model.predict(single_image)
        t3 = time.perf_counter()
        print("[np.float16] Infered 1 image in {:.3f} s ({:.3f} images/s)".format(t3-t2, 1/(t3-t2)))


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", type=str, help="model architecture", required=True)
    ap.add_argument("-w", "--weights", type=str, help="model weights", required=True)
    args = vars(ap.parse_args())
    PATH_ARCHITECTURE = args["model"]
    PATH_WEIGHTS = args["weights"]
    
        # Get some pictures to benchmark
    PATH_IMAGES = '/home/alex/Desktop/Helico/0_Database/RPi_import/'
    METHOD = 'cropsResizedToNn'
    #FOLDER_NAME = ['190622_201853', '190622_202211', '190624_200747', '190622_234007']
    FOLDER_NAME = ['190622_201853', '190622_202211', '190622_234007']
    
    main()
