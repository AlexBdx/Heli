import tensorflow as tf
import numpy as np
import os
import glob
import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp


print("[INFO] Using TF version", tf.__version__)

def load_model(path_architecture, path_weights):
    # load json and create model
    with open(path_architecture, 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(path_weights)
    
    return loaded_model
    
    
def search_path_for_images(root, method='cropsResizedToNn', extension='.png', verbose=False):
    # Returns the list of positive input
    # Detecting the positive and negative images
    
    path_crops = os.path.join(root, root[-13:]+'_NN_crops', method)
    path_negatives = os.path.join(root, root[-13:]+'_NN_crops', 'Negatives')
    print("[INFO] Searching ", path_crops)
    
    input_pos = []
    input_neg = []
    pos_images = [f for f in glob.glob(path_crops + '/*' + extension)]
    neg_images = [f for f in glob.glob(path_negatives + '/*' + extension)]
    if verbose:
      print("[INFO] Folder: {}\tPositive added: {:,}\t Negative added: {:,}"
          .format(root, len(pos_images), len(neg_images)))
    input_pos += pos_images
    input_neg += neg_images
    return input_pos, input_neg
    
    
def test_inference_speed(model, list_path, dtype=np.float16):
    # Test inference speed
    X = np.zeros((len(list_path), 224, 224, 3), dtype=dtype)
    timing = []
    for img in tqdm.tqdm(list_path):
        image = plt.imread(img, format='uint8')/127.5 - 1
        t0 = time.perf_counter()
        model.predict(image)
        timing.append(time.perf_counter() - t0)
    print("[INFO] Mean inference time: {:.3f} s\tStd deviation: {:.3f} s".format(np.mean(timing), np.std(timing)))
    
    
def test_model_on_folder(model, root, method, extension='.png', verbose=False):
    # Search for all images in a root folder and then predict them all
    # Return accuracy of the model
    # Create the list of paths
    DTYPE_IMAGES = np.float32
    DTYPE_LABELS = np.uint8
    input_pos, input_neg = search_path_for_images(root, method=method, extension=extension)
    
    path_X = input_pos + input_neg
    Y = [1]*len(input_pos) + [0]*len(input_neg)
    shuffled_indexes = np.random.permutation(len(path_X))
    path_X = [path_X[i] for i in shuffled_indexes]
    Y = [Y[i] for i in shuffled_indexes]
    Y = np.array(Y, dtype=np.uint8)
    
    # Import images
    X = []
    print("[INFO] Loading {} images in RAM".format(len(path_X)))
    with mp.Pool(os.cpu_count()) as pool:
        X, Y, scalers = zip(*pool.starmap(preprocess_image, zip(path_X, Y, [DTYPE_IMAGES]*len(path_X))))
    X = np.array(X, copy=False, dtype=DTYPE_IMAGES)
    Y = np.array(Y, copy=False, dtype=DTYPE_LABELS)
    
    
    # Infer images with model
    model_prediction = np.round(model.predict(X))
    score = np.abs(Y.reshape(-1, 1)-model_prediction)  # 0 is good, 1 is an issue
    accuracy = 1-np.sum(score, axis=0)[0]/len(X)

    if verbose:
        plt.figure()
        plt.bar(np.linspace(1, len(model_prediction), len(model_prediction)), score.reshape(-1))
        plt.title("Score repartition per predicted frame on {} (overall accuracy: {:.1f} %)"
        .format(os.path.split(root)[1], 100*accuracy))
        plt.xlabel("Frame with bbox number [-]")
        plt.ylabel("Binary score [-]")
        plt.show()
        
        # Display all the errors. Crashes below n_col errors for some reason
        counter = 0
        n_col, n_row = 10, 5
        fig, ax = plt.subplots(n_row, n_col, figsize=(2*n_col, 2*n_row))
        index_errors = [i for i in range(len(X)) if model_prediction[i] != Y[i]]
        print("INFO] There were {} errors".format(len(index_errors)))
        for i in range(n_row):
            for j in range(n_col):
                if counter>=len(index_errors):
                  ax[i, j].axis('off')
                  continue
                ax[i, j].imshow(invert_preprocessing(X[index_errors[counter]], scalers[index_errors[counter]]))
                ax[i, j].axis('off')
                guess = 'Helico' if model_prediction[index_errors[counter]] else 'No helico'
                if model_prediction[index_errors[counter]] != Y[index_errors[counter]]:
                  ax[i, j].set_title(guess.upper(), fontweight='bold', fontsize=14)
                else:
                  ax[i, j].set_title(guess)
                counter +=1
        plt.show()
        
    return accuracy
    
    
def preprocess_image(path, label, dtype):
    # Applies a few transformations to an image to normalize the data
    # image: uint8 array
    # Returns a dtype array of processed image

    # Import image
    image = plt.imread(path, format='uint8')  # Lightweight data load

    mean = np.mean(image)
    deviation = np.std(image)
    image = (image - mean)/deviation
    scaler = [mean, deviation]

    #assert image.dtype==dtype  # Sanity check
    return image.astype(dtype), label, scaler
    
def invert_preprocessing(image, scaler):
    image = image*scaler[1] + scaler[0]
    return image.astype(np.uint8)
