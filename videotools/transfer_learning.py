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
    
    
def search_path_for_images(root, method, extension='.png', verbose=False):
    # Returns the path to positive and negative images contained in a folder
    path_crops = os.path.join(root, method)
    path_negatives = os.path.join(root, 'Negatives')
    input_pos = []
    input_neg = []
    pos_images = [f for f in glob.glob(path_crops + '/*.png')]
    neg_images = [f for f in glob.glob(path_negatives + '/*.png')]
    if verbose:
      print("[INFO] Folder: {}\tPositive added: {:,}\t Negative added: {:,}"
          .format(root, len(pos_images), len(neg_images)))
    input_pos += pos_images
    input_neg += neg_images
    return input_pos, input_neg
    
    
def load_path_images(list_folders, method, verbose=False):# Load the path to the images
    input_pos = []
    input_neg = []
    for folder in list_folders:
        PATH_FOLDER = folder
    if verbose:
        print("[INFO] Processing", PATH_FOLDER)
    # Check if the model 
    info = dict()
    try:
        with open(os.path.join(PATH_FOLDER, 'Info.txt')) as f:
            data = csv.DictReader(f)
            for line in data:
                info = line  # Only one line expected
    except FileNotFoundError:
        print("[WARNING] Info.txt file not found in {}. Skipping folder."
             .format(PATH_FOLDER))
        continue

    if len(HELICOPTER_MODELS):
        if info['Model'] not in HELICOPTER_MODELS:
            continue
    elif len(HELICOPTER_REGISTRATION):
        if info['Registration'] not in HELICOPTER_REGISTRATION:
            continue

    TIMESTAMP = os.path.split(PATH_FOLDER)[1]
    PATH_SOURCE_BBOX = os.path.join(PATH_FOLDER, TIMESTAMP+"_sourceBB.pickle")
    PATH_EXTRAPOLATED_BBOX = os.path.join(PATH_FOLDER, TIMESTAMP+"_extrapolatedBB.pickle")
    PATH_CROP_FOLDER = os.path.join(PATH_FOLDER, TIMESTAMP+'_NN_crops')
    #PATH_CROPS_NN_SIZE = os.path.join(PATH_CROP_FOLDER, 'nnSizeCrops')
    #PATH_CROP_RESIZED_TO_NN = os.path.join(PATH_CROP_FOLDER, 'cropsResizedToNn')
    PATH_CROPS = os.path.join(PATH_CROP_FOLDER, method)

    PATH_NEGATIVES = os.path.join(PATH_CROP_FOLDER, 'Negatives')
    PATH_EXTRACTED = os.path.join(PATH_CROP_FOLDER, 'Extracted_helicopters')
    #PATH_AUGMENTED = os.path.join(PATH_CROP_FOLDER, 'Augmented_data')

    pos_path, neg_path = search_path_for_images(PATH_CROP_FOLDER, method=method)
    input_pos += pos_path
    input_neg += neg_path
  
    print()
    print("[INFO] Method used:", method)
    print("[INFO] Detected {:,} positive images".format(len(input_pos)))
    print("[INFO] Detected {:,} negative images".format(len(input_neg)))

    if DTYPE_IMAGES == np.float16:
        multiplier = 2
    elif DTYPE_IMAGES == np.float32:
        multiplier = 4
    elif DTYPE_IMAGES == np.float64:
        multiplier = 8

    print("[INFO] Total: {:,} images\tSize: {:,} Mb"
        .format(len(input_pos)+len(input_neg),
                (len(input_pos)+len(input_neg))*224*224*3*multiplier//2**20))
    return input_pos, input_neg


def run_data_through_frozen_layers(X_train, nb_layers=0, verbose=False):
    # nb_layers is the nb of layers you want to use when running through the network
    # Import the original model without the last layer
    base_model = k.applications.mobilenet_v2.MobileNetV2(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    # /!\ Keep this for later /!\
    if nb_layers <= 0:
        nb_layers = len(base_model.layers)
    else:
        nb_layers = min(nb_layers, len(base_model.layers))
    # Freeze the desired number of layers
    for layer in base_model.layers[:nb_layers]:
        layer.trainable = False
    if verbose:
        print(base_model.summary())

    return base_model, base_model.predict(X_train)


def top_layers(verbose=False):
    # Add a global_average_pooling2d layer like in the original model
    #input_layer = tf.keras.engine.input_layer.Input(shape=input_shape)  # MIGHT NOT BE NEEDED
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    dropout = tf.keras.layers.Dropout(0.25)
    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))

    model = tf.keras.Sequential([
    global_average_layer,
    dropout,
    prediction_layer
    ])

    # this is the model we will train
    #model = Model(inputs=base_model.input, outputs=predictions)
    if verbose:
        print("[INFO] Top layers summary")
    model.summary()

    return model

    #original_model = rebuild_model()
    #original_model.summary()


def add_noise(image, intensity, noise_type='normal'):
    # Take a float image (in the [-1, 1] interval) as the input
    # Returns an image as a float
    dtype = image.dtype
    assert dtype==np.float16 or dtype==np.float32 or dtype==np.float64
    if noise_type == 'normal':
        noise = np.random.randn(224, 224, 3)  # This is N(0, 1)
    elif noise_type == 'uniform':
        noise = 2*np.random.rand(224, 224, 3) - 1  # Creates [-1, 1]
    else:
        print("[ERROR] This noise_type option does not exist")
        raise
    #print("[INFO] Average is ", np.mean(noise))
    #print("[INFO] Std is ", np.std(noise))
    image += intensity*noise  # Do not modify the image type
    image = np.clip(image, -1, 1)
    return image.astype(dtype)


def process_path_images(input_pos, input_neg, verbose=False):
    # First, control how many pictures are imported from each folder
    #path_X = inputPos + inputNeg
    #Y = [1]*len(inputPos) + [0]*len(inputNeg)

    path_X = [input_pos[i] for i in range(0, len(input_pos), STRIDE)] + [input_neg[i] for i in range(0, len(input_neg), STRIDE)]
    Y = [1 for i in range(0, len(input_pos), STRIDE)] + [0 for i in range(0, len(input_neg), STRIDE)]

    # Shuffle and split the arrays of paths so we don't shuffle actual images
    #trainSplit = 0.8
    #path_X_train, path_X_test, Y_train, Y_test = train_test_split(path_X, Y, test_size=0.2)


    #----OVERRIDE: limit the number of images to import
    #nb_images = min(32768//2, len(path_X))  # 2**15
    # Shuffle array
    trainSplit = 1
    indexes = np.random.permutation(range(len(path_X)))
    path_X_train = [path_X[i] for i in indexes]
    Y_train = [Y[i] for i in indexes]
    path_X_test = [path_X[i] for i in indexes[:round(0.1*len(indexes))]]
    Y_test = [Y[i] for i in indexes[:round(0.1*len(indexes))]]
    """[TBR]
    nb_train_images = round(trainSplit*NB_IMAGES)
    nb_test_images = round((1-trainSplit)*NB_IMAGES)
    path_X_train = path_X_train[:nb_train_images]
    path_X_test = path_X_test[:nb_test_images]
    Y_train = Y_train[:nb_train_images]
    Y_test = Y_test[:nb_test_images]
    """
    #------------------------------------------
    if DTYPE_IMAGES == np.float16:
        multiplier = 2
    elif DTYPE_IMAGES == np.float32:
        multiplier = 4
    elif DTYPE_IMAGES == np.float64:
        multiplier = 8

    if STRIDE != 1:
        print("[WARNING] Database is sub-sampled with a stride of", STRIDE)
        print("[INFO] Train ratio: {}\tTest ratio: {}"
            .format(round(trainSplit, ndigits=2), round(1-trainSplit, ndigits=2)))
        print("[INFO] X_train: {:,} paths\tX_test: {:,} paths"
            .format(len(path_X_train), len(path_X_test), len(path_X_train)+len(path_X_test)))
        print("[INFO] Total to import: {:,} images\tSize: {:,} Mb"
            .format(len(path_X_train)+len(path_X_test),
                    (len(path_X_train)+len(path_X_test))*224*224*3*multiplier//2**20))

    return path_X_train, path_X_test, Y_train, Y_test


def load_image_ram(path_X_train, path_X_test, Y_train, Y_test, verbose=False):

    # Import images in RAM
    print("[INFO] Loading X_train as {} images".format(DTYPE_IMAGES))

    # Multi-processed import of the data
    t0 = time.perf_counter()
    with mp.Pool(os.cpu_count()) as pool:
        X_train, Y_train, scalers = zip(*pool.starmap(preprocess_image, zip(path_X_train, Y_train, [DTYPE_IMAGES]*len(path_X_train))))
    X_train = np.array(X_train, copy=False, dtype=DTYPE_IMAGES)
    Y_train = np.array(Y_train, copy=False, dtype=DTYPE_LABELS)
    #print(Y_train)
    t1 = time.perf_counter()

    print("[INFO] Loaded {} images in {:.3f} s ({:.3f} Mb/s)"
          .format(len(X_train), t1-t0, (X_train.nbytes/2**20)/(t1-t0)))

    return X_train, Y_train, scalers
  
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
        X, Y, scalers = zip(*pool.starmap(load_and_preprocess_image, zip(path_X, Y, [DTYPE_IMAGES]*len(path_X))))
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
    
"""[TBR]
def load_and_preprocess_image(path, label, dtype):
    # Applies a few transformations to an image to normalize the data
    # image: uint8 array
    # Returns a dtype array of processed image

    # Import image
    image = plt.imread(path, format='uint8')  # Lightweight data load

    image = preprocess_image(image, dtype)

    #assert image.dtype==dtype  # Sanity check
    return image.astype(dtype), label, scaler
    
def preprocess_image(image, dtype):
    # Simple, direct normalization without anything else
    assert image.dtype == np.uint8
    mean = np.mean(image)
    deviation = np.std(image)
    image = (image - mean)/deviation
    #scaler = [mean, deviation]

    #assert image.dtype==dtype  # Sanity check
    return image.astype(dtype)
"""

def preprocess_image(path, label, dtype):
    # Applies a few transformations to an image to normalize the data
    # image: uint8 array
    # Returns a dtype array of processed image

    # Import image
    image = plt.imread(path, format='uint8')  # Lightweight data load

    # Scale to [-1, 1]
    image = image.astype(np.float64)/127.5 - 1  # Becomes a float64

    # Add N(0, 1) noise
    image = add_noise(image, 0.1)  # 0.01 is good for normal noise

    # Normalize data to follow N(0, 1)
    mean = np.mean(image)
    deviation = np.std(image)
    image = (image - mean)/deviation
    scaler = [mean, deviation]

    return image.astype(dtype), label, scaler


def invert_preprocessing(image, scaler):
  image = image*scaler[1] + scaler[0]
  image = 127.5*(image+1)  # [-1; 1] -> [0; 255]
  
  return image.astype(np.uint8)
