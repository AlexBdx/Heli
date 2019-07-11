import os
import cv2
import numpy as np
import subprocess as sp

def clean_crop_directory(path_crop_folder):
    # path_crop_folder is a str
    
    # Delete crop folder and recreate an empty structure
    delete_crop_folder = ['rm', '-r', path_crop_folder]
    sp.run(delete_crop_folder)
    
    # Rebuild
    aug = os.join(path_crop_folder, 'Augmented_data')
    extracted = os.join(path_crop_folder, 'Extracted_helicopters')
    negatives = os.join(path_crop_folder, 'Negatives')
    crops_1 = os.join(path_crop_folder, 'cropsResizedToNn')
    crops_2 = os.join(path_crop_folder, 'nnSizeCrops')
    
    sp.run('mkdir', path_crop_folder)
    sp.run('mkdir', aug)
    sp.run('mkdir', extracted)
    sp.run('mkdir', negatives)
    sp.run('mkdir', crops_1)
    sp.run('mkdir', crops_2)


def nn_size_crop(crop, window_size, bbox_center, frame_shape):
    """
    Handle crops near the edge of the frame with black padding.
    :param crop: input crop, taken from a larger frame
    :param window_size: size of the cropping window
    :param bbox_center: center of the bb
    :param frame_shape: np.array.shape of the original frame
    :return: window_size crop centered around bbox_center, potentially black padded
    """
    xc, yc = bbox_center
    frame_height, frame_width, _ = frame_shape  # ignore channel number
    # Calculate how much padding is needed
    top = window_size[1]//2 - yc if yc - window_size[1]//2 < 0 else 0
    bottom = yc + window_size[1]//2 - frame_height if yc + window_size[1]//2 > frame_height else 0
    left = window_size[0]//2 - xc if xc - window_size[0]//2 < 0 else 0
    right = xc + window_size[0]//2 - frame_width if xc + window_size[0]//2 > frame_width else 0
    if top or bottom or left or right:
        # Add a black padding where necessary
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # DEBUG
    # 1. There shall be no negative param
    try:
        assert top >= 0
        assert bottom >= 0
        assert left >= 0
        assert right >= 0
        # 2. The final shape shall be window_size + 3 channels
        assert crop.shape == (window_size[0], window_size[1], 3)
    except AssertionError:
        print("TBLR: ", top, bottom, left, right)
        print("Output crop shape: ", crop.shape, (window_size[0], window_size[1], 3))

    return crop


def crop_negative(frame, nn_size, bbox_center):
    """
    Randomly crops an image with a nn_size window.
    The resulting crop has no intersection with the bb formed by nn_size and bbox_center.
    :param frame: input image
    :param nn_size: size of the cropping window
    :param bbox_center: center of the bb
    :return: image crop
    """
    # nn_size is width x height
    xc, yc = bbox_center
    frame_height, frame_width, _ = frame.shape
    try:
        assert frame_width > 2*nn_size[0] and frame_height > 2*nn_size[1]
    except AssertionError:
        print("[crop_negative] The input image is to small to crop a negative")
        return None
    xn = np.random.randint(frame_width)
    yn = np.random.randint(frame_height)
    while (
            (xc-nn_size[0] < xn < xc+nn_size[0] and yc-nn_size[1] < yn < yc+nn_size[1])
            or xn > frame_width - nn_size[0]
            or yn > frame_height - nn_size[1]
            ):
        """ [TBR]
        print(xn, yn)
        print(xc-nn_size[0] < xn < xc+nn_size[0], xn, xc)
        print(yc-nn_size[1] < yn < yc+nn_size[1], yn, yc)
        print(xn > frame_width - nn_size[0], frame_width - nn_size[0], xn)
        print(yn > frame_height - nn_size[1], frame_height - nn_size[1], yn)
        print("\n")
        """
        xn = np.random.randint(frame_width)
        yn = np.random.randint(frame_height)
    return frame[yn:yn+nn_size[1], xn:xn+nn_size[0]]
