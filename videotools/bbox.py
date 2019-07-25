import pickle
import os
import cv2
import numpy as np
import subprocess as sp


def clean_crop_directory(path_folder):
    # path_folder is an absolute path str
    
    # Delete crop folder if it exists
    timestamp = os.path.split(path_folder)[1]
    path_crop_folder = os.path.join(path_folder, timestamp+'_NN_crops')
    if os.path.isdir(path_crop_folder):
        delete_crop_folder = ['rm', '-r', path_crop_folder]
        sp.run(delete_crop_folder)
    
    # Rebuild the folder structure
    aug = os.path.join(path_crop_folder, 'Augmented_data')
    extracted = os.path.join(path_crop_folder, 'Extracted_helicopters')
    negatives = os.path.join(path_crop_folder, 'Negatives')
    crops_1 = os.path.join(path_crop_folder, 'cropsResizedToNn')
    crops_2 = os.path.join(path_crop_folder, 'nnSizeCrops')
    
    sp.run(['mkdir', path_crop_folder])
    sp.run(['mkdir', aug])
    sp.run(['mkdir', extracted])
    sp.run(['mkdir', negatives])
    sp.run(['mkdir', crops_1])
    sp.run(['mkdir', crops_2])


def import_bbox_heli(heli_bb_file):
    """
    Read the pickle files containing the known location of the helicopter in the form of bb.
    :param heli_bb_file:
    :return: dict {frame: bbox tuple, ...}
    """
    with open(heli_bb_file, 'rb') as f:
        # r = csv.reader(f, delimiter=';')
        bbox_heli_ground_truth = pickle.load(f)
    return bbox_heli_ground_truth


def xywh_to_x1y1x2y2(bbox):
    """
    Convert a bounding box in the (x, y, w, h) format to the (x1, y1, x2, y2) format
    :param bbox: Bounding box
    :return: Converted bounding box
    """
    return bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]


def intersection_over_union(box_a, box_b):
    """
    Calculates IoU (Intersection over Union) for two boxes.
    Bounding boxes have to be submitted in the (x1, y1, x2, y2) format
    :param box_a: bounding box (order irrelevant)
    :param box_b: bounding box (order irrelevant)
    :return: 0 <= score <= 1
    """
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    
    # compute the area of intersection rectangle
    inter_area = max(0, xb - xa) * max(0, yb - ya)
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = inter_area / (box_a_area + box_b_area - inter_area)
    
    # return the intersection over union value
    return iou


def centered_bbox(bbox):
    """
    Returns a centered bbox
    :param bbox: original bounding box
    :return: x, y are replaced by xc, yc
    """
    (x, y, w, h) = bbox
    (xc, yc) = (x + w // 2, y + h // 2)
    return xc, yc, w, h


def nn_size_crop(frame, bbox, crop_size):
    """
    Handle crops near the edge of the frame with black padding.
    :param frame: input frame
    :param crop_size: tuple, size of output crop
    :param bbox: bbox to use for cropping, format is (x, y, w, h)
    :return: crop_size crop centered around bbox_center, potentially black padded
    """
    frame_height, frame_width, _ = frame.shape  # ignore channel number
    (x, y, w, h) = bbox
    xc, yc = x + w//2, y + h//2
    x_start = max(0, xc - crop_size[0]//2)
    x_end = min(frame_width, xc + crop_size[0]//2)
    y_start = max(0, yc - crop_size[1]//2)
    y_end = min(frame_height, yc + crop_size[1]//2)
    crop = frame[y_start:y_end, x_start:x_end]
    
    # Calculate how much padding is needed
    top = crop_size[1]//2 - yc if yc - crop_size[1]//2 < 0 else 0
    bottom = yc + crop_size[1]//2 - frame_height if yc + crop_size[1]//2 > frame_height else 0
    left = crop_size[0]//2 - xc if xc - crop_size[0]//2 < 0 else 0
    right = xc + crop_size[0]//2 - frame_width if xc + crop_size[0]//2 > frame_width else 0
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
        # 2. The final shape shall be crop_size + 3 channels
        assert crop.shape == (crop_size[0], crop_size[1], 3)
    except AssertionError:
        print("[ERROR] TBLR: ", top, bottom, left, right)
        print("[ERROR] Output crop shape: ", crop.shape, (crop_size[0], crop_size[1], 3))
        raise

    return crop
    

def crop_resized_to_nn(frame, bbox, crop_size):
    """
    Handle crops near the edge of the frame with black padding.
    :param frame: input frame
    :param crop_size: tuple, size of output crop
    :param bbox: bbox to use for cropping, format is (x, y, w, h)
    :return: crop_size crop centered around bbox_center, potentially black padded
    """
    frame_height, frame_width, _ = frame.shape  # ignore channel number
    (x, y, w, h) = bbox
    xc, yc = x + w//2, y + h//2
    s = max(w, h) if max(w, h) % 2 == 0 else max(w, h) + 1  # even only
    x_start = max(0, xc - s//2)
    x_end = min(frame_width, xc + s//2)
    y_start = max(0, yc - s//2)
    y_end = min(frame_height, yc + s//2)
    crop = frame[y_start:y_end, x_start:x_end]
    
    # Calculate how much padding is needed
    top = s//2 - yc if yc - s//2 < 0 else 0
    bottom = yc + s//2 - frame_height if yc + s//2 > frame_height else 0
    left = s//2 - xc if xc - s//2 < 0 else 0
    right = xc + s//2 - frame_width if xc + s//2 > frame_width else 0
    if top or bottom or left or right:
        # Add a black padding where necessary
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
    # Finally, resize to crop_size
    crop = cv2.resize(crop, crop_size)  # Resize to NN input size
    
    # DEBUG
    # 1. There shall be no negative param
    try:
        assert top >= 0
        assert bottom >= 0
        assert left >= 0
        assert right >= 0
        # 2. The final shape shall be crop_size + 3 channels
        assert crop.shape == (crop_size[0], crop_size[1], 3)
    except AssertionError:
        print("[ERROR] TBLR: ", top, bottom, left, right)
        print("[ERROR] Output crop shape: ", crop.shape, (crop_size[0], crop_size[1], 3))
        raise

    return crop



def random_negative_crop(frame, bbox, crop_size):
    """
    Randomly crops an image with a crop_size window.
    The resulting crop has no intersection with the bb formed by crop_size and bbox_center.
    :param frame: input image
    :param crop_size: size of the cropping window
    :param bbox_center: center of the bb
    :return: image crop
    """
    # crop_size is width x height
    (x, y, w, h) = bbox
    xc, yc = x + w//2, y + h//2
    frame_height, frame_width, _ = frame.shape
    
    # Verify that the frame is large enough to take random crops
    try:
        assert frame_width > 3*crop_size[0] and frame_height > 3*crop_size[1]
    except AssertionError:
        print("[ERROR] The frame is to small to crop a random negative")
        raise
    xn = np.random.randint(frame_width)
    yn = np.random.randint(frame_height)
    while (
            (xc-crop_size[0] < xn < xc+crop_size[0] and yc-crop_size[1] < yn < yc+crop_size[1])
            or xn > frame_width - crop_size[0]
            or yn > frame_height - crop_size[1]
            ):
        """ [TBR]
        print(xn, yn)
        print(xc-crop_size[0] < xn < xc+crop_size[0], xn, xc)
        print(yc-crop_size[1] < yn < yc+crop_size[1], yn, yc)
        print(xn > frame_width - crop_size[0], frame_width - crop_size[0], xn)
        print(yn > frame_height - crop_size[1], frame_height - crop_size[1], yn)
        print("\n")
        """
        xn = np.random.randint(frame_width)
        yn = np.random.randint(frame_height)
    return frame[yn:yn+crop_size[1], xn:xn+crop_size[0]]


def on_trajectory_negative_crop(frame, positive_bbox, ground_truth_bboxes, crop_size):
    
    
    frame_height, frame_width, _ = frame.shape  # ignore channel number
    flag_success = False
    attempt_counter = 0
    list_attempts = []
    
    try:
        first_bbox = min(ground_truth_bboxes.keys())
        last_bbox = max(ground_truth_bboxes.keys())
    except ValueError:
        print("[ERROR] No bbox found. Aborting")
        raise
    
    # Attempt to find a ground_truth_bbox without intersection, once cropped to crop_size,
    # with the current bbox
    for i in np.random.permutation(range(first_bbox, last_bbox+1)):  # Include the max too
        # Pick a ground_truth_box randomly in the list
        list_attempts.append(i)
        xi, yi, wi, hi = ground_truth_bboxes[i]
        
        #  Calculate the coordinates of the corresponding nn_size_crop
        xc, yc = xi + wi//2, yi + hi//2  # center of that bbox
        x_start = max(0, xc - crop_size[0]//2)
        x_end = min(frame_width, xc + crop_size[0]//2)
        y_start = max(0, yc - crop_size[1]//2)
        y_end = min(frame_height, yc + crop_size[1]//2)
        nnSize_bbox = (x_start, y_start, x_end, y_end)  # (x1, y1, x2, y2) format
        
        # Calculate IoU index between the current bbox and the random ground_truth_bbox 
        converted_current_gt = xywh_to_x1y1x2y2(positive_bbox)
        iou = intersection_over_union(nnSize_bbox, converted_current_gt)

        # If IoU == 0 then there is no intersection and that box can be kept
        attempt_counter += 1
        if iou == 0:
            flag_success = True
            break
        
    
    if flag_success:
        #print("[INFO] Found bbox {} after {} attempt(s)".format(i, attempt_counter))
        crop = frame[y_start:y_end, x_start:x_end]
        crop = nn_size_crop(frame, ground_truth_bboxes[i], crop_size)
    else:
        #print("[WARNING] nb attempts:", attempt_counter, "out of", len(ground_truth_bboxes))
        crop = np.zeros((crop_size[0], crop_size[1], 3))
    
    return flag_success, crop
