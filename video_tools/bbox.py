import pickle

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
