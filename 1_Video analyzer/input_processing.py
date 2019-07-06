# USAGE
# python opencv_object_tracking.py --video dashcam_boston.mp4

# import the necessary packages
# from imutils.video import VideoStream
# from imutils.video import FPS
import argparse
# import imutils
import time
import cv2
# import numpy as np
# import csv
import psutil
import os
# import copy
import pickle
from glob import glob
import numpy as np


# 0. DECLARATIONS
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


def load_video(video_stream, method='generator'):
    # V190624
    # [TBM] Only one version should subsist with md_residual
    nb_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get width and height of video stream
    frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Populate a numpy array
    if method == 'numpy':
        vs = np.zeros((nb_frames, frame_height, frame_width, 3), dtype=np.uint8)
        for i in range(nb_frames):
            vs[i] = video_stream.read()[1]
    # Appends the frames in a list
    elif method == 'list':
        vs = []
        flag_first_entry = True
        while True:
            frame = video_stream.read()[1]
            if flag_first_entry:
                previous_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flag_first_entry = False
            else:
                if frame is not None:
                    # vs.append(frame)  # color
                    # vs.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    vs.append(cv2.absdiff(previous_gray_frame, frame))
                    previous_gray_frame = frame
                else:
                    break
    elif method == 'generator':
        vs = video_stream
    else:
        raise ValueError('[ERROR] Method unknown in load_video')

    print("[INFO] Imported {} frames with shape x-{} y-{}".format(nb_frames, frame_width, frame_height))
    return vs, nb_frames, frame_width, frame_height


def main():
    # I.3. Cache the video
    print("Caching video...")
    t0 = time.perf_counter()
    vs = cv2.VideoCapture(VIDEO_PATH)

    vs_cache, nb_frames, frame_width, frame_height = load_video(vs, method='list')
    """[TBR]
    vs_cache = []
    while True:
        frame = vs.read()[1]
        if frame is not None:
            vs_cache.append(frame)
        else:
            break
    """
    t1 = time.perf_counter()
    print("Caching done in {:.2f} s\tRAM used: {} Mb".format(t1-t0, check_ram_use()//2**20))

    # I.4. Initialize various variables
    index = 0
    frame_change = False
    # Tracking related
    tracker = OPENCV_OBJECT_TRACKERS["csrt"]()
    flag_tracker_active = False
    flag_success = False
    box = (0, 0, 0, 0)
    bbox_heli = []
    skip = args["skip"]
    # nn_size is a pair of even number otherwise next larger even number is used.
    nn_size = tuple(int(s) for s in args["neural_network_size"].split('x'))
    try:
        assert nn_size[0] % 2 == 0
        assert nn_size[1] % 2 == 0
    except AssertionError:
        # Make nn_size the nearest larger even number
        nn_size_0 = nn_size[0] if nn_size[0] % 2 == 0 else nn_size[0]+1
        nn_size_1 = nn_size[1] if nn_size[1] % 2 == 0 else nn_size[1]+1
        nn_size = (nn_size_0, nn_size_1)
        print("neural_network_size needs to be a pair of even numbers. Input was adjusted to nn_size = ({}, {})"
              .format(*nn_size))
    window_name = "Video Feed"

    # II. PROCESS THE VIDEO
    # frame = vs_cache[index]
    # frame = imutils.resize(frame, width=500)
    cv2.imshow(window_name, vs_cache[index])
    while True:
        # wait for user's choice
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('a'):
            if index > 0:
                index -= 1
                frame_change = True
        if key == ord('d'):
            if index < len(vs_cache)-1:
                index += 1
                frame_change = True
        if key == ord('s'):
            if flag_tracker_active:  # Already tracking -> deactivate
                tracker = OPENCV_OBJECT_TRACKERS["csrt"]()  # reset
                flag_tracker_active = False
                flag_success = False
                box = (0, 0, 0, 0)
                print("tracker deactivated!")
            else:  # not elif to let the user move the frame
                roi = cv2.selectROI(window_name, vs_cache[index], fromCenter=False, showCrosshair=True)
                tracker.init(vs_cache[index], roi)  # Init tracker
                flag_tracker_active = True
                print("tracker activated! Selection is ", roi)

        # Update screen if there was a change
        if frame_change:  # The frame index has changed!
            # 1. Update the frame
            frame = vs_cache[index].copy()  # Don't edit the original cache!
            (H, W) = frame.shape[:2]
            info = [("Frame", index)]
            # 2. Manage the tracker
            if flag_tracker_active:
                (flag_success, box) = tracker.update(frame)  # Update tracker
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                info.append(("Box", (x, y, w, h)))
                if flag_success:
                    bbox_heli.append([index, (x, y, w, h)])
                    info.append(("Success", "Yes"))
                    flag_success = False
                else:
                    info.append(("Success", "No"))
            else:
                info.append(("Box", box))
                info.append(("Success", "Deactivated"))
            # 3. Display the new frame
            # print("Showing frame ", index)
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, (i+1) * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow(window_name, frame)
            # 4. Reset flag
            frame_change = False  # Back to normal

    cv2.destroyAllWindows()

    # III. SAVE BBOXES AND EXTRAPOLATED BBOXES TO FILE
    # III.1. Pickle the source BBoxes as a dict
    with open(PATH_SOURCE_BBOX, 'wb') as f:
        heli_bbox_source = dict()
        for entry in bbox_heli:
            heli_bbox_source[entry[0]] = entry[1]  # Store the tuples only
        pickle.dump(heli_bbox_source, f, protocol=pickle.HIGHEST_PROTOCOL)

    # III.2. Pickle the extrapolated BBoxes as a dict
    with open(PATH_EXTRAPOLATED_BBOX, 'wb') as f:
        heli_bbox_extrapolated = dict()
        if len(bbox_heli) >= 2:
            for index in range(len(bbox_heli)-1):  # Not great coding but it works.
                # Case 1: next BBox is in the next frame
                if bbox_heli[index+1][0] == bbox_heli[index][0]+1:
                    heli_bbox_extrapolated[bbox_heli[index][0]] = bbox_heli[index][1]  # Store the bbox
                # Case 2: next BBox is a few frames away -> we extrapolate
                else:
                    (xs, ys, ws, hs) = bbox_heli[index][1]
                    (xf, yf, wf, hf) = bbox_heli[index+1][1]
                    n = bbox_heli[index+1][0] - bbox_heli[index][0]
                    for i in range(n):  # Extrapolate the BBoxes
                        heli_bbox_extrapolated[bbox_heli[index][0]+i] = (
                            round(xs + i*(xf-xs)/n), round(ys + i*(yf-ys)/n),
                            round(ws + i*(wf-ws)/n), round(hs + i*(hf-hs)/n))
        else:  # There are 0 or 1 frame -> we just write that
            for entry in bbox_heli:
                heli_bbox_extrapolated[entry[0]] = entry[1]
        pickle.dump(heli_bbox_extrapolated, f, protocol=pickle.HIGHEST_PROTOCOL)

    # IV. SANITY CHECK: MAKE SURE THE EXTRAPOLATION OVERLAYS WITH ORIGINAL VIDEO
    # IV.1. Import extrapolated bboxes
    with open(PATH_EXTRAPOLATED_BBOX, 'rb') as f:
        heli_bbox_extrapolated = pickle.load(f)

    # IV.2. Replay the video with them & save non-skipped crops for NN
    # IV.2.1. Cleanup the crop directory (if it exists)
    if not os.path.isdir(PATH_CROP_FOLDER):
        os.mkdir(PATH_CROP_FOLDER)
    # IV.2.2. PATH_CROPS_NN_SIZE
    if os.path.isdir(PATH_CROPS_NN_SIZE):
        list_file = glob(os.path.join(PATH_CROPS_NN_SIZE, '*'))
        # print(os.path.join(PATH_CROPS_NN_SIZE,'*'))
        # print(list_file)
        if len(list_file) > 0:
            for f in list_file:
                os.remove(f)
    else:
        os.mkdir(PATH_CROPS_NN_SIZE)
    # IV.2.3. PATH_CROP_RESIZED_TO_NN
    if os.path.isdir(PATH_CROP_RESIZED_TO_NN):
        list_file = glob(os.path.join(PATH_CROP_RESIZED_TO_NN, '*'))
        # print(os.path.join(PATH_CROP_RESIZED_TO_NN,'*'))
        # print(list_file)
        if len(list_file) > 0:
            for f in list_file:
                os.remove(f)
    else:
        os.mkdir(PATH_CROP_RESIZED_TO_NN)
    # IV.2.4. PATH_NEGATIVES
    if os.path.isdir(PATH_NEGATIVES):
        list_file = glob(os.path.join(PATH_NEGATIVES, '*'))
        # print(os.path.join(PATH_NEGATIVES,'*'))
        # print(list_file)
        if len(list_file) > 0:
            for f in list_file:
                os.remove(f)
    else:
        os.mkdir(PATH_NEGATIVES)

    # IV.2.2 Replay the video and save the crops to file
    counter_crop = 0  # Used to increment picture name
    counter_bbox = 0  # Used for the skip function

    for index, frame in enumerate(vs_cache):
        frame = frame.copy()
        try:
            (x, y, w, h) = heli_bbox_extrapolated[index]
            xc, yc = x + w//2, y + h//2

            if counter_bbox % (skip+1) == 0:
                # IV.2.2.1 First option: nn_size crop
                # Limit the size of the crop
                x_start = max(0, xc - nn_size[0]//2)
                x_end = min(frame_width, xc + nn_size[0]//2)
                y_start = max(0, yc - nn_size[1]//2)
                y_end = min(frame_height, yc + nn_size[1]//2)
                crop = frame[y_start:y_end, x_start:x_end]
                crop = nn_size_crop(crop, nn_size, (xc, yc), frame.shape)
                out_path = os.path.join(PATH_CROPS_NN_SIZE, TIMESTAMP+str(counter_crop)+'.jpg')
                cv2.imwrite(out_path, crop)
                # IV.2.2.2 Second option: (square) bbox crop resized to nn_size
                s = max(w, h) if max(w, h) % 2 == 0 else max(w, h) + 1  # even only
                x_start = max(0, xc - s//2)
                x_end = min(frame_width, xc + s//2)
                y_start = max(0, yc - s//2)
                y_end = min(frame_height, yc + s//2)
                crop = frame[y_start:y_end, x_start:x_end]
                crop = nn_size_crop(crop, (s, s), (xc, yc), frame.shape)  # pad to (s, s)
                # Then only we resize to nn_size
                crop = cv2.resize(crop, nn_size)  # Resize to NN input size
                out_path = os.path.join(PATH_CROP_RESIZED_TO_NN, TIMESTAMP+str(counter_crop)+'.jpg')
                cv2.imwrite(out_path, crop)
                # IV.2.2.3 Create a negative image - no helico
                crop = crop_negative(frame, nn_size, (xc, yc))
                out_path = os.path.join(PATH_NEGATIVES, TIMESTAMP+str(counter_crop)+'.jpg')
                cv2.imwrite(out_path, crop)
                # Increment crop counter
                counter_crop += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            counter_bbox += 1
        except KeyError:
            pass
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


if __name__ == '__main__':
    # I. SETUP
    # I.1. Preparing the arguments
    # Videos: ../0_Database/RPi_import/
    ap = argparse.ArgumentParser()
    # ap.add_argument("-v", "--video", type=str, help="path to input video file", required=True)
    ap.add_argument("-v", "--video", type=str, help="path to input video file")
    ap.add_argument("-t", "--tracker", type=str, default="csrt", help="OpenCV object tracker type")
    ap.add_argument("-s", "--skip", type=int, default=0, help="Proportion of BBox to save to file")
    ap.add_argument("-n", "--neural_network_size", type=str, default='224x224', help="BBox crop size for NN input")
    args = vars(ap.parse_args())

    VIDEO_PATH = args["video"]
    VIDEO_FOLDER = os.path.split(VIDEO_PATH)[0]
    TIMESTAMP = os.path.split(VIDEO_PATH)[1][:14]
    PATH_SOURCE_BBOX = os.path.join(VIDEO_FOLDER, TIMESTAMP+"sourceBB.pickle")
    PATH_EXTRAPOLATED_BBOX = os.path.join(VIDEO_FOLDER, TIMESTAMP+"extrapolatedBB.pickle")
    PATH_CROP_FOLDER = os.path.join(os.path.split(VIDEO_PATH)[0], TIMESTAMP+'NN_crops')
    PATH_CROPS_NN_SIZE = os.path.join(PATH_CROP_FOLDER, 'nnSizeCrops')
    PATH_CROP_RESIZED_TO_NN = os.path.join(PATH_CROP_FOLDER, 'cropsResizedToNn')
    PATH_NEGATIVES = os.path.join(PATH_CROP_FOLDER, 'Negatives')

    # I.2. Initialize a dictionary that maps strings to their corresponding
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create
        # "kcf": cv2.TrackerKCF_create,
        # "boosting": cv2.TrackerBoosting_create,
        # "mil": cv2.TrackerMIL_create,
        # "tld": cv2.TrackerTLD_create,
        # "medianflow": cv2.TrackerMedianFlow_create,
        # "mosse": cv2.TrackerMOSSE_create
    }

    # args["video"] = "/home/alex/Desktop/Helico/0_Database/RPi_import/"\
    # "190624_202327/190624_202327_helico_1920x1080_45s_25fps_T.mp4"

    main()
