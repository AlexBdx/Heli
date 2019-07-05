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
from psutil import Process
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
    print('RAM use: ', memory_use)


def nn_size_crop(image, window_size, bbox_center):
    """
    Crop an image with a set window. Handle crops near the edge of the frame with black PADDING.
    :param image: input image
    :param window_size: size of the cropping window
    :param bbox_center: center of the bb
    :return: window_size crop of image centeRED around bbox_center, potentiall black padded
    """
    xc, yc = bbox_center
    # Calculate how much PADDING is needed
    top = window_size[1]//2 - yc if yc - window_size[1]//2 < 0 else 0
    bottom = yc + window_size[1]//2 - FRAME_HEIGHT if yc + window_size[1]//2 > FRAME_HEIGHT else 0
    left = window_size[0]//2 - xc if xc - window_size[0]//2 < 0 else 0
    right = xc + window_size[0]//2 - FRAME_WIDTH if xc + window_size[0]//2 > FRAME_WIDTH else 0
    if top or bottom or left or right:
        # Add a black PADDING where necessary
        image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # DEBUG
    # 1. There shall be no negative param
    try:
        assert top >= 0
        assert bottom >= 0
        assert left >= 0
        assert right >= 0
        # 2. The final shape shall be window_size + 3 channels
        assert image.shape == (window_size[0], window_size[1], 3)
    except AssertionError:
        print("TBLR: ", top, bottom, left, right)
        print("Output image shape: ", image.shape, (window_size[0], window_size[1], 3))

    return image


def crop_negative(frame, nn_size, bbox_center):
    """
    Randomly crops an image with a nn_size window. The resulting crop has no intersection with the bb formed by nn_size and bbox_center.
    :param frame: input image
    :param nn_size: size of the cropping window
    :param bbox_center: center of the bb
    :return: image crop
    """
    # nn_size is width x height
    xc, yc = bbox_center
    FRAME_HEIGHT, FRAME_WIDTH, _ = frame.shape
    try:
        assert FRAME_WIDTH > 2*nn_size[0] and FRAME_HEIGHT > 2*nn_size[1]
    except AssertionError:
        print("[crop_negative] The input image is to small to crop a negative")
        return None
    xn = np.random.randint(FRAME_WIDTH)
    yn = np.random.randint(FRAME_HEIGHT)
    while (
            (xc-nn_size[0] < xn < xc+nn_size[0] and yc-nn_size[1] < yn < yc+nn_size[1])
            or xn > FRAME_WIDTH - nn_size[0]
            or yn > FRAME_HEIGHT - nn_size[1]
            ):
        """ [TBR]
        print(xn, yn)
        print(xc-nn_size[0] < xn < xc+nn_size[0], xn, xc)
        print(yc-nn_size[1] < yn < yc+nn_size[1], yn, yc)
        print(xn > FRAME_WIDTH - nn_size[0], FRAME_WIDTH - nn_size[0], xn)
        print(yn > FRAME_HEIGHT - nn_size[1], FRAME_HEIGHT - nn_size[1], yn)
        print("\n")
        """
        xn = np.random.randint(FRAME_WIDTH)
        yn = np.random.randint(FRAME_HEIGHT)
    return frame[yn:yn+nn_size[1], xn:xn+nn_size[0]]


def load_video(video_stream, method='generator'):
    # V190624
    # [TBM] Only one version should subsist with md_residual
    NB_FRAMES = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get width and height of video stream
    FRAME_WIDTH = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Populate a numpy array
    if method == 'numpy':
        vs = np.zeros((NB_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        for i in range(NB_FRAMES):
            vs[i] = video_stream.read()[1]
    # Appends the frames in a list
    if method == 'list':
        vs = []
        while True:
            frame = video_stream.read()[1]
            if frame is not None:
                vs.append(frame)
            else:
                break
    if method == 'generator':
        vs = video_stream

    print("[INFO] Imported {} frames with shape x-{} y-{}".format(NB_FRAMES, FRAME_WIDTH, FRAME_HEIGHT))
    return vs, NB_FRAMES, FRAME_WIDTH, FRAME_HEIGHT


# I. SETUP
# I.1. Preparing the arguments
# Videos: ../0_Database/RPi_import/
ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", type=str, help="path to input video file", requiRED=True)
ap.add_argument("-v", "--video", type=str, help="path to input video file")
ap.add_argument("-t", "--TRACKER", type=str, default="csrt", help="OpenCV object TRACKER type")
ap.add_argument("-s", "--skip", type=int, default=0, help="Proportion of BBox to save to file")
ap.add_argument("-n", "--neural_network_size", type=str, default='224x224', help="BBox crop size for NN input")
args = vars(ap.parse_args())

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

#args["video"] = "/home/alex/Desktop/Helico/0_Database/RPi_import/190624_202327/190624_202327_helico_1920x1080_45s_25fps_T.mp4"

def main():
    # I.3. Cache the video & create folder architecture
    print("Caching video...")
    t0 = time.perf_counter()
    VIDEO_PATH = args["video"]
    vs = cv2.VideoCapture(VIDEO_PATH)
    # I.3.1 Creating the folder architecture
    video_folder = os.path.split(VIDEO_PATH)[0]
    ts = os.path.split(VIDEO_PATH)[1][:14]
    source_bbox_path = os.path.join(video_folder, ts+"sourceBB.pickle")
    extrapolated_bbox_path = os.path.join(video_folder, ts+"extrapolatedBB.pickle")
    crop_folder = os.path.join(os.path.split(VIDEO_PATH)[0], ts+'NN_crops')
    nn_size_crops_folder = os.path.join(crop_folder, 'nnSizeCrops')
    crops_resized_to_nn_folder = os.path.join(crop_folder, 'cropsResizedToNn')
    negatives = os.path.join(crop_folder, 'negatives')

    # I.3.2 Cache the video
    vs_cache, NB_FRAMES, FRAME_WIDTH, FRAME_HEIGHT = load_video(vs, method='list')
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
    TRACKER = OPENCV_OBJECT_TRACKERS["csrt"]()
    flag_tracker_active = False
    flag_success = False
    box = (0, 0, 0, 0)
    BBOX_HELI = []
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
        print("neural_network_size needs to be a pair of even numbers. Input was adjusted to nn_size = ({}, {})".format(*nn_size))
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
                TRACKER = OPENCV_OBJECT_TRACKERS["csrt"]()  # reset
                flag_tracker_active = False
                flag_success = False
                box = (0, 0, 0, 0)
                print("Tracker deactivated!")
            else:  # not elif to let the user move the frame
                roi = cv2.selectROI(window_name, vs_cache[index], fromCenter=False, showCrosshair=True)
                TRACKER.init(vs_cache[index], roi)  # Init TRACKER
                flag_tracker_active = True
                print("Tracker activated! Selection is ", roi)

        # Update screen if there was a change
        if frame_change:  # The frame index has changed!
            # 1. Update the frame
            frame = vs_cache[index].copy()  # Don't edit the original cache!
            (H, W) = frame.shape[:2]
            info = [("Frame", index)]
            # 2. Manage the TRACKER
            if flag_tracker_active:
                (flag_success, box) = TRACKER.update(frame)  # Update TRACKER
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                info.append(("Box", (x, y, w, h)))
                if flag_success:
                    BBOX_HELI.append([index, (x, y, w, h)])
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
    with open(source_bbox_path, 'wb') as f:
        heli_bbox_source = dict()
        for entry in BBOX_HELI:
            heli_bbox_source[entry[0]] = entry[1]  # Store the tuples only
        pickle.dump(heli_bbox_source, f, protocol=pickle.HIGHEST_PROTOCOL)

    # III.2. Pickle the extrapolated BBoxes as a dict
    with open(extrapolated_bbox_path, 'wb') as f:
        heli_bbox_extrapolated = dict()
        if len(BBOX_HELI) >= 2:
            for index in range(len(BBOX_HELI)-1):  # Not great coding but it works.
                # Case 1: next BBox is in the next frame
                if BBOX_HELI[index+1][0] == BBOX_HELI[index][0]+1:
                    heli_bbox_extrapolated[BBOX_HELI[index][0]] = BBOX_HELI[index][1]  # Store the bbox
                # Case 2: next BBox is a few frames away -> we extrapolate
                else:
                    (xs, ys, ws, hs) = BBOX_HELI[index][1]
                    (xf, yf, wf, hf) = BBOX_HELI[index+1][1]
                    n = BBOX_HELI[index+1][0] - BBOX_HELI[index][0]
                    for i in range(n):  # Extrapolate the BBoxes
                        heli_bbox_extrapolated[BBOX_HELI[index][0]+i] = (round(xs + i*(xf-xs)/n), round(ys + i*(yf-ys)/n), round(ws + i*(wf-ws)/n), round(hs + i*(hf-hs)/n))
        else:  # There are 0 or 1 frame -> we just write that
            for entry in BBOX_HELI:
                heli_bbox_extrapolated[entry[0]] = entry[1]
        pickle.dump(heli_bbox_extrapolated, f, protocol=pickle.HIGHEST_PROTOCOL)


    # IV. SANITY CHECK: MAKE SURE THE EXTRAPOLATION OVERLAYS WITH ORIGINAL VIDEO
    # IV.1. Import extrapolated bboxes
    with open(extrapolated_bbox_path, 'rb') as f:
        heli_bbox_extrapolated = pickle.load(f)

    # IV.2. Replay the video with them & save non-skipped crops for NN
    # IV.2.1. Cleanup the crop directory (if it exists)
    if not os.path.isdir(crop_folder):
        os.mkdir(crop_folder)
    # IV.2.2. nn_size_crops_folder
    if os.path.isdir(nn_size_crops_folder):
        list_file = glob(os.path.join(nn_size_crops_folder, '*'))
        # print(os.path.join(nn_size_crops_folder,'*'))
        # print(list_file)
        if len(list_file) > 0:
            for f in list_file:
                os.remove(f)
    else:
        os.mkdir(nn_size_crops_folder)
    # IV.2.3. crops_resized_to_nn_folder
    if os.path.isdir(crops_resized_to_nn_folder):
        list_file = glob(os.path.join(crops_resized_to_nn_folder, '*'))
        # print(os.path.join(crops_resized_to_nn_folder,'*'))
        # print(list_file)
        if len(list_file) > 0:
            for f in list_file:
                os.remove(f)
    else:
        os.mkdir(crops_resized_to_nn_folder)
    # IV.2.4. negatives
    if os.path.isdir(negatives):
        list_file = glob(os.path.join(negatives, '*'))
        # print(os.path.join(negatives,'*'))
        # print(list_file)
        if len(list_file) > 0:
            for f in list_file:
                os.remove(f)
    else:
        os.mkdir(negatives)

    # IV.2.2 Replay the video and save the crops to file
    counter_crop = 0  # Used to increment picture name
    counter_bbox = 0  # Used for the skip function

    for index, frame in enumerate(vs_cache):
        frame = frame.copy()
        try:
            (x, y, w, h) = heli_bbox_extrapolated[index]
            xc, yc = x + w//2, y + h//2
            # Take the max but make it even (odd numbers are a mess with //2)
            s = max(w, h) if max(w, h) % 2 == 0 else max(w, h) + 1
            if counter_bbox % (skip+1) == 0:
                # IV.2.2.1 First option: nn_size crop
                # Limit the size of the crop
                x_start = max(0, xc - nn_size[0]//2)
                x_end = min(FRAME_WIDTH, xc + nn_size[0]//2)
                y_start = max(0, yc - nn_size[1]//2)
                y_end = min(FRAME_HEIGHT, yc + nn_size[1]//2)
                image = frame[y_start:y_end, x_start:x_end]
                image = nn_size_crop(image, nn_size, (xc, yc))
                out_path = os.path.join(nn_size_crops_folder, ts+str(counter_crop)+'.jpg')
                cv2.imwrite(out_path, image)
                # IV.2.2.2 Second option: (square) bbox crop resized to nn_size
                x_start = max(0, xc - s//2)
                x_end = min(FRAME_WIDTH, xc + s//2)
                y_start = max(0, yc - s//2)
                y_end = min(FRAME_HEIGHT, yc + s//2)
                image = frame[y_start:y_end, x_start:x_end]
                image = nn_size_crop(image, (s, s), (xc, yc))  # pad to (s, s)
                # Then only we resize to nn_size
                image = cv2.resize(image, nn_size)  # Resize to NN input size
                out_path = os.path.join(crops_resized_to_nn_folder, ts+str(counter_crop)+'.jpg')
                cv2.imwrite(out_path, image)
                # IV.2.2.3 Create a negative image - no helico
                image = crop_negative(frame, nn_size, (xc, yc))
                out_path = os.path.join(negatives, ts+str(counter_crop)+'.jpg')
                cv2.imwrite(out_path, image)
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
    main()