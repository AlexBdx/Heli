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
import video_tools as vt


def main():
    # I.3. Cache the video
    print("Caching video...")
    t0 = time.perf_counter()
    vs, nb_frames, frame_width, frame_height = vt.init.import_stream(PATH_VIDEO)

    vs_cache = vt.init.cache_video(vs, method='list')
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
    print("Caching done in {:.2f} s\tRAM used: {} Mb".format(t1-t0, vt.init.check_ram_use()))

    # I.4. Initialize various variables
    index = 0
    frame_change = False
    # Tracking related
    tracker = OPENCV_OBJECT_TRACKERS[TRACKER]()
    flag_tracker_active = False
    flag_success = False
    box = (0, 0, 0, 0)
    bbox_heli = []
    

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
                cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR['WHITE'], 2)
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
                cv2.putText(frame, text, (10, (i+1) * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR['RED'], 2)
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
        bbox_heli_ground_truth = dict()
        if len(bbox_heli) >= 2:
            for index in range(len(bbox_heli)-1):  # Not great coding but it works.
                # Case 1: next BBox is in the next frame
                if bbox_heli[index+1][0] == bbox_heli[index][0]+1:
                    bbox_heli_ground_truth[bbox_heli[index][0]] = bbox_heli[index][1]  # Store the bbox
                # Case 2: next BBox is a few frames away -> we extrapolate
                else:
                    (xs, ys, ws, hs) = bbox_heli[index][1]
                    (xf, yf, wf, hf) = bbox_heli[index+1][1]
                    n = bbox_heli[index+1][0] - bbox_heli[index][0]
                    for i in range(n):  # Extrapolate the BBoxes
                        bbox_heli_ground_truth[bbox_heli[index][0]+i] = (
                            round(xs + i*(xf-xs)/n), round(ys + i*(yf-ys)/n),
                            round(ws + i*(wf-ws)/n), round(hs + i*(hf-hs)/n))
        else:  # There are 0 or 1 frame -> we just write that
            for entry in bbox_heli:
                bbox_heli_ground_truth[entry[0]] = entry[1]
        pickle.dump(bbox_heli_ground_truth, f, protocol=pickle.HIGHEST_PROTOCOL)

    # IV. SANITY CHECK: MAKE SURE THE EXTRAPOLATION OVERLAYS WITH ORIGINAL VIDEO
    # IV.1. Import extrapolated bboxes
    with open(PATH_EXTRAPOLATED_BBOX, 'rb') as f:
        bbox_heli_ground_truth = pickle.load(f)

    # Rebuild the crop directory
    vt.bbox.clean_crop_directory(PATH_FOLDER)

    # IV.2.2 Replay the video and save the crops to file
    counter_crop = 0  # Used to increment picture name
    counter_bbox = 0  # Used for the skip function
    print("[INFO] Number of bbox on record:", len(bbox_heli_ground_truth))
    try:
        first_bbox = min(bbox_heli_ground_truth.keys())
        last_bbox = max(bbox_heli_ground_truth.keys())
    except ValueError:
        print("[ERROR] No bbox found. Aborting")
        raise
        
    for index, frame in enumerate(vs_cache):
        if not first_bbox < index < last_bbox:
            continue
        
        frame = frame.copy()
        try:
            (x, y, w, h) = bbox_heli_ground_truth[index]
            xc, yc = x + w//2, y + h//2

            if counter_bbox % (SKIP+1) == 0:
                # IV.2.2.1 First option: NN_SIZE crop
                crop = vt.bbox.nn_size_crop(frame, (x, y, w, h), NN_SIZE)
                out_path = os.path.join(PATH_CROPS_NN_SIZE, TIMESTAMP+str(counter_crop)+EXT)
                cv2.imwrite(out_path, crop)
                
                # IV.2.2.2 Second option: (square) bbox crop resized to NN_SIZE
                crop = vt.bbox.crop_resized_to_nn(frame, (x, y, w, h), NN_SIZE)
                out_path = os.path.join(PATH_CROP_RESIZED_TO_NN, TIMESTAMP+str(counter_crop)+EXT)
                cv2.imwrite(out_path, crop)
                
                # IV.2.2.3 Create a negative image - no helico
                # Take the crops from locations on the trajectory where the helicopter is not
                type_negative_crop = np.random.random()
                if type_negative_crop <= RATIO_NEGATIVE_ON_PATH:  # On trajectory crop
                    flag_success, crop = vt.bbox.on_trajectory_negative_crop(frame, (x, y, w, h), bbox_heli_ground_truth, NN_SIZE)
                    if not flag_success:
                        print("[WARNING] In {} (frame: {}): failed taking negative on trajectory.".format(os.path.split(PATH_VIDEO)[1][:13], index))
                        crop = vt.bbox.random_negative_crop(frame, (x, y, w, h), NN_SIZE)
                else:
                    crop = vt.bbox.random_negative_crop(frame, (x, y, w, h), NN_SIZE)
                out_path = os.path.join(PATH_NEGATIVES, TIMESTAMP+str(counter_crop)+EXT)
                cv2.imwrite(out_path, crop)
                
                # Increment crop counter
                counter_crop += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), COLOR['WHITE'], 2)
            counter_bbox += 1
        except KeyError:
            print("[ERROR] An incorrect dictionnary key was requested")
            raise
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

    PATH_VIDEO = args["video"]
    PATH_FOLDER = os.path.split(PATH_VIDEO)[0]
    TIMESTAMP = os.path.split(PATH_VIDEO)[1][:14]
    PATH_SOURCE_BBOX = os.path.join(PATH_FOLDER, TIMESTAMP+"sourceBB.pickle")
    PATH_EXTRAPOLATED_BBOX = os.path.join(PATH_FOLDER, TIMESTAMP+"extrapolatedBB.pickle")
    PATH_CROP_FOLDER = os.path.join(os.path.split(PATH_VIDEO)[0], TIMESTAMP+'NN_crops')
    PATH_CROPS_NN_SIZE = os.path.join(PATH_CROP_FOLDER, 'nnSizeCrops')
    PATH_CROP_RESIZED_TO_NN = os.path.join(PATH_CROP_FOLDER, 'cropsResizedToNn')
    PATH_NEGATIVES = os.path.join(PATH_CROP_FOLDER, 'Negatives')

    # I.2. Initialize a dictionary that maps strings to their corresponding
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }
    TRACKER = "csrt"
    # args["video"] = "/home/alex/Desktop/Helico/0_Database/RPi_import/"\
    # "190624_202327/190624_202327_helico_1920x1080_45s_25fps_T.mp4"
    COLOR = {'WHITE': (255, 255, 255), 'BLUE': (255, 0, 0), 'GREEN': (0, 255, 0), 'RED': (0, 0, 255), 'BLACK': (0, 0, 0)}
    
    SKIP = args["skip"]
    EXT = '.png'
    RATIO_NEGATIVE_ON_PATH = 0.5
    MAX_NEGATIVE_ATTEMPT = 50
    # NN_SIZE is a pair of even number otherwise next larger even number is used.
    NN_SIZE = tuple(int(s) for s in args["neural_network_size"].split('x'))
    try:
        assert NN_SIZE[0] % 2 == 0
        assert NN_SIZE[1] % 2 == 0
    except AssertionError:
        # Make NN_SIZE the nearest larger even number
        nn_size_0 = NN_SIZE[0] if NN_SIZE[0] % 2 == 0 else NN_SIZE[0]+1
        nn_size_1 = NN_SIZE[1] if NN_SIZE[1] % 2 == 0 else NN_SIZE[1]+1
        NN_SIZE = (nn_size_0, nn_size_1)
        print("neural_network_size needs to be a pair of even numbers. Input was adjusted to NN_SIZE = ({}, {})"
              .format(*NN_SIZE))
    window_name = "Video Feed"
    

    main()
