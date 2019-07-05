# Run Monte Carlo simulations to find the best set of parameters for motion detection
# The ranking is done based on the f1_score for each param set

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
import os
import psutil
from sklearn.model_selection import ParameterGrid
import pickle

# Custom made files
from . import imageStabilizer


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


def centered_bbox(bbox):
    """
    Returns a centered bbox
    :param bbox: original bounding box
    :return: x, y are replaced by xc, yc
    """
    (x, y, w, h) = bbox
    (xc, yc) = (x + w // 2, y + h // 2)
    return xc, yc, w, h


def show_feed(s, thresh_feed, delta_frame, current_frame):
    """
    Selectively show the different layers of the image processing.
    :param s: string using binary representation to select the frames to display
    :param thresh_feed: current BW frame
    :param delta_frame: current gaussian frame
    :param current_frame: current color frame
    :return: void
    """
    if s[0] == '1':
        cv2.imshow("Thresh", thresh_feed)
    if s[1] == '1':
        cv2.imshow("current_frame Delta", delta_frame)
    if s[2] == '1':
        cv2.imshow("Security Feed", current_frame)


def import_stream(video_stream_path=None, verbose=False):
    """
    Connect to /dev/video0 or a given file.
    :param video_stream_path:
    :param verbose: more prints
    :return: stream, nb frames, width, height
    """
    # if the video argument is None, then we are reading from webcam
    if video_stream_path is None:
        video_stream = cv2.VideoCapture("/dev/video0")
        time.sleep(2.0)
    # otherwise, we are reading from a video file
    else:
        video_stream = cv2.VideoCapture(video_stream_path)

    # Stream properties
    nb_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get width and height of video stream
    frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if verbose:
        print("[INFO] Imported {} frames with shape x-{} y-{}".format(nb_frames, frame_width, frame_height))
    return video_stream, nb_frames, frame_width, frame_height


def cache_video(video_stream, method):
    """
    Loads in RAM a video_stream as a list or numpy array.
    :param video_stream: the local video file to cache
    :param method: currently, numpy array or list
    :return: the cached video
    """
    nb_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get width and height of video stream
    frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Populate a numpy array
    if method == 'numpy':
        vs_cache = np.zeros((nb_frames, frame_height, frame_width, 3), dtype=np.uint8)
        for i in range(nb_frames):
            vs_cache[i] = video_stream.read()[1]
    # Appends the frames in a list
    elif method == 'list':
        vs_cache = []
        while True:
            frame = video_stream.read()[1]
            if frame is not None:
                vs_cache.append(frame)
            else:
                break
    else:
        raise TypeError('This caching method is not supported')
    print("[INFO] Cached {} frames with shape x-{} y-{}".format(nb_frames, frame_width, frame_height))
    return vs_cache


def create_log(path, params):
    """
    Create a new log for this optimization run
    :param path: where the log will be stored
    :param params: params studied in this run. Whill make up the header
    :return: [TBR] first iteration - should be void
    """
    # Create a log and populate with the header. Wipes out previous logs with same name
    with open(path, 'w') as f:
        w = csv.writer(f)
        new_header = list(params.keys()) + ["real_fps", "avg_nb_boxes", "avg_nb_filtered_boxes", "avg_nb_heli_bbox",
                                            "precision", "recall", "f1_score"]
        w.writerow(new_header)
        print("Log header is now ", new_header)
    return new_header


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


def bb_intersection_over_union(box_a, box_b):
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


def main():
    """

    :return:
    """

    # Display best params or run the whole sim
    if DISPLAY_BEST_PARAMS:
        try:
            with open(PATH_BEST_PARAMS, 'rb') as f:
                best_params, best_recall, best_precision = pickle.load(f)
        except FileNotFoundError:
            print("[ERROR] Best param file not found.")
            raise
        iteration_dict = [best_params, best_recall, best_precision]  # Matches the dump order
    else:
        params = {
            'gaussWindow': range(3, 8, 2),
            'mgp': range(25, 26, 25),
            'residualConnections': range(1, 10, 2),
            'winSize': range(3, 4, 2),
            'maxLevel': range(5, 6, 3),
            'threshold_low': range(65, 66, 10),
            'threshold_gain': np.linspace(1.25, 1.26, 1),
            'sigma': np.linspace(0.1, 0.9, 5),
            'diffMethod': range(0, 1, 1),
            'dilationIterations': range(1, 8, 2),
            'skipFrame': range(0, 1, 1)
        }
        header = create_log(PATH_ALL_RESULTS, params)
        iteration_dict = ParameterGrid(params)

    video_stream, nb_frames, frame_width, frame_height = import_stream(VIDEO_STREAM_PATH)
    bbox_heli_ground_truth = import_bbox_heli(PATH_BBOX)  # Creates a dict


    # Min/Max area for the helicopter detection.
    # Min is difficult: it could be as small as a speck in the distance
    # Max is easier: you know how close it can possibly get (the helipad)
    min_area = 1
    if (
            (frame_width == 1920 and frame_height == 1080) or
            (frame_width == 3280 and frame_height == 2464)):
        binning = 1
    else:
        binning = 2 * 2
        print("[WARNING] Input resolution unusual. Camera sensor understood to be working with a 2x2 binning.")
    max_area = 200 * 200 / binning

    print("[INFO] Starting {} iterations".format(len(iteration_dict)))
    first_bbox = min(bbox_heli_ground_truth.keys())
    last_bbox = max(bbox_heli_ground_truth.keys())
    print("[INFO] Using bbox frames {} to {}".format(first_bbox, last_bbox))

    # Save the best results in memory
    if DISPLAY_BEST_PARAMS:
        counter_best_params = 0  # Used when displaying the 3 best runs
    highest_f1_score = 0
    highest_recall = 0
    highest_precision = 0
    vs2 = cache_video(video_stream, 'list')
    for sd in tqdm.tqdm(iteration_dict):
        # -------------------------------------
        # 1. RESET THE SIM DEPENDENT VARIABLES
        # -------------------------------------

        timing = {'Read frame': 0, 'Convert to grayscale': 0, 'Stabilize': 0, 'Double Gauss': 0, 'Abs diff': 0,
                  'Thresholding': 0, 'Dilation': 0, 'Count boxes': 0, 'Finalize': 0}
        nb_bbox = []  # Stores bbox data for a sim

        # Get ready to store residualConnections frames over and over
        previous_gray_frame = collections.deque(maxlen=sd['residualConnections'])
        # previous_gauss_frame = collections.deque(maxlen=sd['residualConnections'])

        img_stab = imageStabilizer.imageStabilizer(frame_width, frame_height, maxGoodPoints=sd['mgp'],
                                                   maxLevel=sd['maxLevel'], winSize=sd['winSize'])

        counter_skip_frame = sd['skipFrame']  # Go through the if statement the first time

        fps = FPS().start()
        # ----------------------------
        # 2. FRAME PROCESSING - GO THROUGH ALL FRAMES WITH A BBOX
        # -----------------------------

        for frame_number in range(nb_frames):

            t0 = time.perf_counter()
            # frame = vs.read()[1] # No cache
            frame = vs2[frame_number].copy()  # Prevents editing the original frames!
            t1 = time.perf_counter()
            # Skip all the frames that do not have a Bbox
            if frame_number < first_bbox:
                continue
            if frame_number > min(nb_frames - 2, last_bbox):
                break

            # 0. Skip frames - subsampling of FPS
            if counter_skip_frame < sd['skipFrame']:
                counter_skip_frame += 1
                continue
            else:
                counter_skip_frame = 0

            # Create a 0 based index that tracks how many bboxes we have gone through
            bbox_frame_number = frame_number - first_bbox  # Starts at 0, automatically incremented
            # Populate the deque with sd['residualConnections'] gray frames
            if bbox_frame_number < sd['residualConnections']:
                current_frame = frame
                current_gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                previous_gray_frame.append(current_gray_frame)
                continue

            # I. Grab the current in color space
            # t0=time.perf_counter()
            current_frame = frame

            # II. Convert to gray scale
            t2 = time.perf_counter()
            current_gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

            # III. Stabilize the image in the gray space with latest gray frame, fwd to color space
            # Two methods (don't chain them): phase correlation & optical flow
            t3 = time.perf_counter()
            if FLAG_PHASE_CORRELATION:
                """[TBR/Learning XP] Phase correlation is linearly faster as the area 
                to process is reduced, which is nice. However, if the expected translation is small
                (like ~ 1px) the results predictions can vary widely as the crop size is reduced.
                If the motion gets larger (even just 10 px), results between small and large crop match very accurately!
                plt.figure()
                plt.imshow(crop)
                plt.show()
                
                lCrop = 1000 # Large crop
                motion = 10 # controlled displacement
                for sCrop in range(100, 1001, 100):
                    #sCrop = 200
                    
                    t31 = time.perf_counter()
                    retvalSmall, response = cv2.phaseCorrelate(np.float32(current_gray_frame[:sCrop, :sCrop])/255.0, 
                    np.float32(current_gray_frame[motion:sCrop+motion, motion:sCrop+motion])/255.0)
                    t32 = time.perf_counter()
                    retvalLarge, response = cv2.phaseCorrelate(np.float32(current_gray_frame[:lCrop, :lCrop])/255.0, 
                    np.float32(current_gray_frame[motion:lCrop+motion, motion:lCrop+motion])/255.0)
                    t33 = time.perf_counter()
                    print("Full image is {} bigger and takes {} more time".format((lCrop/sCrop)**2, (t33-t32)/(t32-t31)))
                    print("xs {:.3f} xl {:.3f} Rx={:.3f} ys {:.3f} yl {:.3f} Ry={:.3f}".format(
                    retvalSmall[0], retvalLarge[0], retvalSmall[0]/retvalLarge[0], 
                    retvalSmall[1], retvalLarge[1], retvalSmall[1]/retvalLarge[1]))
            assert 1==0
            """
                pass
            if FLAG_OPTICAL_FLOW:
                m, current_gray_frame = img_stab.stabilizeFrame(previous_gray_frame[-1], current_gray_frame)
                current_frame = cv2.warpAffine(current_frame, m, (frame_width, frame_height))
            t4 = time.perf_counter()
            # current_frame = current_frame[int(cropPerc*frame_height):int((1-cropPerc)*frame_height),
            # int(cropPerc*frame_width):int((1-cropPerc)*frame_width)]
            # modif[bbox_frame_number-1] = img_stab.extractMatrix(m)

            # IV. Gaussian Blur
            # Done between current_frame and the grayFrame from residualConnections ago (first element in the deque)
            current_gauss_frame = cv2.GaussianBlur(current_gray_frame, (sd['gaussWindow'], sd['gaussWindow']), 0)
            previous_gauss_frame = cv2.GaussianBlur(previous_gray_frame[0], (sd['gaussWindow'], sd['gaussWindow']), 0)

            t5 = time.perf_counter()

            # V. Differentiation in the Gaussian space
            diff_frame = cv2.absdiff(current_gauss_frame, previous_gauss_frame)
            """[TBR/XP] absdiff strategies in the gaussian space"""
            """#Average of the absdiff with the current_frame for all residual connections (1toN strategy)
            # Basically, you do (1/m)*sum(|current_frame-previousGauss[i]|, i=0..N), N being dictated by residualConnections
            diff_frame = np.zeros(current_gauss_frame.shape)
            for gaussFrame in previous_gauss_frame:
                diff_frame += cv2.absdiff(current_gauss_frame, gaussFrame)
            diff_frame /= len(previous_gauss_frame)
            diff_frame = diff_frame.astype(np.uint8)  # float -> uint8
            # Best f1_score was about 0.32 (0.34 for simple absdiff(N, N-k))
            """
            """#Average of the absdiff between n and n-1 frame (NtoN-1 strategy)
            # Basically, you do (1/m)*sum(|previousGauss[i]-previousGauss[i+1]|, i=0..N-1), N being dictated by residualConnections
            # In that case, an array of the differences in the gaussian space could be cached to just pick 
            # what you want, but there is not enough RAM.
            diff_frame = np.zeros(current_gauss_frame.shape)
            for index in range(len(previous_gauss_frame)-1):
                diff_frame += cv2.absdiff(previous_gauss_frame[index], previous_gauss_frame[index+1])
            # Finish with current_gauss_frame and the latest previous_gauss_frame
            diff_frame += cv2.absdiff(current_gauss_frame, previous_gauss_frame[-1])
            diff_frame /= len(previous_gauss_frame)
            diff_frame = diff_frame.astype(np.uint8)  # float -> uint8
            # Best f1_score was about 0.29 (0.34 for simple absdiff(N, N-k))
            """
            t6 = time.perf_counter()
            if DISPLAY_FEED != '000':
                delta_frame = diff_frame.copy()

            # VI. BW space manipulations
            # diff_frame = cv2.threshold(diff_frame, sd['threshold'], 255, cv2.THRESH_BINARY)[1]

            # v = np.median(diff_frame)
            v = 127
            lower = int(max(0, (1.0 - sd['sigma']) * v))
            upper = int(min(255, (1.0 + sd['sigma']) * v))
            # diff_frame = cv2.Canny(diff_frame, sd['threshold_low'], sd['threshold_low']*sd['threshold_gain'])
            diff_frame = cv2.Canny(diff_frame, lower, upper)

            t7 = time.perf_counter()
            # dilate the thresholded image to fill in holes, then find contours
            if sd['diffMethod'] == 0:
                diff_frame = cv2.dilate(diff_frame, None, iterations=sd['dilationIterations'])
                t8 = time.perf_counter()
            elif sd['diffMethod'] == 1:
                diff_frame = cv2.morphologyEx(diff_frame, cv2.MORPH_OPEN, None)
            if DISPLAY_FEED != '000':
                thresh_feed = diff_frame.copy()
            cnts = cv2.findContours(diff_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            t8 = time.perf_counter()

            # Circle around the actual corner of the helicoBBox
            # Obtained via manual CSRT TRACKER
            # cv2.circle(current_frame, bbox_heli_ground_truth[bbox_frame_number], BBOX_ERROR, (0,0,255), -1)

            large_box = 0
            counter_bbox_heli = 0

            # VII. Process the BB and classify them
            x_gt, y_gt, w_gt, h_gt = bbox_heli_ground_truth[frame_number]  # Ground Truth data
            for c in cnts:
                # A. Filter out useless BBs
                # 1. if the contour is too small or too large, ignore it
                if cv2.contourArea(c) < min_area:
                    continue
                if cv2.contourArea(c) > max_area:
                    continue
                # compute the bounding box for the contour, draw it on the current_frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                
                # 2. Box partially out of the frame
                # if x < 0 or x+s > frame_width or y < 0 or y+s > frame_height: # Square box
                if x < 0 or x + w > frame_width or y < 0 or y + h > frame_height:
                    continue
                # 3. Box center in the PADDING area
                if not (PADDING < x + w // 2 < frame_width - PADDING and PADDING < y + h // 2 < frame_height - PADDING):
                    continue

                # B. Classify BBs - a large_box is a potential bbox_heli_ground_truth
                large_box += 1
                # Check if the corner is within range of the actual corner
                # That data was obtained by running a CSRT TRACKER on the helico

                # Classify bboxes based on their IOU with ground truth
                if bb_intersection_over_union((x, y, w, h), (x_gt, y_gt, w_gt, h_gt)) >= IOU:
                    counter_bbox_heli += 1
                    if DISPLAY_FEED == '001':  # Display positive bbox found in GREEN
                        cv2.putText(current_frame, "heli", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
                        cv2.rectangle(current_frame, (x, y), (x + w, y + h), GREEN, 2)
                else:
                    if DISPLAY_FEED == '001':  # Display negative bbox found in BLUE
                        cv2.putText(current_frame, "not heli", (x, y + h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 2)
                        cv2.rectangle(current_frame, (x, y), (x + w, y + h), BLUE, 2)
                    pass

            # C. Generate a square BB
            # cv2.rectangle(current_frame, (x, y), (x + s, y + s), GREEN, 2)
            # cv2.rectangle(current_frame, (x, y), (x + w, y + h), GREEN, 2)
            if DISPLAY_FEED == '001':
                cv2.rectangle(current_frame, (x_gt, y_gt), (x_gt + w_gt, y_gt + h_gt), RED, 2)  # Display ground truth in RED
            t9 = time.perf_counter()

            # VIII. draw the text and timestamp on the current_frame
            if DISPLAY_FEED != '000':
                if DISPLAY_BEST_PARAMS:
                    if counter_best_params == 0:
                        run = "best_f1_score"
                    elif counter_best_params == 1:
                        run = "best_recall"
                    elif counter_best_params == 2:
                        run = "best_precision"
                    else:
                        raise ValueError('There should only be 3 best results in the best_param log file')
                    cv2.putText(current_frame, "Current run: {} - f1_score: {:.3f} - recall: {:.3f} - precision: {:.3f}"
                                .format(run, sd['f1_score'], sd['recall'], sd["precision"]),
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)

                cv2.putText(current_frame, "BBoxes: {} found, {} heliBox"
                            .format(len(cnts), counter_bbox_heli),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
                # cv2.putText(current_frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, 30),
                # cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1) # Shows current date/time

                # IX. show the current_frame and record if the user presses a key
                show_feed(DISPLAY_FEED, thresh_feed, delta_frame, current_frame)
                key = cv2.waitKey(1) & 0xFF
                # if the `q` key is pressed, break from the loop
                if key == ord("q"):
                    break

            # X. Save frames & track KPI
            # The deque has a maxlen of residualConnections so the first-in will pop
            previous_gray_frame.append(current_gray_frame)
            # previous_gauss_frame.append(cv2.GaussianBlur(current_gray_frame, (sd['gaussWindow'], sd['gaussWindow']), 0))
            nb_bbox.append([len(cnts), large_box, counter_bbox_heli, 1 if counter_bbox_heli else 0])

            fps.update()
            t10 = time.perf_counter()
            if FLAG_DISPLAY_TIMING:
                new_timing = {'Read frame': t1 - t0, 'Convert to grayscale': t3 - t2, 'Stabilize': t4 - t3,
                              'Double Gauss': t5 - t4, 'Abs diff': t6 - t5, 'Thresholding': t7 - t6, 'Dilation': t8 - t7,
                              'Count boxes': t9 - t8, 'Finalize': t10 - t9}
                for key in timing.keys():
                    timing[key] += new_timing[key]

        # XI. Display results
        fps.stop()
        # vs.release()  # Done with going through this simulation, get ready for another pass
        if FLAG_DISPLAY_TIMING:
            print("Code profiling for various operations (in s):\n", timing)
        cv2.destroyAllWindows()

        # elapsed_time = fps.elapsed()
        average_fps = fps.fps()
        # real_fps = bbox_frame_number / elapsed_time
        # ratio = average_fps / real_fps
        # print("[INFO] elasped time: {:.2f}".format(elapsed_time))
        # print("[INFO] frame count: {}".format(bbox_frame_number))
        # print("[INFO] approx. FPS: {:.2f} \t real FPS: {:.2f}\tRatio (approx/real): {:.2f}".format(average_fps, real_fps, ratio))
        print("[INFO] FPS: {:.2f}".format(average_fps))

        # print(img_stab.detailedTiming())

        # Impact of stabilization on number of boxes
        bb = np.array(nb_bbox)
        bb = bb[1:]  # Delete first frame which is not motion controlled

        # KPI
        # per simulation
        # print(bb)
        avg_nb_boxes = np.mean(bb[:, 0])
        avg_nb_filtered_boxes = np.mean(bb[:, 1])
        avg_nb_heli_bbox = np.mean(bb[:, 2])
        # Precision: how efficient is the algo at rulling out irrelevant boxes?
        precision = avg_nb_heli_bbox / avg_nb_filtered_boxes  # Ratio of helibox/nb of boxes
        # Recall: how many frames had a positive heliBox? There should be one in each.
        recall = np.sum(bb[:, 3]) / bbox_frame_number  # Proportion of frames with helicopter

        # -----------------
        # SANITY CHECKS & f1_score
        # -----------------
        try:
            assert 0 < recall <= 1
            assert 0 < precision <= 1
            assert 0 <= avg_nb_heli_bbox <= avg_nb_filtered_boxes
            assert 0 <= avg_nb_filtered_boxes <= avg_nb_boxes
            f1_score = 2 / (1 / precision + 1 / recall)
        except AssertionError:
            print('[WARNING] KPIs out of bounds - set to 0')
            print("[WARNING] KPI: ", recall, precision, avg_nb_heli_bbox, avg_nb_filtered_boxes)
            recall, precision, avg_nb_heli_bbox, avg_nb_filtered_boxes = (0, 0, 0, 0)
            f1_score = 0

        """kpis
        plt.figure()
        plt.plot(bb[:, 0])
        plt.plot(bb[:, 1])
        plt.plot(bb[:, 2])
        plt.legend(("Number of boxes", "Boxes large enough", "Heli box"))
        titl = \
        "Boxes detected - av: {:.2f} - std: {:.2f} at {:.2f} FPS\n\
        Av Helibb per frame: {:.3f} - Ratio of helibb: {:.3f}\tFrame with heli: {:.3f} "\
        .format(\
        avg_nb_filtered_boxes, np.std(bb[:, 1]), real_fps, \
        avg_nb_heli_bbox, precision, recall\
        )
        plt.title(titl)
        plt.show()
        """
        # Display best params or append best results to log
        if DISPLAY_BEST_PARAMS:
            counter_best_params += 1
            print(sd)
        else:
            # Output results - parameters+kpis
            kpis = [average_fps, avg_nb_boxes, avg_nb_filtered_boxes, avg_nb_heli_bbox,
                    precision, recall, f1_score]
            # Warning: they are both int array of the same length so they can be added!
            sim_output = [sd[k] for k in params.keys()] + list(kpis)

            # Log the best f1_score, recall and precision
            if f1_score > highest_f1_score:
                highest_f1_score = f1_score
                best_params = sim_output
            if recall > highest_recall:
                highest_recall = recall
                best_recall = sim_output
            if precision > highest_precision:
                highest_precision = precision
                best_precision = sim_output

            with open(PATH_ALL_RESULTS, 'a') as f:
                w = csv.writer(f)
                w.writerow(sim_output)

    # XII. Wrap-up the search & output some logs for quick review
    # XII.1. Save the best param after inputting the header
    if not DISPLAY_BEST_PARAMS:
        with open(PATH_BEST_PARAMS, 'wb') as f:
            best_params = dict(zip(header, best_params))
            best_precision = dict(zip(header, best_precision))
            best_recall = dict(zip(header, best_recall))
            pickle.dump([best_params, best_recall, best_precision], f, protocol=pickle.HIGHEST_PROTOCOL)

        # XII.2. Pickle the params dict
        with open(PATH_PARAM_SPACE, 'wb') as f:
            pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)

        # XII.3. Final message!!
        print("Done. Highest f1_score: ", highest_f1_score)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# md_residual
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ap = argparse.ArgumentParser()  #
# ap.add_argument("-v", "--video", help="path to the video file", required=True)
# ap.add_argument("-bb", "--bounding_boxes", type=str, help="path to ground truth bounding boxes", required=True)
ap.add_argument("-bp", "--best_params", action='store_true', help="Display best overall/best precision/best recall")
ap.add_argument("-r", "--restart", type=int, help="iteration restart")
args = vars(ap.parse_args())

# ------------------
# Path constructions
# ------------------
# VIDEO_STREAM_PATH = args["video"]
# PATH_BBOX = args["bounding_boxes"]
# DISPLAY_BEST_PARAMS = args["best_params"]
VIDEO_STREAM_PATH = '/home/alex/Desktop/Helico/0_Database/RPi_import/190624_200747/190624_200747_helico_1920x1080_45s_25fps_FB.mp4'
PATH_BBOX = '/home/alex/Desktop/Helico/0_Database/RPi_import/190624_200747/190624_200747_extrapolatedBB.pickle'
DISPLAY_BEST_PARAMS = True

# Need to change the PATH_ALL_RESULTS name
FOLDER_PATH = os.path.split(VIDEO_STREAM_PATH)[0]
TIMESTAMP = os.path.split(VIDEO_STREAM_PATH)[1][:14]
PATH_ALL_RESULTS = os.path.join(FOLDER_PATH, TIMESTAMP + "MD_ParamSearch_All.csv")
PATH_BEST_PARAMS = os.path.join(FOLDER_PATH, TIMESTAMP + "MD_ParamSearch_Best.pickle")
PATH_PARAM_SPACE = os.path.join(FOLDER_PATH, TIMESTAMP + "MD_ParamSearch_Space.pickle")

# --------------
# CONSTANTS
# --------------
# [0]: threshold, [1]: absdiff, [2]: color frame
DISPLAY_FEED = '000' if not DISPLAY_BEST_PARAMS else '001'
# BBOX_ERROR is max error ratio to count a bbox as matching ground truth
# This applies to all axis (xc, yc, w, h)
IOU = 0.5
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
PADDING = 10  # px
FLAG_PHASE_CORRELATION = False  # This is too slow (>3x slower than mgp)
FLAG_OPTICAL_FLOW = False  # A bit better, but still way too slow
FLAG_DISPLAY_TIMING = False

if __name__ == '__main__':
    main()
