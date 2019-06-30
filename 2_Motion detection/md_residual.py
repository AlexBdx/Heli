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
import imageStabilizer


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


def bounding_square(c):
    """
    Creates a centered square bounding box rather than a upper left corner, rectangle, one.
    :param c: opencv contour
    :return: center of the bb and characteristic dimension
    """
    (x, y, w, h) = cv2.boundingRect(c)
    (xc, yc) = (x + w // 2, y + h // 2)
    s = max(w, h)
    (xs, ys) = (xc - s // 2, yc - s // 2)
    # (xs, ys, s) = (1200, 1, 100) # Test square boundaries
    # (xs, ys, s) = (-1, 1, 100) # Test square boundaries
    # (xs, ys, s) = (1000, -1, 100) # Test square boundaries
    # (xs, ys, s) = (1000, 700, 100) # Test square boundaries
    return (xs, ys, s)


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
    NB_FRAMES = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get width and height of video stream
    FRAME_WIDTH = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if verbose:
        print("[INFO] Imported {} frames with shape x-{} y-{}".format(NB_FRAMES, FRAME_WIDTH, FRAME_HEIGHT))
    return video_stream, NB_FRAMES, FRAME_WIDTH, FRAME_HEIGHT


def cache_video(video_stream, method):
    """
    Loads in RAM a video_stream as a list or numpy array.
    :param video_stream: the local video file to cache
    :param method: currently, numpy array or list
    :return: the cached video
    """
    NB_FRAMES = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get width and height of video stream
    FRAME_WIDTH = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    FRAME_HEIGHT = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Populate a numpy array
    if method == 'numpy':
        vs_cache = np.zeros((NB_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        for i in range(NB_FRAMES):
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
    print("[INFO] Cached {} frames with shape x-{} y-{}".format(NB_FRAMES, FRAME_WIDTH, FRAME_HEIGHT))
    return vs_cache


def manage_log(PATH_LOG_FILE, params, restart, VERBOSE=False):
    """
    Create a new log for this optimization run
    :param PATH_LOG_FILE: where the log will be stored
    :param params: params studied in this run. Whill make up the header
    :param restart: [TBR] Restart from a given iteration number
    :param VERBOSE: print more stuff
    :return: [TBR] first iteration - should be void
    """
    # Start from scratch
    if restart is None:
        with open(PATH_LOG_FILE, 'w') as f:
            w = csv.writer(f)
            new_header = list(params.keys()) + ["real_fps", "avg_nb_boxes", "avg_nb_filtered_boxes", "avg_nb_heli_bbox",
                                               "percent_heli_total_filtered", "percent_frame_with_heli", "f1_score"]
            w.writerow(new_header)
            print("Log header is now ", new_header)
        ITERATION_START = 0
    # Restart case study from a given sim
    else:
        ITERATION_START = args["restart"]

    if VERBOSE:
        print("Starting at iteration {}".format(ITERATION_START))
    return ITERATION_START


def import_bbox_heli(heli_bb_file):
    """
    Read the pickle files containing the known location of the helicopter in the form of bb.
    :param heli_bb_file:
    :return: dict {frame: bbox tuple, ...}
    """
    with open(heli_bb_file, 'rb') as f:
        # r = csv.reader(f, delimiter=';')
        BBOX_HELI_GROUND_TRUTH = pickle.load(f)
    return BBOX_HELI_GROUND_TRUTH


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()  #
# ap.add_argument("-v", "--video", help="path to the video file", required=True)
# ap.add_argument("-bb", "--bounding_boxes", type=str, help="path to ground truth bounding boxes", required=True)
ap.add_argument("-r", "--restart", type=int, help="iteration restart")
args = vars(ap.parse_args())

# ------------------
# Import the file/stream
# ------------------
# video_stream_path = args["video"]
video_stream_path = '/home/alex/Desktop/Helico/0_Database/RPi_import/190622_234007/190622_234007_helico_1920x1080_75s_25fps_T.mp4'

video_stream, NB_FRAMES, FRAME_WIDTH, FRAME_HEIGHT = import_stream(video_stream_path)

# --------------------------
# ITERATION TABLE
# --------------------------

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
iteration_dict = ParameterGrid(params)

# PATH_BBOX = args["bounding_boxes"]
PATH_BBOX = '/home/alex/Desktop/Helico/0_Database/RPi_import/190622_234007/190622_234007_extrapolatedBB.pickle'
# Need to change the PATH_LOG_FILE name
NAME_LOG_FILE = os.path.split(PATH_BBOX)[1][:14] + "Detection_paramSearch.csv"  # Replace the (pickle) extension by a csv
PATH_LOG_FILE = os.path.join(os.path.split(PATH_BBOX)[0], NAME_LOG_FILE)
print("Path to log file", PATH_LOG_FILE)
ITERATION_START = manage_log(PATH_LOG_FILE, params, args["restart"])
BBOX_HELI_GROUND_TRUTH = import_bbox_heli(PATH_BBOX)  # Creates a dict

# modif = np.zeros((NB_FRAMES, 3))
# timing = np.zeros((NB_FRAMES, 4))
# MIN_AREA = args["min_area"]

# Hyperparam
# Stabilization
# mgp = 100
# Gaussian blurring
# gaussWindow = 5 # px of side
# assert gaussWindow%2 == 1
# MIN_AREA = 10

# --------------
# STATIC HYPERPARAMS
# --------------
# [0]: threshold, [1]: absdiff, [2]: color frame
DISPLAY_FEED = '000'
# BBOX_ERROR is max error ratio to count a bbox as matching ground truth
# This applies to all axis (xc, yc, w, h)
BBOX_ERROR = 0.5
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
VERBOSE = False
PADDING = 10  # px
FLAG_PHASE_CORRELATION = False  # This is too slow (>3x slower than mgp)
FLAG_OPTICAL_FLOW = False  # A bit better, but still a lot


# Min/Max area for the helicopter detection.
# Min is difficult: it could be as small as a speck in the distance
# Max is easier: you know how close it can possibly get (the helipad)
MIN_AREA = 1
if (
        (FRAME_WIDTH == 1920 and FRAME_HEIGHT == 1080) or
        (FRAME_WIDTH == 3280 and FRAME_HEIGHT == 2464)):
    BINNING = 1
else:
    BINNING = 2 * 2
    print("[WARNING] Input resolution unusual. Camera sensor understood to be working with a 2x2 BINNING.")
MAX_AREA = 200 * 200 / BINNING

print("[INFO] Starting {} iterations".format(len(iteration_dict)))
first_bbox = min(BBOX_HELI_GROUND_TRUTH.keys())
last_bbox = max(BBOX_HELI_GROUND_TRUTH.keys())
print("[INFO] Using bbox frames {} to {}".format(first_bbox, last_bbox))

vs2 = cache_video(video_stream, 'list')
for sd in tqdm.tqdm(iteration_dict):
    # -------------------------------------
    # 1. RESET THE SIM DEPENDENT VARIABLES
    # -------------------------------------
    # Reload the video from disk as we do not cache it
    vs, NB_FRAMES, FRAME_WIDTH, FRAME_HEIGHT = import_stream(video_stream_path)
    timing = {'Read frame': 0, 'Convert to grayscale': 0, 'Stabilize': 0, 'Double Gauss': 0, 'Abs diff': 0,
              'Thresholding': 0, 'Dilation': 0, 'Count boxes': 0, 'Finalize': 0}
    nb_bbox = []  # Stores bbox data for a sim

    # Get ready to store residualConnections frames over and over
    previous_gray_frame = collections.deque(maxlen=sd['residualConnections'])
    # previous_gauss_frame = collections.deque(maxlen=sd['residualConnections'])

    iS = imageStabilizer.imageStabilizer(FRAME_WIDTH, FRAME_HEIGHT, maxGoodPoints=sd['mgp'], maxLevel=sd['maxLevel'],
                                         winSize=sd['winSize'])
    
    counter_skip_frame = sd['skipFrame']  # Go through the if statement the first time

    fps = FPS().start()
    # ----------------------------
    # 2. FRAME PROCESSING - GO THROUGH ALL FRAMES WITH A BBOX
    # -----------------------------

    for frame_number in range(NB_FRAMES):

        t0 = time.perf_counter()
        # frame = vs.read()[1] # No cache
        frame = vs2[frame_number].copy()  # Prevents editing the original frames!
        t1 = time.perf_counter()
        # Skip all the frames that do not have a Bbox
        if frame_number < first_bbox:
            continue
        if frame_number > min(NB_FRAMES - 2, last_bbox):
            # print("Done with this sim. FirstBox {}, lastBox {}, frame_number {}".format(first_bbox, last_bbox, frame_number))
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
            # print(frame.shape)
            current_gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            previous_gray_frame.append(current_gray_frame)
            # previous_gauss_frame.append(cv2.GaussianBlur(current_gray_frame, (sd['gaussWindow'], sd['gaussWindow']), 0))
            continue

        # I. Grab the current in color space
        # t0=time.perf_counter()
        current_frame = frame

        # II. Convert to gray scale
        t2 = time.perf_counter()
        current_gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # III. Stabilize the image in the gray space with latest gray frame, fwd to color space
        # Two methods (dont' chain them): phase correlation & optical flow
        t3 = time.perf_counter()
        if FLAG_PHASE_CORRELATION:
            """[TBR/Learning XP] Phase correlation is linearly faster as the area 
            to process is reduced, which is nice. However, if the expected translation is small
            (like ~ 1px) the results predictions can vary widely as the crop size is reduced.
            If the motion gets larger (even just 10 px), the results between the small and large crop match very accurately!
            plt.figure()
            plt.imshow(crop)
            plt.show()
            
            lCrop = 1000 # Large crop
            motion = 10 # controlled displacement
            for sCrop in range(100, 1001, 100):
                #sCrop = 200
                
                t31 = time.perf_counter()
                retvalSmall, response = cv2.phaseCorrelate(np.float32(current_gray_frame[:sCrop, :sCrop])/255.0, np.float32(current_gray_frame[motion:sCrop+motion, motion:sCrop+motion])/255.0)
                t32 = time.perf_counter()
                retvalLarge, response = cv2.phaseCorrelate(np.float32(current_gray_frame[:lCrop, :lCrop])/255.0, np.float32(current_gray_frame[motion:lCrop+motion, motion:lCrop+motion])/255.0)
                t33 = time.perf_counter()
                print("Full image is {} bigger and takes {} more time".format((lCrop/sCrop)**2, (t33-t32)/(t32-t31)))
                print("xs {:.3f} xl {:.3f} Rx={:.3f} ys {:.3f} yl {:.3f} Ry={:.3f}".format(retvalSmall[0], retvalLarge[0], retvalSmall[0]/retvalLarge[0], retvalSmall[1], retvalLarge[1], retvalSmall[1]/retvalLarge[1]))
        assert 1==0
        """
            pass
        if FLAG_OPTICAL_FLOW:
            m, current_gray_frame = iS.stabilizeFrame(previous_gray_frame[-1], current_gray_frame)
            current_frame = cv2.warpAffine(current_frame, m, (FRAME_WIDTH, FRAME_HEIGHT))
        t4 = time.perf_counter()
        # current_frame = current_frame[int(cropPerc*FRAME_HEIGHT):int((1-cropPerc)*FRAME_HEIGHT), int(cropPerc*FRAME_WIDTH):int((1-cropPerc)*FRAME_WIDTH)]
        # modif[bbox_frame_number-1] = iS.extractMatrix(m)

        # IV. Gaussian Blur
        # Done between current_frame and the grayFrame from residualConnections ago (first element in the deque)
        current_gauss_frame = cv2.GaussianBlur(current_gray_frame, (sd['gaussWindow'], sd['gaussWindow']), 0)
        previous_gauss_frame = cv2.GaussianBlur(previous_gray_frame[0], (sd['gaussWindow'], sd['gaussWindow']),
                                              0)  # single deque case

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
        # In that case, an array of the differences in the gaussian space could be cached to just pick what you want, but there is not enough RAM.
        diff_frame = np.zeros(current_gauss_frame.shape)
        for index in range(len(previous_gauss_frame)-1):
            diff_frame += cv2.absdiff(previous_gauss_frame[index], previous_gauss_frame[index+1])
        diff_frame += cv2.absdiff(current_gauss_frame, previous_gauss_frame[-1]) # Finish with current_gauss_frame and the latest previous_gauss_frame
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
        # diff_frame = cv2.Canny(diff_frame, sd['threshold_low'], sd['threshold_low']*sd['threshold_gain'])  # WOW: much better!
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

        # Cirle around the actual corner of the helicoBBox
        # Obtained via manual CSRT TRACKER
        # cv2.circle(current_frame, BBOX_HELI_GROUND_TRUTH[bbox_frame_number], BBOX_ERROR, (0,0,255), -1)

        large_box = 0
        counter_bbox_heli = 0

        # VII. Process the BB and classify them
        xGT, yGT, wGT, hGT = BBOX_HELI_GROUND_TRUTH[frame_number]  # Ground Truth data
        for c in cnts:
            # A. Filter out useless BBs
            # 1. if the contour is too small or too large, ignore it
            if cv2.contourArea(c) < MIN_AREA:
                continue
            if cv2.contourArea(c) > MAX_AREA:
                continue
            # compute the bounding box for the contour, draw it on the current_frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            # (x, y, s) = bounding_square(c) # Great idea but we will make a square ourselves later

            # 2. Box partially out of the frame
            # if x < 0 or x+s > FRAME_WIDTH or y < 0 or y+s > FRAME_HEIGHT: # Square box
            if x < 0 or x + w > FRAME_WIDTH or y < 0 or y + h > FRAME_HEIGHT:
                continue
            # 3. Box center in the PADDING area
            # if not(PADDING < x+s//2 < FRAME_WIDTH-PADDING and PADDING < y+s//2 < FRAME_HEIGHT-PADDING): # Square box version
            if not (PADDING < x + w // 2 < FRAME_WIDTH - PADDING and PADDING < y + h // 2 < FRAME_HEIGHT - PADDING):
                continue

            # B. Classify BBs - a large_box is a potential BBOX_HELI_GROUND_TRUTH
            large_box += 1
            # Check if the corner is within range of the actual corner
            # That data was obtained by running a CSRT TRACKER on the helico

            # Is this bbox close enough to the ground truth bbox? Rectangular window
            if abs(x - xGT) < BBOX_ERROR * wGT and \
                    abs(y - yGT) < BBOX_ERROR * hGT and \
                    (1 - BBOX_ERROR) * wGT < w < (1 + BBOX_ERROR) * wGT and \
                    (1 - BBOX_ERROR) * hGT < h < (1 + BBOX_ERROR) * hGT:
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
            cv2.rectangle(current_frame, (xGT, yGT), (xGT + wGT, yGT + hGT), RED, 2)  # Display ground truth in RED
        t9 = time.perf_counter()

        # VIII. draw the text and timestamp on the current_frame
        if DISPLAY_FEED != '000':
            if bbox_frame_number % 2 == 0:
                cv2.putText(current_frame, "BBoxes: {} found, {} heliBox".format(len(cnts), counter_bbox_heli), (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
            # cv2.putText(current_frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1) # Shows current date/time

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
        if VERBOSE:
            new_timing = {'Read frame': t1 - t0, 'Convert to grayscale': t3 - t2, 'Stabilize': t4 - t3,
                         'Double Gauss': t5 - t4, 'Abs diff': t6 - t5, 'Thresholding': t7 - t6, 'Dilation': t8 - t7,
                         'Count boxes': t9 - t8, 'Finalize': t10 - t9}
            for key in timing.keys():
                timing[key] += new_timing[key]

    # XI. Display results
    fps.stop()
    vs.release()  # Done with going through this simulation, get ready for another pass
    if VERBOSE:
        print("Code profiling for various operations (in s):\n", timing)
    cv2.destroyAllWindows()

    elapsed_time = fps.elapsed()
    predicted_fps = fps.fps()
    real_fps = bbox_frame_number / elapsed_time
    ratio = predicted_fps / real_fps
    # print("[INFO] elasped time: {:.2f}".format(elapsed_time))
    # print("[INFO] frame count: {}".format(bbox_frame_number))
    # print("[INFO] approx. FPS: {:.2f} \t real FPS: {:.2f}\tRatio (approx/real): {:.2f}".format(predicted_fps, real_fps, ratio))
    print("[INFO] FPS: {:.2f}".format(real_fps))

    # print(iS.detailedTiming())

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
    percent_heli_total_filtered = avg_nb_heli_bbox / avg_nb_filtered_boxes  # Ratio of helibox/nb of boxes
    # Recall: how many frames had a positive heliBox? There should be one in each.
    percent_frame_with_heli = np.sum(bb[:, 3]) / bbox_frame_number  # Proportion of frames with heli

    # -----------------
    # SANITY CHECKS & f1_score
    # -----------------
    try:
        assert 0 < percent_frame_with_heli <= 1
        assert 0 < percent_heli_total_filtered <= 1
        assert 0 <= avg_nb_heli_bbox <= avg_nb_filtered_boxes
        assert 0 <= avg_nb_filtered_boxes <= avg_nb_boxes
        f1_score = 2 / (1 / percent_heli_total_filtered + 1 / percent_frame_with_heli)
    except:
        print('[WARNING] kpis out of bounds - set to 0')
        print("[WARNING] KPI: ", percent_frame_with_heli, percent_heli_total_filtered, avg_nb_heli_bbox, avg_nb_filtered_boxes)
        percent_frame_with_heli, percent_heli_total_filtered, avg_nb_heli_bbox, avg_nb_filtered_boxes = (0, 0, 0, 0)
        # print("[WARNING] Set to: ", percent_frame_with_heli, percent_heli_total_filtered, avg_nb_heli_bbox, avg_nb_filtered_boxes)
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
    avg_nb_heli_bbox, percent_heli_total_filtered, percent_frame_with_heli\
    )
    plt.title(titl)
    plt.show()
    """

    # Output results - parameters+kpis
    kpis = [real_fps, avg_nb_boxes, avg_nb_filtered_boxes, avg_nb_heli_bbox, percent_heli_total_filtered, percent_frame_with_heli,
            f1_score]
    # Warning: they are both int array of the same length so they can be added!

    sim_output = [sd[k] for k in params.keys()] + list(kpis)

    with open(PATH_LOG_FILE, 'a') as f:
        w = csv.writer(f)
        w.writerow(sim_output)

# XII. Wrap-up the search & output some logs for quick review
# XII.1. Find the best params based on the f1_score metric
with open(PATH_LOG_FILE, 'r') as f:
    r = csv.reader(f)
    next(r)  # Skip header
    highest_f1_score = 0
    for entry in r:  # Keep the score always last item
        if float(entry[-1]) > highest_f1_score:
            highest_f1_score = float(entry[-1])
            best_params = entry

# XII.2. Create a new log file with the best params
_ = manage_log(PATH_LOG_FILE[:-4] + '_best_params.csv', params, None)
with open(PATH_LOG_FILE[:-4] + '_best_params.csv', 'a') as f:
    w = csv.writer(f)
    w.writerow(best_params)

# XII.3. Pickle the params dict
with open(PATH_LOG_FILE[:-4] + '_paramSpace.pickle', 'wb') as f:
    pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)

# XII.4. Final message!!
print("Done. Highest f1_score: ", highest_f1_score)
