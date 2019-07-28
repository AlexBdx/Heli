""" Goals:
/!\ Only works for RGB images, binary classifier
- Import video & CNN
- Detect moving shape
- Confirm it is an helico by running the CNN
- Track the shape and confirm every so often that we are still good
"""

# from imutils.video import VIDEO_STREAM
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
from videotools import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def gaussian_blur(list_images, ws):
    blurred_images = []
    for image in list_images:
        assert image.dtype == np.uint8
        blurred_images.append(cv2.GaussianBlur(image, (ws, ws), 0))
    return blurred_images


def canny_contours(image, sigma):
    assert image.dtype == np.uint8
    lower = int(max(0, (1.0 - sigma) * 127))
    upper = int(min(255, (1.0 + sigma) * 127))
    return cv2.Canny(image, lower, upper)


def get_contours(image, iterations):
    assert image.dtype == np.uint8 
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, None, iterations=iterations)
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return imutils.grab_contours(cnts)


def generate_positive_crop(frame, roi_bbox, method, size=(224, 224)):
    # Takes a frame and a ROI
    # Crop using the specified method (nnSizeCrops or cropsResizedToNn)
    # Output the crop and the final bbox that was used for it
    assert frame.dtype == np.uint8
    x, y, w, h = roi_bbox
    xc, yc = x + w//2, y + h//2
    
    if method == 'nnSizeCrops':
        x_start = max(0, xc - size[0]//2)
        x_end = min(FRAME_WIDTH, xc + size[0]//2)
        y_start = max(0, yc - size[1]//2)
        y_end = min(FRAME_HEIGHT, yc + size[1]//2)
        #print(x_start, x_end, y_start, y_end)
        crop = frame[y_start:y_end, x_start:x_end]
        crop = bbox.nn_size_crop(crop, size, (xc, yc), frame.shape)
    elif method == 'cropsResizedToNn':
        s = max(w, h) if max(w, h) % 2 == 0 else max(w, h) + 1  # even only
        x_start = max(0, xc - s//2)
        x_end = min(FRAME_WIDTH, xc + s//2)
        y_start = max(0, yc - s//2)
        y_end = min(FRAME_HEIGHT, yc + s//2)
        crop = frame[y_start:y_end, x_start:x_end]
        crop = bbox.nn_size_crop(crop, (s, s), (xc, yc), frame.shape)  # pad to (s, s)
        # Then only we resize to size
        crop = cv2.resize(crop, size)  # Resize to NN input size
    else:
        print('[ERROR] Cropping method unknown')
        raise
    crop_bbox = (x_start, y_start, x_end-x_start, y_end-y_start)

    assert crop.shape == (size[0], size[1], 3)  # If this fails, you might be using a gray version
    assert len(crop_bbox) == 4
    assert type(crop_bbox) == type(tuple())
    return crop, crop_bbox


def infer_bbox(model, frame, bbox_roi, method):
    # Returns a uint8
    
    crop = bbox.nn_size_crop(frame, bbox_roi, NN_SIZE)
    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  # The CNN was trained on RGB data    
    pre_processed_crop, _, _ = transfer_learning.preprocess_image(crop, 1, DTYPE_IMAGES)
    single_sample = np.expand_dims(pre_processed_crop, 0)  # Make it a single inference
    #print(pre_processed_crop.shape)
    #print(single_sample.shape)
    prediction = model.predict(single_sample)[0][0] # Single sample, single class
    prediction = np.round(prediction).astype(np.uint8)
    
    return prediction, crop

def plot_confusion_matrix(Y, prediction, name=""):
    offset_h = 0.2
    offset_v = -0.05
    conf_mx = confusion_matrix(Y, prediction)
    print("[INFO] Confusion matrix: ", conf_mx)
    print()
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(conf_mx.shape[0]):
        for j in range(conf_mx.shape[1]):
            plt.text(j-offset_h, i-offset_v, str(conf_mx[i, j]), color='red', fontdict={"weight": "bold", "size": 20})
    plt.title("Confusion matrix: "+name, fontweight='bold', fontsize='14')
    # save_fig("confusion_matrix_plot", tight_layout=False)
    plt.show()


def hardware_metrics(t0, bbox_speed):
    info = dict()

    # Hardware
    info['cpu_usage'] = psutil.cpu_percent()
    info['cpu_count'] = os.cpu_count()
    info['ram_usage'] = core.check_ram_use()

    # Frame
    info['fps'] = HW_INFO_REFRESH_RATE/(time.perf_counter() - t0)

    # Tracker
    info['bbox_speed'] = int(HW_INFO_REFRESH_RATE*bbox_speed)

    return info


def real_time_metrics(frame_number, flag_tracker_active, flag_tracker_stationary):
    info = dict()

    # Frame
    info['frame_number'] = frame_number

    # Tracker
    info['flag_tracker_active'] = flag_tracker_active
    info['flag_tracker_stationary'] = flag_tracker_stationary

    return info


def display_frame(frame, hw_info, real_time_info, bbox_roi=None, width=0):
    # dynamic_info is a dict with all the info you need
    #print("[INFO] bbox_roi:", bbox_roi)
    if bbox_roi:
        x, y, w, h = bbox_roi
        if real_time_info['flag_tracker_active']:
            title = 'Helico'
            color = COLOR['RED']
        else:
            title = 'Motion'
            color = COLOR['BLUE']
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, title, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        pass  # Nothing was detected in this frame

    if len(hw_info):  # If empty, we only display the frame
        text_hardware = 'CPU: {} % of all {} cores | RAM: {} Mb'.format(hw_info['cpu_usage'], hw_info['cpu_count'], hw_info['ram_usage'])
        text_count = 'FPS: {:.1f} | Frame : {}/{}'.format(hw_info['fps'], real_time_info['frame_number'], NB_FRAMES)
        text_state = 'Tracker active: {} | Bbox speed: {} px/s'.format(real_time_info['flag_tracker_active'], hw_info['bbox_speed'])

        text_stationary = 'Tracker stationary: {}'.format(real_time_info['flag_tracker_stationary'])
        cv2.putText(frame, text_hardware, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR['WHITE'], 3)
        cv2.putText(frame, text_count, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR['WHITE'], 3)
        cv2.putText(frame, text_state, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR['WHITE'], 3)
        cv2.putText(frame, text_stationary, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR['WHITE'], 3)

    if width:
        frame = imutils.resize(frame, width=width)
    cv2.imshow(VIDEO_STREAM_PATH, frame)
    key = cv2.waitKey(1) & 0xFF
    flag_quit_program = True if key == ord('q') else False
    
    return flag_quit_program


def tracked_motion_analyzer(tracked_bbox):
    # Takes a deque (or list) of (x, y, w, h) bboxes
    # Return the the current center and the average speed over the logged data
    # Not the absolute most efficient but very readable with all the numpy steps
    tracked_centers = np.array(bbox.bbox_center(tracked_bbox))  # Get the centers
    
    # Calculate the average of the velocity norms
    overall_speed = np.diff(tracked_centers, axis=0)  # Get all the speed vectors
    overall_speed = np.mean(overall_speed, axis=0)  # Average speed over the last len(tracked_bbox) frames, in px/frame
    overall_speed = np.linalg.norm(overall_speed)  # Norm of the vector

    return tracked_centers[-1], overall_speed


def main():
    # Load the parameters for the motion detection
    try:
        with open(PATH_MD_PARAMS) as f:
            data = csv.DictReader(f)
            for line in data:
                params = line  # Header is read too
    except FileNotFoundError:
        print("[ERROR] Motion detection parameters file not found.")
        raise
    params['iou'] = float(params['iou'])/2
    params['gaussWindow'] = int(params['gaussWindow'])
    params['residualConnections'] = int(params['residualConnections'])
    params['dilationIterations'] = int(params['dilationIterations'])
    params['sigma'] = float(params['sigma'])
    print("[INFO] Motion detection params:\n", params)

    # Load the CNN
    try:
        loaded_model = transfer_learning.load_model(PATH_ARCHITECTURE, PATH_WEIGHTS)
    except FileNotFoundError:
        print("[ERROR] CNN model|weights file not found.")
        raise
        

    # Run through the video
    first_bbox = min(bbox_heli_ground_truth.keys())
    last_bbox = max(bbox_heli_ground_truth.keys())
    print("[INFO] Using bbox frames {} to {}".format(first_bbox, last_bbox))

    # Single core example
    timing = {'Read frame': 0, 'Convert to grayscale': 0, 'Stabilize': 0, 'Double Gauss': 0, 'Abs diff': [], 'Thresholding': 0, 'Dilation': 0, 'Count boxes': 0, 'Finalize': 0}
    nb_bbox = []  # Stores bbox data for a sim
    # Get ready to store residualConnections frames over and over


        
    # Create Tracker
    Tracker = OPENCV_OBJECT_TRACKERS["csrt"]()
    flag_tracker_active = False
    #fps = FPS().start()
    # Create an extractor to get the contours later on
    extractor = extract.extractor()
    # Create a deque for frame accumulation
    previous_gray_frame = collections.deque(maxlen=params['residualConnections'])
    # Track the results
    Y_prediction = []
    Y_test = []
    # Count global number of bboxes
    counter_bbox = 0
    # Start re-encoding
    if REENCODE:
        reencoded_video = cv2.VideoWriter(PATH_REENCODED_VIDEO, FOURCC, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
    t0 = time.perf_counter()  # Mostly used for first frame display
    
    # Create two deque to store tracking data on the CSRT bboxes
    tracked_bbox = collections.deque(maxlen=params['residualConnections'])
    flag_tracker_stationary = False  # Is the tracker's bbox moving?
    tracked_speed = 0
    # Number of chances before calling the tracking off
    patience = 0
    patience_stationary = 0
    for frame_number in range(NB_FRAMES):
        if frame_number % HW_INFO_REFRESH_RATE == 0:
            hw_info = hardware_metrics(t0, tracked_speed)
            t0 = time.perf_counter()  # Mostly used for first frame display
        current_frame = VIDEO_STREAM.read()[1]
        
        t1 = time.perf_counter()
        # Check the status of the tracker
        if flag_tracker_active:
            # ANALYZE PREVIOUS MOTION: HAS THE BBOX STALLED?
            flag_success, bbox_roi = Tracker.update(current_frame)
            bbox_roi = [int(value) for value in bbox_roi]  # cast to int
            tracked_bbox.append(bbox_roi)  # Deque
            
            if len(tracked_bbox) == params['residualConnections']:
                tracked_center, tracked_speed = tracked_motion_analyzer(tracked_bbox)
                #print('[INFO] Frame {} | Speed is {} px/s'.format(frame_number, int(FPS*tracked_speed)))
                if tracked_speed*FPS < MIN_TRACKED_SPEED:  # in px/s
                    # Tracked object has stopped. Could be hovering/landed or a tracker bug
                    patience_stationary += 1
                    if patience_stationary >= PATIENCE_STATIONARY:
                        flag_tracker_stationary = True
                else:
                    patience_stationary = max(0, patience_stationary-1)
            else:
                tracked_speed = 0


            if flag_tracker_stationary:
                # Run the md algo instead
                print("[INFO] Frame {} | Bbox is stationary".format(frame_number))
                pass
            else:
                # ROUTINE CHECK: IS THE TRACKER STILL LOOKING AT A HELI BBOX?
                if frame_number%CNN_CHECK_PERIOD == 0:
                    # Time to verify that the Tracker is still locked on the helico
                    prediction_tracked_bbox, crop = infer_bbox(loaded_model, current_frame, bbox_roi, METHOD)

                    if prediction_tracked_bbox == 1:  # All good, keep going
                        sw_info = real_time_metrics(frame_number, flag_tracker_active, flag_tracker_stationary)
                        flag_quit_program = display_frame(current_frame, hw_info, sw_info, bbox_roi=bbox_roi, width=1000)
                        print("[INFO] Frame {} | Bbox content was verified to contain a helico.".format(frame_number))
                        patience = max(0, patience-1)  # Decrement patience for this correct inference
                        # Re-encode the frame
                        if REENCODE:
                            reencoded_video.write(current_frame)
                        #t0 = time.perf_counter()
                        if flag_quit_program:
                            if REENCODE:
                                reencoded_video.release()
                            return
                        continue
                    elif prediction_tracked_bbox == 0:  # There is actually no helico in the tracked frame!
                        patience += 1  # Increment patience as we found a FP
                        if patience >= PATIENCE_TRACKER_CHECK:  # Reset tracker
                            print("[WARNING] Frame {} | After {} inferences, bbox content was classified as FP. Tracked is deactivated.".format(frame_number, patience))
                            Tracker = OPENCV_OBJECT_TRACKERS["csrt"]()  # Reset tracker
                            flag_tracker_active = False
                            tracked_bbox = collections.deque(maxlen=params['residualConnections'])
                            previous_gray_frame = collections.deque(maxlen=params['residualConnections'])
                            patience = 0
                            pass  # Engage the motion detection algo below
                        else:
                            print("[WARNING] Frame {} | Tracker bbox was infered {} times as FP during routine check.".format(frame_number, patience))
                            continue
                    else:
                        print("[ERROR] The model is supposed to be a binary classifier")
                        raise
            
                # Between checks from the CNN, go to the next frame
                else:
                    sw_info = real_time_metrics(frame_number, flag_tracker_active, flag_tracker_stationary)
                    flag_quit_program = display_frame(current_frame, hw_info, sw_info, bbox_roi= bbox_roi, width=1000)
                    # Re-encode the frame
                    if REENCODE:
                        reencoded_video.write(current_frame)
                    #t0 = time.perf_counter()
                    if flag_quit_program:
                        if REENCODE:
                            reencoded_video.release()
                        return
                    continue
        else: # The tracker is OFF
            pass

        t2 = time.perf_counter()
        # Switch the gray space
        current_gray_frame = cv2.cvtColor(current_frame.copy(), cv2.COLOR_RGB2GRAY)
        #print(len(previous_gray_frame))
        # Populate the deque with the params['residualConnections'] next gray frames
        if len(previous_gray_frame) < params['residualConnections']:
            previous_gray_frame.append(current_gray_frame)
            continue
        t4 = time.perf_counter()

        # Gaussian blur
        current_gauss_frame, previous_gauss_frame = gaussian_blur([current_gray_frame, previous_gray_frame[0]], params['gaussWindow'])
        
        t5 = time.perf_counter()
        
        # Differentiation
        diff_frame = cv2.absdiff(current_gauss_frame, previous_gauss_frame)
        t7 = time.perf_counter()
        
        # Canny
        canny_frame = canny_contours(diff_frame, params['sigma'])            
        t7 = time.perf_counter()
        
        # Morphological transformations
        morph_frame = cv2.dilate(canny_frame, None, iterations=params['dilationIterations'])
        #diff_frame = cv2.morphologyEx(diff_frame, cv2.MORPH_CLOSE, None, iterations=params['dilationIterations'])
        
        # Get the contours sorted by area (largest first)
        contours = extractor.image_contour(morph_frame, sorting='area', min_area=MIN_AREA)
        t8 = time.perf_counter()
        
        # Process the bbox that came out of the contours
        large_box = 0
        counter_failed_detection = 0
        md_bbox = None  # Bbox to display on the frame.
        for contour in contours:
            c = contour[0]
            # A. Filter out useless BBs
            # 1. if the contour is too small or too large, ignore it
            if cv2.contourArea(c) < MIN_AREA:
                continue
            if cv2.contourArea(c) > MAX_AREA:
                continue
            # compute the bounding box for the contour, draw it on the current_frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            
            # 2. Box partially out of the frame
            if x < 0 or x + w > FRAME_WIDTH or y < 0 or y + h > FRAME_HEIGHT:
                continue
            # 3. Box center is not in the PADDING area next to the edges of the frame
            if not (PADDING < x + w // 2 < FRAME_WIDTH - PADDING and PADDING < y + h // 2 < FRAME_HEIGHT - PADDING):
                continue

            # B. Classify BBs - a large_box is a potential bbox_heli_ground_truth
            large_box += 1
            counter_bbox += 1  # global counter
            
            # Infer bbox
            prediction, crop = infer_bbox(loaded_model, current_frame, (x, y, w, h), METHOD)
            
            # PREDICTION ANALYSIS
            if prediction == 1:  # Helico detected in the contours!
                # Have we previously flagged a stationary tracker?
                if flag_tracker_stationary:
                    # Is that heli close to the last place it was seen?
                    # If we are here, we have a very low average velocity on file so we have forgotten it all
                    distance_bboxes = np.linalg.norm(np.array(bbox.bbox_center((x, y, w, h))) - np.array(bbox.bbox_center(tracked_bbox)[-1]))
                    if distance_bboxes < STALLED_DISTANCE:
                        # Option 1: do nothing
                        print("[WARNING] Frame {} | Bbox was stationary but a helico was found.".format(frame_number))
                        md_bbox = bbox_roi
                        """
                        # Option 2: latch on this new bbox
                        # Then it stalled but the helico has slipped away. Latch on it
                        Tracker = OPENCV_OBJECT_TRACKERS["csrt"]()  # Reset the tracker but not the rest
                        Tracker.init(current_frame, (x, y, w, h)) # Re-init the tracker on that bbox
                        print("[WARNING] Frame {} | Bbox was stationary, locked on a nearby box {} px away.".format(frame_number, int(distance_bboxes)))
                        """
                    else:
                        # No helico found nearby. Cancel tracker and get back to md
                        print("[WARNING] Frame {} | Box was stationary w/o helico around, tracker deactivated!".format(frame_number))
                        Tracker = OPENCV_OBJECT_TRACKERS["csrt"]()
                        flag_tracker_active = False
                        tracked_bbox = collections.deque(maxlen=params['residualConnections'])
                    flag_tracker_stationary = False
                    break
                
                # No stationary tracker, so it is a new helico found
                else:
                    print("[INFO] Frame {} | Heli detected, start tracking.".format(frame_number))
                    Tracker.init(current_frame, (x, y, w, h))
                    flag_tracker_active = True
                    tracked_bbox.append((x, y, w, h))  # Log the coordinates of the tracked
                    break
            elif prediction == 0:
                counter_failed_detection += 1
                if counter_failed_detection >= MAX_FAILED_INFERENCES:
                    md_bbox = (x, y, w, h)  # Display the last bbox analyzed
                    #md_bbox = None  # Display the last bbox analyzed
                    print("[INFO] Frame {} | No helico found after infering the {} largest contours.".format(frame_number, MAX_FAILED_INFERENCES))
                    break  # Not time for another attempt. Improve the motion detection.
            else:
                raise ValueError("[ERROR] The model is supposed to be a binary classifier")

        t9 = time.perf_counter()
        # Update the deque
        previous_gray_frame.append(current_gray_frame)
        sw_info = real_time_metrics(frame_number, flag_tracker_active, flag_tracker_stationary)
        flag_quit_program = display_frame(current_frame, hw_info, sw_info, bbox_roi=md_bbox, width=1000)
        t10 = time.perf_counter()

        print("[INFO] case mgt: {:.3f}, md: {:.3f} ms, contour management: {:.3f} ms, display: {:.3f} ms".format(1000*(t2-t1), 1000*(t8-t2), 1000*(t9-t8), 1000*(t10-t9)))
        # Re-encode the frame
        if REENCODE:
            reencoded_video.write(current_frame)
        #t0 = time.perf_counter()
        if flag_quit_program:
            if REENCODE:
                reencoded_video.release()
            return

    

if __name__ == '__main__':
    # Gather the arguments

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", type=str, help="Video to analyze", required=True)
    ap.add_argument("-bb", "--bounding_boxes", type=str, help="Path to ground truth bounding boxes", required=True)
    ap.add_argument("-p", "--params", type=str, help="Parameters for the motion detection algo", required=True)
    ap.add_argument("-ma", "--model_architecture", type=str, help="CNN architecture (.json file)", required=True)
    ap.add_argument("-mw", "--model_weights", type=str, help="CNN weights (.h5 file)", required=True)
    args = vars(ap.parse_args())
    
    # Unpack arguments
    VIDEO_STREAM_PATH = args["video"]
    FOLDER_NAME = os.path.split(VIDEO_STREAM_PATH)[1][:13]
    PATH_BBOX = args["bounding_boxes"]
    PATH_MD_PARAMS = args["params"]
    PATH_ARCHITECTURE = args["model_architecture"]
    PATH_WEIGHTS = args["model_weights"]


    """
    # To be run by PyCharm
    VIDEO_STREAM_PATH = "../0_Database/RPi_import/190622_201853/190622_201853_helico_1920x1080_45s_25fps_L.mp4"
    FOLDER_NAME = os.path.split(VIDEO_STREAM_PATH)[1][:13]
    PATH_BBOX = "../0_Database/RPi_import/190622_201853/190622_201853_extrapolatedBB.pickle"
    PATH_MD_PARAMS = "md_params.csv"
    PATH_ARCHITECTURE = "../4_CNN/190727_104133/190727_104133.json"
    PATH_WEIGHTS = "../4_CNN/190727_104133/190727_104133.h5"
    """
    
    # Offline method - load a video & its bboxes
    VIDEO_STREAM, NB_FRAMES, FRAME_WIDTH, FRAME_HEIGHT = core.import_stream(VIDEO_STREAM_PATH)
    bbox_heli_ground_truth = bbox.import_bbox_heli(PATH_BBOX)  # Creates a dict
    FPS = 25  # This is assumed. Not sure I can read that info easily
    
    # Set some global constants
    METHOD = 'nnSizeCrops'
    DTYPE_IMAGES = np.float32  # Can the RPi run TF lite in float16 and be faster?
    MIN_AREA = 100
    MAX_AREA = 112*112
    PADDING = 1  # Exclude the bboxes from contours that are less than PADDING px from a frame edge
    CNN_CHECK_PERIOD_S = 1  # How often, in s, do you check the tracker with the CNN?
    CNN_CHECK_PERIOD = CNN_CHECK_PERIOD_S*FPS
    HW_INFO_REFRESH_RATE = 10
    MAX_FAILED_INFERENCES = 1
    NN_SIZE = (224, 224)
    #FONTSIZE =
    
    PATIENCE_TRACKER_CHECK = 2  # Number of negative prediction of the tracked bbox before calling the tracking off.
    PATIENCE_STATIONARY = 2  # Number of times the bboxes can be seen as stationary before we id them
    MIN_TRACKED_SPEED = 5  # In px/s
    STALLED_DISTANCE = 100  # In px
    
    # Re-encode the video to save the result
    REENCODE = True
    if REENCODE:
        FOURCC = cv2.VideoWriter_fourcc(*'H264')
        #PATH_REENCODED_VIDEO = os.path.join(os.path.split(VIDEO_STREAM_PATH)[0], FOLDER_NAME + '_CNN_simulation.mp4')
        PATH_REENCODED_VIDEO = os.path.join('/home/alex/Desktop/', FOLDER_NAME + '_CNN_simulation.mp4')
        print(PATH_REENCODED_VIDEO)
        print("[INFO] Reencoding in", PATH_REENCODED_VIDEO)
    
    # Instantiate tracker object
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    COLOR = {'WHITE': (255, 255, 255), 'BLUE': (255, 0, 0), 'GREEN': (0, 255, 0), 'RED': (0, 0, 255), 'BLACK': (0, 0, 0)}


    main()
    cv2.destroyAllWindows()
