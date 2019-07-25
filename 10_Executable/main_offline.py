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
import video_tools as vt
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
        crop = vt.bbox.nn_size_crop(crop, size, (xc, yc), frame.shape)
    elif method == 'cropsResizedToNn':
        s = max(w, h) if max(w, h) % 2 == 0 else max(w, h) + 1  # even only
        x_start = max(0, xc - s//2)
        x_end = min(FRAME_WIDTH, xc + s//2)
        y_start = max(0, yc - s//2)
        y_end = min(FRAME_HEIGHT, yc + s//2)
        crop = frame[y_start:y_end, x_start:x_end]
        crop = vt.bbox.nn_size_crop(crop, (s, s), (xc, yc), frame.shape)  # pad to (s, s)
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


def infer_bbox(model, frame, bbox, method):
    # Returns a uint8
    
    crop = vt.bbox.nn_size_crop(frame, bbox, NN_SIZE)
    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)  # The CNN was trained on RGB data    
    pre_processed_crop = vt.transfer_learning.preprocess_image(crop, DTYPE_IMAGES)
    single_sample = np.expand_dims(pre_processed_crop, 0)  # Make it a single inference
    #print(pre_processed_crop.shape)
    #print(single_sample.shape)
    prediction = model.predict(single_sample)[0][0] # Single sample, single class
    prediction = np.round(prediction).astype(np.uint8)
    
    return prediction, crop

def plot_confusion_matrix(matrix):
    """If you prefer color and a colorbar"""
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)


def display_frame(frame, bbox, frame_count, flag_tracker_active):
    #print("[INFO] bbox:", bbox)
    if bbox != (0, 0, 0, 0):
        x, y, w, h = bbox
        if flag_tracker_active:
            title = 'Helico'
            color = COLOR['RED']
        else:
            title = 'Motion'
            color = COLOR['BLUE']
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, title, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        pass  # Nothing was detected in this frame
    tracker_state = 'ON' if flag_tracker_active else 'OFF'
    text_count = 'Frame number: {}'.format(frame_count)
    text_tracker = 'Tracker: {}'.format(tracker_state)
    cv2.putText(frame, text_count, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR['WHITE'], 2)
    cv2.putText(frame, text_tracker, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR['WHITE'], 2)
    
    cv2.imshow(VIDEO_STREAM_PATH, frame)
    key = cv2.waitKey(1) & 0xFF
    flag_quit_program = True if key == ord('q') else False
    
    return flag_quit_program

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
        loaded_model = vt.transfer_learning.load_model(PATH_ARCHITECTURE, PATH_WEIGHTS)
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
    extractor = vt.extract.extractor()
    # Create a deque for frame accumulation
    previous_gray_frame = collections.deque(maxlen=params['residualConnections'])
    # Track the results
    Y_prediction = []
    Y_test = []
    # Count global number of bboxes
    counter_bbox = 0
    # Start re-encoding
    if REENCODE:
        reencoded_video = cv2.VideoWriter(PATH_REENCODED_VIDEO, FOURCC, 25, (1920, 1080))
    t0 = time.perf_counter()  # Mostly used for first frame display
    for frame_number in range(NB_FRAMES):
        
        current_frame = VIDEO_STREAM.read()[1]
        
        t1 = time.perf_counter()
        
        # Check the status of the tracker
        #print("[INFO] Frame {}\tTracker status: {}".format(frame_number, flag_tracker_active))
        flag_tracker_active = False
        if flag_tracker_active:  # The tracker is ON
            flag_success, bbox = Tracker.update(current_frame)
            bbox = [int(value) for value in bbox]  # The update returns floating point values...
            if frame_number%CNN_CHECK_STRIDE == 0:
                print("[INFO] Verifying tracker at frame", frame_number)
                # Time to verify that the Tracker is still locked on the helico
                prediction, crop = infer_bbox(loaded_model, current_frame, bbox, METHOD)
                if prediction == 1:  # All good, keep going
                    flag_quit_program = display_frame(current_frame, bbox, frame_number, flag_tracker_active)
                    #fps.update()
                    if flag_quit_program:
                        return
                    continue
                elif prediction == 0:  # There is actually no helico in the tracked frame!
                    Tracker = OPENCV_OBJECT_TRACKERS["csrt"]()  # Reset tracker
                    flag_tracker_active = False
                    previous_gray_frame = collections.deque(maxlen=params['residualConnections'])
                    pass  # Engage the motion detection algo below
                else:
                    print("[ERROR] The model is supposed to be a binary classifier")
                    raise
            else:  # Between checks from the CNN, go to the next frame
                flag_quit_program = display_frame(current_frame, bbox, frame_number, flag_tracker_active)
                #fps.update()
                if flag_quit_program:
                        return
                continue
        else: # The tracker is OFF
            #if frame_number%CNN_CHECK_STRIDE:
                #continue
             # Engage the motion detection algo below
            pass

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
        
        # Differentation
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
        counter_bbox_heli = 0
        if first_bbox <= frame_number <= last_bbox:
            (x_gt, y_gt, w_gt, h_gt) = bbox_heli_ground_truth[frame_number]  # Ground Truth data
        else:
            (x_gt, y_gt, w_gt, h_gt) = (1919, 1079, 1, 1)
        counter_failed_detection = 0
        (x, y, w, h) = (0, 0, 0, 0)
        success = []
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
            
            # Determine the label for this box based on the IoU with the ground truth one.
            converted_bbox = vt.bbox.xywh_to_x1y1x2y2((x, y, w, h))
            converted_gt_bbox = vt.bbox.xywh_to_x1y1x2y2((x_gt, y_gt, w_gt, h_gt))
            label = 1 if vt.bbox.intersection_over_union(converted_bbox, converted_gt_bbox) >= params['iou'] else 0
            
            
            # Append both results to their respective lists
            Y_prediction.append(prediction)
            Y_test.append(label)
            
            
            
            #prediction = 0
            name = 'Helico' if prediction else 'Motion'
            color = COLOR['RED'] if prediction else COLOR['BLUE']
            cv2.putText(current_frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(current_frame, (x, y), (x + w, y + h), color, 2)
        
        # Update the deque
        previous_gray_frame.append(current_gray_frame)
        # Add the total number of bboxes detected so far
        text_bboxes = 'Number of detected bboxes: {}'.format(counter_bbox)
        cv2.putText(current_frame, text_bboxes, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR['WHITE'], 2)
        # Add the current accuracy for the CNN
        #accuracy = 1-np.sum(np.abs(np.array(Y_prediction)-np.array(Y_test)))/len(Y_test)
        if len(Y_test):
            conf_mx = confusion_matrix(Y_test, Y_prediction)
            if conf_mx.shape == (2, 2):
                accuracy = (conf_mx[0, 0]+conf_mx[1, 1])/np.sum(conf_mx)
            else:
                conf_mx = np.zeros((2, 2))
                accuracy = 1
        else:
            conf_mx = np.zeros((2, 2))
            accuracy = 1
        text_accuracy = 'Cumulative accuracy: {:.1f} % TN: {} TP: {} FN: {} FP: {}'.format(100*accuracy, conf_mx[0, 0], conf_mx[1, 1], conf_mx[1, 0], conf_mx[0, 1])
        cv2.putText(current_frame, text_accuracy, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR['WHITE'], 2)
        
        # Update tracker state
        tracker_state = 'ON' if flag_tracker_active else 'OFF'
        text_tracker = 'Tracker: {}'.format(tracker_state)
        cv2.putText(current_frame, text_tracker, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR['WHITE'], 2)
        # Update FPS and frame count
        fps = 1/(time.perf_counter() - t0)
        text_count = 'FPS: {:.1f} Frame number: {}'.format(fps, frame_number)
        cv2.putText(current_frame, text_count, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR['WHITE'], 2)
        t0 = time.perf_counter()
        # Re-encode the frame
        if REENCODE:
            reencoded_video.write(current_frame)
        
        # Display frame
        cv2.imshow(VIDEO_STREAM_PATH, imutils.resize(current_frame, width=1000))
        #cv2.imshow("Diff_frame", diff_frame)
        #cv2.imshow("Canny frame", canny_frame)
        
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            #print(Y_prediction)
            #print(Y_test)
            if REENCODE:
                reencoded_video.release()
            return
            
        """
            print("[INFO] Inferred bbox: ", prediction)
            if prediction == 1:  # First helico detected!
                #Tracker.init(current_frame, (x, y, w, h))
                #flag_tracker_active = True
                #success.append([prediction, crop])
                
                #break
                #continue
            elif prediction == 0:
                counter_failed_detection += 1
                if counter_failed_detection >= MAX_FAILED_INFERENCES:
                    #break  # Not time for another attempt. Improve the motion detection.
                    #continue
            else:
                print("[ERROR] The model is supposed to be a binary classifier")

        print("[INFO] Length success:", len(success))
        if len(success):
            fig, ax = plt.subplots(1, max(2, len(success)))
            print(len(success))
            for index, res in enumerate(success):
                print(index)
                ax[index].imshow(cv2.cvtColor(res[1], cv2.COLOR_BGR2RGB))
                ax[index].set_title("Heli")
                ax[index].axis('off')
            plt.show()
        """
        #fps.update()
        """[Is display_frame still useful?]
        # Display the result from motion detection
        #print("[INFO] Display frame")
        #flag_quit_program = display_frame(current_frame, (x, y, w, h), frame_number, flag_tracker_active)
        
        print()
        
        if flag_quit_program:
            return
        """
        
        """[For later]
        # Classify bboxes based on their IOU with ground truth
        converted_current_bbox = vt.bbox.xywh_to_x1y1x2y2(bbox_crop)
        converted_ground_truth_bbox = vt.bbox.xywh_to_x1y1x2y2((x_gt, y_gt, w_gt, h_gt))
        if vt.bbox.intersection_over_union(converted_current_bbox, converted_ground_truth_bbox) >= IOU:
            counter_bbox_heli += 1
        """
    #fps.stop()
    conf_mx = confusion_matrix(Y_test, Y_prediction)
    print("[INFO] Confusion Matrix:\n", conf_mx)
    #print("[INFO] Final accuracy: {:.1f}".format(100*accuracy))
    #plt.figure()
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.title("Confusion matrix on {} | Accuracy: {:.1f}%"
    .format(FOLDER_NAME, 100*accuracy))
    plt.savefig("Confusion_matrix_"+FOLDER_NAME, tight_layout=False)
    plt.show()
    if REENCODE:
        reencoded_video.release()
    

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
    
    # Offline method - load a video & its bboxes
    VIDEO_STREAM, NB_FRAMES, FRAME_WIDTH, FRAME_HEIGHT = vt.init.import_stream(VIDEO_STREAM_PATH)
    bbox_heli_ground_truth = vt.bbox.import_bbox_heli(PATH_BBOX)  # Creates a dict
    
    # Set some global constants
    METHOD = 'nnSizeCrops'
    DTYPE_IMAGES = np.float32  # Can the RPi run TF in float16 and do so faster?
    MIN_AREA = 100
    MAX_AREA = 112*112
    PADDING = 1
    CNN_CHECK_STRIDE = 25
    MAX_FAILED_INFERENCES = 1
    NN_SIZE = (224, 224)

    # Re-encode the video to save the result
    REENCODE = True
    if REENCODE:
        FOURCC = cv2.VideoWriter_fourcc(*'H264')
        PATH_REENCODED_VIDEO = os.path.join(os.path.split(VIDEO_STREAM_PATH)[0], FOLDER_NAME + '_CNN_simulation.mp4')
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
