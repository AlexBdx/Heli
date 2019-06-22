# USAGE
# python opencv_object_tracking.py
# python opencv_object_tracking.py --video dashcam_boston.mp4 --tracker csrt

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import numpy as np
import csv

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]

# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
	tracker = cv2.Tracker_create(args["tracker"].upper())

# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
else:
	# initialize a dictionary that maps strings to their corresponding
	# OpenCV object tracker implementations
	OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create
		#"kcf": cv2.TrackerKCF_create,
		#"boosting": cv2.TrackerBoosting_create,
		#"mil": cv2.TrackerMIL_create,
		#"tld": cv2.TrackerTLD_create,
		#"medianflow": cv2.TrackerMedianFlow_create,
		#"mosse": cv2.TrackerMOSSE_create
	}


# initialize the bounding box coordinates of the object we are going
# to track
initBB = None
f0 = 40 # First frame in the video where we make the measurement
# initialize the FPS throughput estimator
fps = None
perf = np.zeros((len(OPENCV_OBJECT_TRACKERS), 2)) # Success/fail counter
firstTracker = True
bbCorners = []

# loop over frames from the video stream
for c, trackerName in enumerate(OPENCV_OBJECT_TRACKERS):
	fpsVideo = []
	# if a video path was not supplied, grab the reference to the web cam
	if not args.get("video", False):
		print("[INFO] starting video stream...")
		vs = VideoStream(src=0).start()
		time.sleep(1.0)

	# otherwise, grab a reference to the video file
	else:
		vs = cv2.VideoCapture(args["video"])


	tracker = OPENCV_OBJECT_TRACKERS[trackerName]()
	firstFrame = True
	frameCounter = -1
	while True:
		# grab the current frame, then handle if we are using a
		# VideoStream or VideoCapture object
		frameCounter +=1
		if frameCounter < f0:
			frame = vs.read()
			
			continue

		frame = vs.read()

		frame = frame[1] if args.get("video", False) else frame

		# check to see if we have reached the end of the stream
		if frame is None:
			break

		# resize the frame (so we can process it faster) and grab the
		# frame dimensions
		#frame = imutils.resize(frame, width=500)
		(H, W) = frame.shape[:2]

		# check to see if we are currently tracking an object
		if initBB is not None:
			# grab the new bounding box coordinates of the object
			(success, box) = tracker.update(frame)

			# check to see if the tracking was a success
			if success: #First frame is for the initBB
				(x, y, w, h) = [int(v) for v in box]
				cv2.rectangle(frame, (x, y), (x + w, y + h),
					(0, 255, 0), 2)
				bbCorners.append([x, y])
			if not firstFrame:
				perf[c] += [1, 0] if success else [0, 1]

			# update the FPS counter
			fps.update()
			fps.stop()

			# initialize the set of information we'll be displaying on
			# the frame
			currentFps = fps.fps()
			fpsVideo.append(currentFps)
			info = [
				("Tracker", trackerName),
				("Success", "Yes" if success else "No"),
				("FPS", "{:.2f}".format(currentFps)),
			]

			# loop over the info tuples and draw them on our frame
			for (i, (k, v)) in enumerate(info):
				text = "{}: {}".format(k, v)
				cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the 's' key is selected, we are going to "select" a bounding
		# box to track
		#if key == ord("s"):

		if firstTracker and frameCounter==f0:
			# select the bounding box of the object we want to track (make
			# sure you press ENTER or SPACE after selecting the ROI)
			initBB = cv2.selectROI("Frame", frame, fromCenter=False,
				showCrosshair=True)
			firstTracker = False
			print("Video: ", args["video"])
		if firstFrame:
			# start OpenCV object tracker using the supplied bounding box
			# coordinates, then start the FPS throughput estimator as well
			tracker.init(frame, initBB)
			fps = FPS().start()
			firstFrame = False

		# if the `q` key was pressed, break from the loop
		elif key == ord("q"):
			break
		

	# if we are using a webcam, release the pointer
	if not args.get("video", False):
		vs.stop()

	# otherwise, release the file pointer
	else:
		vs.release()

	print("{:<10}\t{} correct \t{} failed\tRatio: {:.2f}\t Av FPS: {:.2f}".format(trackerName, int(perf[c, 0]), int(perf[c, 1]), perf[c, 0]/(perf[c, 0]+perf[c, 1]), np.mean(fpsVideo)))

# close all windows
cv2.destroyAllWindows()
with open("frameLocations.csv", 'w') as f:
	w = csv.writer(f)
	for corner in bbCorners:
		w.writerow(corner)
