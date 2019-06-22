# USAGE
# python opencv_object_tracking.py --video dashcam_boston.mp4

# import the necessary packages
from imutils.video import VideoStream
#from imutils.video import FPS
import argparse
import imutils
import time
import cv2
#import numpy as np
import csv
from psutil import Process
from os import getpid

def checkRamUse():
	pid = getpid()
	py = Process(pid)
	return py.memory_info()[0]

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="csrt",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

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


print("Caching video...")
# Cache the video to better manipulate it - will take a ton of RAM
t0 = time.perf_counter()
vs_cache = []
vs = cv2.VideoCapture(args["video"])
while True:
	frame = vs.read()[1]
	if frame is not None:
		vs_cache.append(frame)
	else:
		break
t1 = time.perf_counter()
print("Caching done in {:.2f} s\tRAM used: {} Mb".format(t1-t0, checkRamUse()//2**20))

index = 0
frameChange = False
# Tracking related
tracker = OPENCV_OBJECT_TRACKERS["csrt"]()
flagTrackerActive = False
flagSuccess = False
box = (0, 0, 0, 0)
heliBBox = []

cv2.imshow("Frame", vs_cache[index])
while True:
	# wait for user's choice
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break
	if key == ord('a'):
		if index > 0:
			index -= 1
			frameChange = True
	if key == ord('d'):
		if index < len(vs_cache)-1:
			index += 1
			frameChange = True
	if key == ord('s'):
		if flagTrackerActive: # Already tracking -> deactivate
			tracker = OPENCV_OBJECT_TRACKERS["csrt"]() # reset
			flagTrackerActive = False
			flagSuccess = False
			box = (0, 0, 0, 0)
			print("Tracker deactivated!")
		else: # not elif to let the user move the frame 
			roi = cv2.selectROI("Frame", vs_cache[index], fromCenter=False,
				showCrosshair=True)
			tracker.init(vs_cache[index], roi) # Init tracker
			flagTrackerActive = True
			print("Tracker activated!")
	
	# Update screen if there was a change
	if frameChange: # The frame index has changed!
		# 1. Update the frame
		frame = vs_cache[index]
		(H, W) = frame.shape[:2]
		info = [("Frame", index)]
		# 2. Manage the tracker
		if flagTrackerActive:
			(flagSuccess, box) = tracker.update(frame) # Update tracker
			(x, y, w, h) = [int(v) for v in box]
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			info.append(("Box", (x, y, w, h)))
			if flagSuccess:
				heliBBox.append([index, (x, y, w, h), 'Tracked'])
				info.append(("Success", "Yes"))
				flagSuccess = False
			else:
				info.append(("Success", "No"))
		else:
			info.append(("Box", box))
			info.append(("Success", "Deactivated"))
		# 3. Display the new frame
		print("Showing frame ", index)
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
		cv2.imshow("Frame", frame)
		# 4. Reset flag
		frameChange = False # Back to normal

# Dump the BBox to file
cv2.destroyAllWindows()
with open("frameLocations.csv", 'w') as f:
	out = csv.writer(f, delimiter=';')
	for entry in heliBBox:
		out.writerow(entry)
with open("extrapolatedFrameLocations.csv", 'w') as f:
	out = csv.writer(f, delimiter=';')
	if len(heliBBox) >= 2:
		for index in range(len(heliBBox)-1):
			# Case 1: next BBox is in the next frame
			if heliBBox[index+1][0] == heliBBox[index][0]+1:
				out.writerow(heliBBox[index])
			# Case 2: next BBox is a few frames away -> we extrapolate
			else:
				(xs, ys, ws, hs) = heliBBox[index][1]
				(xf, yf, wf, hf) = heliBBox[index+1][1]
				n = heliBBox[index+1][0]-heliBBox[index][0]
				for i in range(n):# Ideally, just 1 iteration
					(xi, yi, wi, hi) = (round(xs+i*(xf-xs)/n), round(ys+i*(yf-ys)/n), round(ws+i*(wf-ws)/n), round(hs+i*(hf-hs)/n))
					if i==0:
						out.writerow(heliBBox[index]) # Leave it as tracked
					else:
						out.writerow([heliBBox[index][0]+i, (xi, yi, wi, hi), 'Linear_extrapolation'])
	else: # There are 0 or 1 frame -> we just write that
		for entry in heliBBox:
			out.writerow(entry)
