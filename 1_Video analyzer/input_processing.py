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
import os
import copy

def checkRamUse():
	pid = os.getpid()
	py = Process(pid)
	return py.memory_info()[0]


# I. Setup
# I.1. Preparing the arguments
# Videos: ../0_Database/RPi_import/
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="csrt", help="OpenCV object tracker type")
ap.add_argument("-s", "--skip", type=int, default=0, help="Proportion of BBox to save to file")
ap.add_argument("-n", "--neural_network_size", type=str, default='224x224', help="BBox crop size for NN input")
args = vars(ap.parse_args())

# I.2. Initialize a dictionary that maps strings to their corresponding
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create
	#"kcf": cv2.TrackerKCF_create,
	#"boosting": cv2.TrackerBoosting_create,
	#"mil": cv2.TrackerMIL_create,
	#"tld": cv2.TrackerTLD_create,
	#"medianflow": cv2.TrackerMedianFlow_create,
	#"mosse": cv2.TrackerMOSSE_create
}

# I.3. Cache the video
print("Caching video...")
t0 = time.perf_counter()
vs_cache = []
videoPath = args["video"]
vs = cv2.VideoCapture(videoPath)
while True:
	frame = vs.read()[1]
	if frame is not None:
		vs_cache.append(frame)
	else:
		break
t1 = time.perf_counter()
print("Caching done in {:.2f} s\tRAM used: {} Mb".format(t1-t0, checkRamUse()//2**20))

# I.4. Initialize various variables
index = 0
frameChange = False
# Tracking related
tracker = OPENCV_OBJECT_TRACKERS["csrt"]()
flagTrackerActive = False
flagSuccess = False
box = (0, 0, 0, 0)
heliBBox = []
skip = args["skip"]
nnSize = tuple(int(s) for s in args["neural_network_size"].split('x'))
windowName = "Video Feed"


# II. Process the video
#frame = vs_cache[index]
#frame = imutils.resize(frame, width=500)
cv2.imshow(windowName, vs_cache[index])
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
			roi = cv2.selectROI(windowName, vs_cache[index], fromCenter=False,
				showCrosshair=True)
			tracker.init(vs_cache[index], roi) # Init tracker
			flagTrackerActive = True
			print("Tracker activated!")
	
	# Update screen if there was a change
	if frameChange: # The frame index has changed!
		# 1. Update the frame
		frame = vs_cache[index].copy() # Don't edit the original cache!
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
			cv2.putText(frame, text, (10, (i+1) * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
		cv2.imshow(windowName, frame)
		# 4. Reset flag
		frameChange = False # Back to normal


# III. Save BBoxes and extrapolated bbox to file
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


# IV. Sanity check: make sure the extrapolation was meaningful
# IV.1. Import extrapolated bboxes
with open("extrapolatedFrameLocations.csv", 'r') as f:
	inputFile = csv.reader(f, delimiter=';')
	extraBbox = dict()
	for entry in inputFile:
		frameNumber = int(entry[0])
		bboxImport = entry[1][1:-1].split(',')
		bbox = tuple(int(s) for s in bboxImport)
		#print(frameNumber, '\t', bbox)
		extraBbox[frameNumber] = bbox

# IV.2. Replay the video with them & save crops for NN
cropCounter = 0 # Used to increment picture name
bboxCounter = 0 # Used for the skip function
ts = os.path.split(videoPath)[1][:14]
pictureFolder = os.path.join(os.path.split(videoPath)[0], ts+'NN_crops')
if not os.path.isdir(pictureFolder):
	os.mkdir(pictureFolder)
for index, frame in enumerate(vs_cache):
	frame = frame.copy()
	try:
		(x, y, w, h) = extraBbox[index]
		xc, yc = x+w//2, y+h//2
		s = max(w, h)
		if bboxCounter % (skip+1)==0: # Depends how often you want to save crops
			image = frame[yc-s//2:yc+s//2, xc-s//2:xc+s//2]
			image = cv2.resize(image, nnSize) # Resize to NN input size
			outPath = os.path.join(pictureFolder, ts+str(cropCounter)+'.jpg')
			cv2.imwrite(outPath, image)
			cropCounter += 1
		
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		bboxCounter += 1
	except KeyError:
		pass
	cv2.imshow(windowName, frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

