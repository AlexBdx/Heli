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
#import csv
from psutil import Process
import os
import copy
import pickle
from glob import glob
import numpy as np


# 0. DECLARATIONS
def checkRamUse():
	# V190624
	pid = os.getpid()
	py = Process(pid)
	return py.memory_info()[0]

def nnSizeCrop(image, windowSize, bboxCenter):
	# V190625
	# Calculate the required black padding around the image to make it windowSize[0]windowSize[1]
	# image.shape can only be smaller or equal to windowSize as a frame slice of windowSize
	#print("Initial shape: ", image.shape)
	xc, yc = bboxCenter
	# Calculate how much padding is needed
	top = windowSize[1]//2-yc if yc-windowSize[1]//2<0 else 0
	bottom = yc+windowSize[1]//2-frameHeight if yc+windowSize[1]//2>frameHeight else 0
	left = windowSize[0]//2-xc if xc-windowSize[0]//2<0 else 0
	right = xc+windowSize[0]//2-frameWidth if xc+windowSize[0]//2>frameWidth else 0
	if top or bottom or left or right:
		# Add a black padding where necessary
		image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
	# DEBUG
	# 1. There shall be no negative param
	try:
		assert top >= 0
		assert bottom >= 0
		assert left >= 0
		assert right >= 0
		# 2. The final shape shall be windowSize + 3 channels
		assert image.shape == (windowSize[0], windowSize[1], 3)
	except AssertionError:
		print("TBLR: ", top, bottom, left, right)
		print("Output image shape: ", image.shape, (windowSize[0], windowSize[1], 3))

	return image

def cropNegative(frame, nnSize, bboxCenter):
	# This function will extract a negative - an image that does not contain the object in the ROI
	# nnSize is width x height
	xc, yc = bboxCenter
	frameHeight, frameWidth, _ = frame.shape
	try:
		assert frameWidth > 2*nnSize[0] and frameHeight > 2*nnSize[1]
	except AssertionError:
		print("[cropNegative] The input image is to small to crop a negative")
		return None
	xn = np.random.randint(frameWidth)
	yn = np.random.randint(frameHeight)
	while (xc-nnSize[0] < xn < xc+nnSize[0] and yc-nnSize[1] < yn < yc+nnSize[1]) or xn > frameWidth - nnSize[0] or yn > frameHeight - nnSize[1]:
		""" [TBR]
		print(xn, yn)
		print(xc-nnSize[0] < xn < xc+nnSize[0], xn, xc)
		print(yc-nnSize[1] < yn < yc+nnSize[1], yn, yc)
		print(xn > frameWidth - nnSize[0], frameWidth - nnSize[0], xn)
		print(yn > frameHeight - nnSize[1], frameHeight - nnSize[1], yn)
		print("\n")
		"""
		xn = np.random.randint(frameWidth)
		yn = np.random.randint(frameHeight)
	return frame[yn:yn+nnSize[1], xn:xn+nnSize[0]]

def loadVideo(videoStream, method='generator'):
	# V190624
	# Get frame count
	nFrames = int(videoStream.get(cv2.CAP_PROP_FRAME_COUNT)) 
	# Get width and height of video stream
	frameWidth = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH)) 
	frameHeight = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))

	# Populate a numpy array
	if method == 'numpy':
		vs = np.zeros((nFrames, frameHeight, frameWidth, 3), dtype=np.uint8)
		for i in range(nFrames):
			vs[i] = videoStream.read()[1]
	# Appends the frames in a list
	if method == 'list':
		vs = []
		while True:
			frame = videoStream.read()[1]
			if frame is not None:
				vs.append(frame)
			else:
				break
	if method == 'generator':
		vs = videoStream

	print("[INFO] Imported {} frames with shape x-{} y-{}".format(nFrames, frameWidth, frameHeight))
	return vs, nFrames, frameWidth, frameHeight

# I. SETUP
# I.1. Preparing the arguments
# Videos: ../0_Database/RPi_import/
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, help="path to input video file", required=True)
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


# I.3. Cache the video & create folder architecture
print("Caching video...")
t0 = time.perf_counter()
videoPath = args["video"]
vs = cv2.VideoCapture(videoPath)
# I.3.1 Creating the folder architecture
videoFolder = os.path.split(videoPath)[0]
ts = os.path.split(videoPath)[1][:14]
sourceBbPath = os.path.join(videoFolder, ts+"sourceBB.pickle")
extrapolatedBbPath = os.path.join(videoFolder, ts+"extrapolatedBB.pickle")
cropFolder = os.path.join(os.path.split(videoPath)[0], ts+'NN_crops')
nnSizeCropsFolder = os.path.join(cropFolder, 'nnSizeCrops')
cropsResizedToNnFolder = os.path.join(cropFolder, 'cropsResizedToNn')
negatives = os.path.join(cropFolder, 'negatives')

# I.3.2 Cache the video
vs_cache, nFrames, frameWidth, frameHeight = loadVideo(vs, method='list')
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
# nnSize needs to be a pair of even number. If not, the next larger even number is used.
nnSize = tuple(int(s) for s in args["neural_network_size"].split('x')) # w, h coordinates
try:
	assert nnSize[0]%2 == 0
	assert nnSize[1]%2 == 0
except AssertionError:
	# Make nnSize the nearest larger even number
	nnSize_0 = nnSize[0] if nnSize[0]%2 == 0 else nnSize[0]+1
	nnSize_1 = nnSize[1] if nnSize[1]%2 == 0 else nnSize[1]+1
	nnSize = (nnSize_0, nnSize_1)
	print("neural_network_size needs to be a pair of even numbers. Input was adjusted to nnSize = ({}, {})".format(*nnSize))
windowName = "Video Feed"


# II. PROCESS THE VIDEO
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
				heliBBox.append([index, (x, y, w, h)])
				info.append(("Success", "Yes"))
				flagSuccess = False
			else:
				info.append(("Success", "No"))
		else:
			info.append(("Box", box))
			info.append(("Success", "Deactivated"))
		# 3. Display the new frame
		#print("Showing frame ", index)
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, (i+1) * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
		cv2.imshow(windowName, frame)
		# 4. Reset flag
		frameChange = False # Back to normal

cv2.destroyAllWindows()

# III. SAVE BBOXES AND EXTRAPOLATED BBOXES TO FILE
# III.1. Pickle the source BBoxes as a dict
with open(sourceBbPath, 'wb') as f:
	heliBBoxSource = dict()
	for entry in heliBBox:
		heliBBoxSource[entry[0]] = entry[1] # Store the tuples only
	pickle.dump(heliBBoxSource, f, protocol=pickle.HIGHEST_PROTOCOL)

# III.2. Pickle the extrapolated BBoxes as a dict
with open(extrapolatedBbPath, 'wb') as f:
	heliBBoxExtrapolated = dict()
	if len(heliBBox) >= 2:
		for index in range(len(heliBBox)-1): # Not great coding but it works.
			# Case 1: next BBox is in the next frame
			if heliBBox[index+1][0] == heliBBox[index][0]+1:
				heliBBoxExtrapolated[heliBBox[index][0]] = heliBBox[index][1] # Store the bbox
			# Case 2: next BBox is a few frames away -> we extrapolate
			else:
				(xs, ys, ws, hs) = heliBBox[index][1]
				(xf, yf, wf, hf) = heliBBox[index+1][1]
				n = heliBBox[index+1][0]-heliBBox[index][0]
				for i in range(n):# Extrapolate the BBoxes
					heliBBoxExtrapolated[heliBBox[index][0]+i] = (round(xs+i*(xf-xs)/n), round(ys+i*(yf-ys)/n), round(ws+i*(wf-ws)/n), round(hs+i*(hf-hs)/n))
	else: # There are 0 or 1 frame -> we just write that
		for entry in heliBBox:
			heliBBoxExtrapolated[entry[0]] = entry[1]
	pickle.dump(heliBBoxExtrapolated, f, protocol=pickle.HIGHEST_PROTOCOL)


# IV. SANITY CHECK: MAKE SURE THE EXTRAPOLATION OVERLAYS WITH ORIGINAL VIDEO
# IV.1. Import extrapolated bboxes
with open(extrapolatedBbPath, 'rb') as f:
	heliBBoxExtrapolated = pickle.load(f)

# IV.2. Replay the video with them & save non-skipped crops for NN
# IV.2.1. Cleanup the crop directory (if it exists)
if not os.path.isdir(cropFolder):
	os.mkdir(cropFolder)
# IV.2.2. nnSizeCropsFolder
if os.path.isdir(nnSizeCropsFolder):
	fileList = glob(os.path.join(nnSizeCropsFolder,'*'))
	#print(os.path.join(nnSizeCropsFolder,'*'))
	#print(fileList)
	if len(fileList)>0:
		for f in fileList:
			os.remove(f)
else:
	os.mkdir(nnSizeCropsFolder)
# IV.2.3. cropsResizedToNnFolder
if os.path.isdir(cropsResizedToNnFolder):
	fileList = glob(os.path.join(cropsResizedToNnFolder,'*'))
	#print(os.path.join(cropsResizedToNnFolder,'*'))
	#print(fileList)
	if len(fileList)>0:
		for f in fileList:
			os.remove(f)
else:
	os.mkdir(cropsResizedToNnFolder)
# IV.2.4. negatives
if os.path.isdir(negatives):
	fileList = glob(os.path.join(negatives,'*'))
	#print(os.path.join(negatives,'*'))
	#print(fileList)
	if len(fileList)>0:
		for f in fileList:
			os.remove(f)
else:
	os.mkdir(negatives)

# IV.2.2 Replay the video and save the crops to file
cropCounter = 0 # Used to increment picture name
bboxCounter = 0 # Used for the skip function

for index, frame in enumerate(vs_cache):
	frame = frame.copy()
	try:
		(x, y, w, h) = heliBBoxExtrapolated[index]
		xc, yc = x+w//2, y+h//2
		# Take the max but make it an even number (odd numbers are a mess with //2)
		s = max(w, h) if max(w, h)%2==0 else max(w, h)+1
		if bboxCounter % (skip+1)==0:
			# IV.2.2.1 First option: nnSizeCrops - nnSize crop
			# Limit the size of the crop
			xStart, xEnd = max(0, xc-nnSize[0]//2), min(frameWidth, xc+nnSize[0]//2)
			yStart, yEnd = max(0, yc-nnSize[1]//2), min(frameHeight, yc+nnSize[1]//2)
			image = frame[yStart:yEnd, xStart:xEnd]
			image = nnSizeCrop(image, nnSize, (xc, yc)) # Forces it to nnSize[0]xnnSize[1]
			outPath = os.path.join(nnSizeCropsFolder, ts+str(cropCounter)+'.jpg')
			cv2.imwrite(outPath, image)
			# IV.2.2.2 Second option: (square) cropsResizedToNn - bbox crop resized to nnSize
			xStart, xEnd = max(0, xc-s//2), min(frameWidth, xc+s//2)
			yStart, yEnd = max(0, yc-s//2), min(frameHeight, yc+s//2)
			image = frame[yStart:yEnd, xStart:xEnd]
			image = nnSizeCrop(image, (s, s), (xc, yc)) # pad to (s, s)
			# Then only we resize to nnSize
			image = cv2.resize(image, nnSize) # Resize to NN input size
			outPath = os.path.join(cropsResizedToNnFolder, ts+str(cropCounter)+'.jpg')
			cv2.imwrite(outPath, image)
			# IV.2.2.3 Create a negative image - no helico
			image = cropNegative(frame, nnSize, (xc, yc))
			outPath = os.path.join(negatives, ts+str(cropCounter)+'.jpg')
			cv2.imwrite(outPath, image)
			# Increment crop counter
			cropCounter += 1
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		bboxCounter += 1
	except KeyError:
		pass
	cv2.imshow(windowName, frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break

