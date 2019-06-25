# Run Monte Carlo simulations to find the best set of parameters for motion detection
# The ranking is done based on the f1_score for each param set

#from imutils.video import videoStream
from imutils.video import FPS
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mp
import tqdm
import csv
import collections
import os
import psutil
from sklearn.model_selection import ParameterGrid
import pickle
import copy

# Custom made files
import imageStabilizer

# 0. DECLARATIONS
def checkRamUse():
	pid = os.getpid()
	py = psutil.Process(pid)
	memoryUse = py.memory_info()[0]
	print('RAM use: ', memoryUse)

def boundingSquare(c):
	# Create a square box rather than a rectangle one.
	# Needed for the NN
	(x, y, w, h) = cv2.boundingRect(c)
	(xc, yc) = (x+w//2, y+h//2)
	s = max(w, h)
	(xs, ys) = (xc-s//2, yc-s//2)
	#(xs, ys, s) = (1200, 1, 100) # Test square boundaries
	#(xs, ys, s) = (-1, 1, 100) # Test square boundaries
	#(xs, ys, s) = (1000, -1, 100) # Test square boundaries
	#(xs, ys, s) = (1000, 700, 100) # Test square boundaries
	return (xs, ys, s)

def showFeed(s, threshFeed, deltaFrame, currentFrame):
	if s[0] == '1':
		cv2.imshow("Thresh", threshFeed)
	if s[1] == '1':
		cv2.imshow("currentFrame Delta", deltaFrame)
	if s[2] == '1':
		cv2.imshow("Security Feed", currentFrame)

def importStream(videoStreamPath=None, verbose=False):
	# if the video argument is None, then we are reading from webcam
	if videoStreamPath is None:
		videoStream = cv2.VideoCapture("/dev/video0")
		time.sleep(2.0)
	# otherwise, we are reading from a video file
	else:
		videoStream = cv2.VideoCapture(videoStreamPath)
	
	# Stream properties
	nFrames = int(videoStream.get(cv2.CAP_PROP_FRAME_COUNT)) 
	# Get width and height of video stream
	frameWidth = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH)) 
	frameHeight = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))
	if verbose:
		print("[INFO] Imported {} frames with shape x-{} y-{}".format(nFrames, frameWidth, frameHeight))
	return videoStream, nFrames, frameWidth, frameHeight

def cacheVideo(videoStream, method):
	# Only works from a file!
	nFrames = int(videoStream.get(cv2.CAP_PROP_FRAME_COUNT)) 
	# Get width and height of video stream
	frameWidth = int(videoStream.get(cv2.CAP_PROP_FRAME_WIDTH)) 
	frameHeight = int(videoStream.get(cv2.CAP_PROP_FRAME_HEIGHT))
	# Populate a numpy array
	if method == 'numpy':
		vs_cache = np.zeros((nFrames, frameHeight, frameWidth, 3), dtype=np.uint8)
		for i in range(nFrames):
			vs_cache[i] = videoStream.read()[1]
	# Appends the frames in a list
	if method == 'list':
		vs_cache = []
		while True:
			frame = videoStream.read()[1]
			if frame is not None:
				vs_cache.append(frame)
			else:
				break

	print("[INFO] Cached {} frames with shape x-{} y-{}".format(nFrames, frameWidth, frameHeight))
	return vs_cache

def manageLog(logFilePath, params, restart, verbose=False):
	# Start from scratch
	if restart is None:
		with open(logFilePath, 'w') as f:
			w = csv.writer(f)
			newHeader = list(params.keys())+["realFps", "avNbBoxes", "avNbFilteredBoxes", "avNbHeliBox", "percentHeliTotalFiltered", "percentFrameWithHeli", "f1_score"]
			w.writerow(newHeader)
			print("Log header is now ", newHeader)
		iterationStart = 0
	# Restart case study from a given sim
	else:
		iterationStart = args["restart"]
	
	if verbose:
		print("Starting at iteration {}".format(iterationStart))
	return iterationStart

def importHeliBB(heliBBfile):
	# Import the known locations of the helicopter
	with open(heliBBfile, 'rb') as f:
		#r = csv.reader(f, delimiter=';')
		bbHelicopter = pickle.load(f)
		"""[TBR]
		for entry in r:
			bbHelicopter.append([entry[0], tuple(int(v) for v in entry[1][1:-1].split(','))])
		"""
	return bbHelicopter

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()#
ap.add_argument("-v", "--video", help="path to the video file", required=True)
ap.add_argument("-bb", "--bounding_boxes", type=str, help="path to ground truth bounding boxes", required=True)
ap.add_argument("-r", "--restart", type=int, help="iteration restart")
args = vars(ap.parse_args())

#------------------
# Import the file/stream
#------------------
videoStreamPath = args["video"]


#--------------------------
# ITERATION TABLE
#--------------------------

params = {
'gaussWindow': range(3, 6, 2), \
'mgp': range(25, 26, 25), \
'minArea': [x**2 for x in range(1, 2)],\
'residualConnections': range(1, 4),\
'winSize': range(3, 4, 2),\
'maxLevel': range(5, 6, 3),\
'threshold': range(25, 46, 10),\
'diffMethod': range(0, 1, 1),\
'dilationIterations': range(4, 11, 2),\
'skipFrame': range(0, 1, 1)
}
iterationDict = ParameterGrid(params)

bbPath = args["bounding_boxes"]
# Need to change the logFilePath name
logFileName = os.path.split(bbPath)[1][:14]+"Detection_paramSearch.csv" # Replace the (pickle) extension by a csv
logFilePath = os.path.join(os.path.split(bbPath)[0], logFileName)
print(logFilePath)
iterationStart = manageLog(logFilePath, params, args["restart"])
bbHelicopter = importHeliBB(bbPath) # Creates a dict

#modif = np.zeros((nFrames, 3))
#timing = np.zeros((nFrames, 4))
#minArea = args["min_area"]

# Hyperparam
# Stabilization
#mgp = 100
# Gaussian blurring
#gaussWindow = 5 # px of side
#assert gaussWindow%2 == 1
#minArea = 10

#--------------
# STATIC HYPERPARAMS
#--------------
displayFeed = '000'
# bboxError is max error ratio to count a bbox as matching ground truth
# This applies to all axis (xc, yc, w, h)
bboxError = 0.5
red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)
flagPhaseCorrelation = False # This is too slow (>3x slower than mgp)
flagOpticalFlow = False # A bit better, but still a lot
verbose = False


print("[INFO] Starting {} iterations".format(len(iterationDict)))
firstBbox = min(bbHelicopter.keys())
lastBbox = max(bbHelicopter.keys())
print("[INFO] Using bbox frames {} to {}".format(firstBbox, lastBbox))

vs2 = cacheVideo(importStream(videoStreamPath)[0], 'list')
for sd in tqdm.tqdm(iterationDict):
	#-------------------------------------
	# 1. RESET THE SIM DEPENDENT VARIABLES
	#-------------------------------------
	# Reload the video from disk as we do not cache it
	vs, nFrames, frameWidth, frameHeight = importStream(videoStreamPath)
	timing = {'Read frame': 0, 'Convert to grayscale': 0, 'Stabilize': 0, 'Double Gauss': 0, 'Abs diff': 0, 'Thresholding': 0, 'Dilation': 0, 'Count boxes': 0, 'Finalize':0}
	nbBoxes = [] # Stores bbox data for a sim

	# Get ready to store residualConnections frames over and over
	previousGrayFrame = collections.deque(maxlen=sd['residualConnections'])
	#previousGaussFrame = collections.deque(maxlen=sd['residualConnections'])
	
	iS = imageStabilizer.imageStabilizer(frameWidth, frameHeight, maxGoodPoints=sd['mgp'], maxLevel=sd['maxLevel'], winSize=sd['winSize'])
	padding = 10 # px
	skipFrameCounter = sd['skipFrame'] # Go through the if statement the first time
	
	fps = FPS().start()
	# ----------------------------
	# 2. FRAME PROCESSING - GO THROUGH ALL FRAMES WITH A BBOX
	#-----------------------------
	
	for frameNumber in range(nFrames):
		
		t0 = time.perf_counter()
		#frame = vs.read()[1] # No cache
		frame = vs2[frameNumber].copy() # Prevents editing the original frames!
		t1 = time.perf_counter()
		# Skip all the frames that do not have a Bbox
		if frameNumber < firstBbox:
			continue
		if frameNumber > min(nFrames-2, lastBbox):
			#print("Done with this sim. FirstBox {}, lastBox {}, frameNumber {}".format(firstBbox, lastBbox, frameNumber))
			break
		
		# 0. Skip frames - subsampling of FPS
		if skipFrameCounter < sd['skipFrame']:
			skipFrameCounter += 1
			continue
		else:
			skipFrameCounter = 0
		
		# Create a 0 based index that tracks how many bboxes we have gone through
		bboxFrameNumber = frameNumber - firstBbox # Starts at 0, automatically incremented
		# Load the previous frames in memory for residual connections
		if bboxFrameNumber < sd['residualConnections']:
			currentFrame = frame
			#print(frame.shape)
			currentGrayFrame = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)
			previousGrayFrame.append(currentGrayFrame)
			continue

		# I. Grab the current in color space
		#t0=time.perf_counter()
		currentFrame = frame
		
		# II. Convert to gray scale
		t2 = time.perf_counter()
		currentGrayFrame = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)
		

		# III. Stabilize the image in the gray space with latest gray frame, fwd to color space
		# Two methods (dont' chain them): phase correlation & optical flow
		t3 = time.perf_counter()
		if flagPhaseCorrelation:
			retval, response = cv2.phaseCorrelate(np.float32(previousGrayFrame[-1][100:1000, 100:1000])/255.0, np.float32(currentGrayFrame[100:1000, 50:950])/255.0)
		if flagOpticalFlow:
			m, currentGrayFrame = iS.stabilizeFrame(previousGrayFrame[-1], currentGrayFrame)
			currentFrame = cv2.warpAffine(currentFrame, m, (frameWidth, frameHeight))
		t4 = time.perf_counter()
		#currentFrame = currentFrame[int(cropPerc*frameHeight):int((1-cropPerc)*frameHeight), int(cropPerc*frameWidth):int((1-cropPerc)*frameWidth)]
		#modif[bboxFrameNumber-1] = iS.extractMatrix(m)
		
		# IV. Gaussian Blur
		# Done between currentFrame and the grayFrame from residualConnections ago
		currentGaussFrame = cv2.GaussianBlur(currentGrayFrame, (sd['gaussWindow'], sd['gaussWindow']), 0)
		previousGaussFrame = cv2.GaussianBlur(previousGrayFrame[0], (sd['gaussWindow'], sd['gaussWindow']), 0)
		t5 = time.perf_counter()
		
		# V. Differentiation in the Gaussian space
		diffFrame = cv2.absdiff(currentGaussFrame, previousGaussFrame)
		t6 = time.perf_counter()
		if displayFeed != '000':
			deltaFrame = diffFrame.copy()

		# VI. BW space manipulations
		diffFrame = cv2.threshold(diffFrame, sd['threshold'], 255, cv2.THRESH_BINARY)[1]
		t7 = time.perf_counter()
		# dilate the thresholded image to fill in holes, then find contours
		if sd['diffMethod'] == 0:
			diffFrame = cv2.dilate(diffFrame, None, iterations=sd['dilationIterations'])
			t8 = time.perf_counter()
		elif sd['diffMethod'] == 1:
			diffFrame = cv2.morphologyEx(diffFrame, cv2.MORPH_OPEN, None)
		if displayFeed != '000':
			threshFeed = diffFrame.copy()
		cnts = cv2.findContours(diffFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		t8 = time.perf_counter()

		# Cirle around the actual corner of the helicoBBox
		# Obtained via manual CSRT tracker
		#cv2.circle(currentFrame, bbHelicopter[bboxFrameNumber], bboxError, (0,0,255), -1)
		
		largeBox = 0
		heliBB = 0

		# VII. Process the BB and classify them
		xGT, yGT, wGT, hGT = bbHelicopter[frameNumber] # Ground Truth data
		for c in cnts:
			# A. Filter out useless BBs
			# 1. if the contour is too small, ignore it
			if cv2.contourArea(c) < sd['minArea']:
				continue
			# compute the bounding box for the contour, draw it on the currentFrame,
			# and update the text
			(x, y, w, h) = cv2.boundingRect(c)
			#(x, y, s) = boundingSquare(c) # Great idea but we will make a square ourselves later
			
			# 2. Box partially out of the frame
			#if x < 0 or x+s > frameWidth or y < 0 or y+s > frameHeight: # Square box
			if x < 0 or x+w > frameWidth or y < 0 or y+h > frameHeight:
				continue
			# 3. Box center in the padding area
			#if not(padding < x+s//2 < frameWidth-padding and padding < y+s//2 < frameHeight-padding): # Square box version
			if not(padding < x+w//2 < frameWidth-padding and padding < y+h//2 < frameHeight-padding):
				continue

			# B. Classify BBs - a largeBox is a potential heliBB
			largeBox += 1
			# Check if the corner is within range of the actual corner
			# That data was obtained by running a CSRT tracker on the helico
			
			
			# Is this bbox close enough to the ground truth bbox? Rectangular window
			if abs(x-xGT) < bboxError*wGT and\
			abs(y-yGT) < bboxError*hGT and\
			(1-bboxError)*wGT < w < (1+bboxError)*wGT and\
			(1-bboxError)*hGT < h < (1+bboxError)*hGT:
				heliBB += 1
				if displayFeed == '001': # Display positive heliBBox found in green
					cv2.putText(currentFrame, "heli", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)
					cv2.rectangle(currentFrame, (x, y), (x + w, y + h), green, 2)
			else:
				if displayFeed == '001': # Display negative heliBBox found in blue
					cv2.putText(currentFrame, "not heli", (x, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, blue, 2)
					cv2.rectangle(currentFrame, (x, y), (x + w, y + h), blue, 2)
				pass

			# C. Generate a square BB
			#cv2.rectangle(currentFrame, (x, y), (x + s, y + s), green, 2)
			#cv2.rectangle(currentFrame, (x, y), (x + w, y + h), green, 2)
		if displayFeed == '001':
			cv2.rectangle(currentFrame, (xGT, yGT), (xGT + wGT, yGT + hGT), red, 2) # Display ground truth in red
		t9 = time.perf_counter()

		# VIII. draw the text and timestamp on the currentFrame
		if displayFeed != '000':
			if bboxFrameNumber%2 == 0:
				cv2.putText(currentFrame, "BBoxes: {} found, {} heliBox".format(len(cnts), heliBB), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 2)
				#cv2.putText(currentFrame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red, 1) # Shows current date/time

			# IX. show the currentFrame and record if the user presses a key
			showFeed(displayFeed, threshFeed, deltaFrame, currentFrame)
			
			key = cv2.waitKey(1) & 0xFF
			# if the `q` key is pressed, break from the loop
			if key == ord("q"):
				break


		# X. Save frames & track KPI
		#previousGaussFrame = currentGaussFrame.copy()
		# The deque has a maxlen of residualConnections so the first-in will pop
		previousGrayFrame.append(currentGrayFrame)
		nbBoxes.append([len(cnts), largeBox, heliBB, 1 if heliBB else 0])
		#print([len(cnts), largeBox, heliBB, 1 if heliBB else 0])

		fps.update()
		t10 = time.perf_counter()
		if verbose:
			newTiming = {'Read frame': t1-t0, 'Convert to grayscale': t3-t2, 'Stabilize': t4-t3, 'Double Gauss': t5-t4, 'Abs diff': t6-t5, 'Thresholding': t7-t6, 'Dilation': t8-t7, 'Count boxes': t9-t8, 'Finalize':t10-t9}
			for key in timing.keys():
				timing[key] += newTiming[key]

	# XI. Display results
	fps.stop()
	vs.release() # Done with going through this simulation, get ready for another pass
	if verbose:
		print("Code profiling for various operations (in s):\n", timing)
	cv2.destroyAllWindows()

	elapsedTime = fps.elapsed()
	predictedFps = fps.fps()
	realFps = bboxFrameNumber/elapsedTime
	ratio = predictedFps/realFps
	#print("[INFO] elasped time: {:.2f}".format(elapsedTime))
	#print("[INFO] frame count: {}".format(bboxFrameNumber))
	#print("[INFO] approx. FPS: {:.2f} \t real FPS: {:.2f}\tRatio (approx/real): {:.2f}".format(predictedFps, realFps, ratio))
	print("[INFO] FPS: {:.2f}".format(realFps))

	#print(iS.detailedTiming())



	#Impact of stabilization on number of boxes
	bb=np.array(nbBoxes)
	bb = bb[1:] # Delete first frame which is not motion controlled

	# KPI
	# per simulation
	#print(bb)
	avNbBoxes = np.mean(bb[:, 0])
	avNbFilteredBoxes = np.mean(bb[:, 1])
	avNbHeliBox = np.mean(bb[:, 2])
	# Precision: how efficient is the algo at rulling out irrelevant boxes?
	percentHeliTotalFiltered = avNbHeliBox/avNbFilteredBoxes # Ratio of helibox/nb of boxes
	# Recall: how many frames had a positive heliBox? There should be one in each.
	percentFrameWithHeli = np.sum(bb[:, 3])/bboxFrameNumber # Proportion of frames with heli
	f1_score = 1/(1/percentHeliTotalFiltered+1/percentFrameWithHeli)

	#-----------------
	# SANITY CHECKS
	#-----------------
	assert percentFrameWithHeli<=1
	assert percentHeliTotalFiltered<=1
	assert avNbHeliBox <= avNbFilteredBoxes
	assert avNbFilteredBoxes <= avNbBoxes

	"""KPIs
	plt.figure()
	plt.plot(bb[:, 0])
	plt.plot(bb[:, 1])
	plt.plot(bb[:, 2])
	plt.legend(("Number of boxes", "Boxes large enough", "Heli box"))
	titl = \
	"Boxes detected - av: {:.2f} - std: {:.2f} at {:.2f} FPS\n\
	Av Helibb per frame: {:.3f} - Ratio of helibb: {:.3f}\tFrame with heli: {:.3f} "\
	.format(\
	avNbFilteredBoxes, np.std(bb[:, 1]), realFps, \
	avNbHeliBox, percentHeliTotalFiltered, percentFrameWithHeli\
	)
	plt.title(titl)
	plt.show()
	"""

	# Output results - parameters+KPIs
	KPIs = [realFps, avNbBoxes, avNbFilteredBoxes, avNbHeliBox, percentHeliTotalFiltered, percentFrameWithHeli, f1_score]
	# Warning: they are both int array of the same length so they can be added!
	
	simOutput = [sd[k] for k in params.keys()] + list(KPIs)

	with open(logFilePath, 'a') as f:
		w = csv.writer(f)
		w.writerow(simOutput)

# XII. Wrap-up the search & output some logs for quick review
# XII.1. Find the best params based on the f1_score metric
with open(logFilePath, 'r') as f:
	r = csv.reader(f)
	next(r) # Skip header
	highestF1Score = 0
	for entry in r:# Keep the score always last item
		if float(entry[-1]) > highestF1Score:
			highestF1Score = float(entry[-1])
			bestParams = entry

# XII.2. Create a new log file with the best params
_ = manageLog(logFilePath[:-4]+'_bestParams.csv', params, None)
with open(logFilePath[:-4]+'_bestParams.csv', 'a') as f:
	w = csv.writer(f)
	w.writerow(bestParams)

# XII.3. Pickle the params dict
with open(logFilePath[:-4]+'_paramSpace.pickle', 'wb') as f:
	pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)

# XII.4. Final message!!
print("Done. Highest f1_score: ", highestF1Score)
