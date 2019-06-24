# Run Monte Carlo simulations to find the best set of parameters for motion detection
# The ranking is done based on the f1_score for each param set

from imutils.video import VideoStream
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

# Custom made files
import imageStabilizer

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

def importStream(stream):
	# if the video argument is None, then we are reading from webcam
	if stream is None:
		vs = cv2.VideoCapture("/dev/video0")
		time.sleep(2.0)
	# otherwise, we are reading from a video file
	else:
		vs = cv2.VideoCapture(stream)
	return vs

def loadVideo(videoStream, method='generator'):
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

def manageLog(logFile, params, restart):
	# Start from scratch
	if restart is None:
		with open(logFile, 'w') as f:
			w = csv.writer(f)
			newHeader = list(params.keys())+["realFps", "avNbBoxes", "avNbFilteredBoxes", "avNbHeliBox", "percentHeliTotalFiltered", "percentFrameWithHeli"]
			w.writerow(newHeader)
			print("Log header is now ", newHeader)
		iterationStart = 0
	# Restart case study from a given sim
	else:
		iterationStart = args["restart"]
	
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

videoStream = importStream(args["video"])
vs, nFrames, frameWidth, frameHeight = loadVideo(videoStream)

#--------------------------
# ITERATION TABLE
#--------------------------

params = {
'gaussWindow': range(3, 8, 2), \
'mgp': range(125, 176, 25), \
'minArea': [x**2 for x in range(1, 4)],\
'residualConnections': range(1, 4),\
'winSize': range(3, 4, 2),\
'maxLevel': range(7, 8, 3),\
'diffMethod': range(0, 1, 1)\
}
iterationDict = ParameterGrid(params)

bbPath = args["bounding_boxes"]
logFile = os.path.split(bbPath)[1]+".csv"
iterationStart = manageLog(logFile, params, args["restart"])
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
# STATIC PARAMS
#--------------
displayFeed = '000'
distanceToHelico = 15 # acceptable distance to actual BBox center


print("[INFO] Starting {} iterations".format(len(iterationDict)))
firstBbox = int(bbHelicopter[0][0])
lastBbox = int(bbHelicopter[-1][0])
print("Starting at frame {}".format(firstBbox))
print("Ending at frame {}".format(lastBbox))
frameInit = True

for sd in tqdm.tqdm(iterationDict):
	#-------------------------------------
	# 1. RESET THE SIM DEPENDENT VARIABLES
	#-------------------------------------
	nbBoxes = []

	# Get ready to store residualConnections frames over and over
	previousGrayFrame = collections.deque(maxlen=sd['residualConnections'])
	#previousGaussFrame = collections.deque(maxlen=sd['residualConnections'])
	
	iS = imageStabilizer.imageStabilizer(frameWidth, frameHeight, maxGoodPoints=sd['mgp'], maxLevel=sd['maxLevel'], winSize=sd['winSize'])
	padding = 10 # px
	
	fps = FPS().start()
	# ----------------------------
	# 2. FRAME PROCESSING
	#-----------------------------
	for frameCounter in range(nFrames):
		print(frameCounter)
		frame = vs.read()[1]
	# We need to skip all the first frames when the helicopter is not in sight
	# And stop running the alg when there are no more ground truth BBoxes 
		if frameCounter == firstBbox and frameInit:
			print("Offsetting the frame number")
			frameCounter = 0
			frameInit = False
		else:
			continue
		if frameCounter == lastBbox+1-firstBbox:
			pr
			break
	# Now we have skipped the first frames for which there are no 
	#while True:
		if frameCounter > nFrames-2: # Due to tracker output - preserves frameLocations.csv
			#time.sleep(10)
			break
		if frameCounter < sd['residualConnections']:
			currentFrame = frame
			currentGrayFrame = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)
			previousGrayFrame.append(currentGrayFrame)
			continue

		# I. Grab the current in color space
		#t0=time.perf_counter()
		currentFrame = frame
		
		# II. Convert to gray scale
		currentGrayFrame = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)

		# III. Stabilize the image in the gray space with latest gray frame, fwd to color space
		# We use the previous frame
		m, currentGrayFrame = iS.stabilizeFrame(previousGrayFrame[-1], currentGrayFrame)
		currentFrame = cv2.warpAffine(currentFrame, m, (frameWidth, frameHeight))
		#currentFrame = currentFrame[int(cropPerc*frameHeight):int((1-cropPerc)*frameHeight), int(cropPerc*frameWidth):int((1-cropPerc)*frameWidth)]
		#modif[frameCounter-1] = iS.extractMatrix(m)
		
		# IV. Gaussian Blur
		# Done between currentFrame and the grayFrame from residualConnections ago
		currentGaussFrame = cv2.GaussianBlur(currentGrayFrame, (sd['gaussWindow'], sd['gaussWindow']), 0)
		previousGaussFrame = cv2.GaussianBlur(previousGrayFrame[0], (sd['gaussWindow'], sd['gaussWindow']), 0)

		
		# V. Differentiation in the Gaussian space
		diffFrame = cv2.absdiff(currentGaussFrame, previousGaussFrame)
		if displayFeed != '000':
			deltaFrame = diffFrame.copy()

		# VI. BW space manipulations
		diffFrame = cv2.threshold(diffFrame, 25, 255, cv2.THRESH_BINARY)[1]
		# dilate the thresholded image to fill in holes, then find contours
		if sd['diffMethod'] == 0:
			diffFrame = cv2.dilate(diffFrame, None, iterations=2)
		elif sd['diffMethod'] == 1:
			diffFrame = cv2.morphologyEx(diffFrame, cv2.MORPH_OPEN, None)
		if displayFeed != '000':
			threshFeed = diffFrame.copy()
		cnts = cv2.findContours(diffFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)


		# Cirle around the actual corner of the helicoBBox
		# Obtained via manual CSRT tracker
		#cv2.circle(currentFrame, bbHelicopter[frameCounter], distanceToHelico, (0,0,255), -1)
		
		largeBox = 0
		heliBB = 0

		# VII. Process the BB and classify them
		for c in cnts:
			# A. Filter out useless BBs
			# 1. if the contour is too small, ignore it
			if cv2.contourArea(c) < sd['minArea']:
				continue
			# compute the bounding box for the contour, draw it on the currentFrame,
			# and update the text
			#(x, y, w, h) = cv2.boundingRect(c)
			(x, y, s) = boundingSquare(c)
			
			# 2. Box partially out of the frame
			if x < 0 or x+s > frameWidth or y < 0 or y+s > frameHeight:
				continue
			# 3. Box center in the padding area

			if not(padding < x+s//2 < frameWidth-padding and padding < y+s//2 < frameHeight-padding):
				continue

			# B. Classify BBs
			largeBox += 1
			# Check if the corner is within range of the actual corner
			# That data was obtained by running a CSRT tracker on the helico
			xHelico, yHelico = bbHelicopter[frameCounter][1][:2] 
			print(bbHelicopter[frameCounter])
			print(xHelico, yHelico)
			if np.sqrt((x-xHelico)**2+(y-yHelico)**2) < distanceToHelico:
				heliBB += 1

			# C. Generate a square BB
			#cv2.rectangle(currentFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			#cv2.rectangle(currentFrame, (x, y), (x + s, y + s), (0, 255, 0), 2)


		# VIII. draw the text and timestamp on the currentFrame
		if displayFeed != '000':
			if frameCounter%10 == 0:
				text = len(cnts)
			cv2.putText(currentFrame, "Helicopter: {} found".format(text), (10, 20),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
			cv2.putText(currentFrame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
				(10, currentFrame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

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

		fps.update()


	# XI. Display results
	fps.stop()
	
	"""REAL TIME
	vs.release()
	"""
	cv2.destroyAllWindows()

	elapsedTime = fps.elapsed()
	predictedFps = fps.fps()
	realFps = frameCounter/elapsedTime
	ratio = predictedFps/realFps
	#print("[INFO] elasped time: {:.2f}".format(elapsedTime))
	#print("[INFO] frame count: {}".format(frameCounter))
	#print("[INFO] approx. FPS: {:.2f} \t real FPS: {:.2f}\tRatio (approx/real): {:.2f}".format(predictedFps, realFps, ratio))
	print("[INFO] FPS: {:.2f}".format(realFps))

	#print(iS.detailedTiming())



	#Impact of stabilization on number of boxes
	print(nbBoxes)
	bb=np.array(nbBoxes)
	bb = bb[1:] # Delete first frame which is not motion controlled

	# KPI
	# per simulation
	avNbBoxes = np.mean(bb[:, 0])
	avNbFilteredBoxes = np.mean(bb[:, 1])
	avNbHeliBox = np.mean(bb[:, 2])
	percentHeliTotalFiltered = avNbHeliBox/avNbFilteredBoxes
	percentFrameWithHeli = np.sum(bb[:, 3])/frameCounter

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
	KPIs = [realFps, avNbBoxes, avNbFilteredBoxes, avNbHeliBox, percentHeliTotalFiltered, percentFrameWithHeli]
	# Warning: they are both int array of the same length so they can be added!
	
	simOutput = [sd[k] for k in params.keys()] + list(KPIs)

	with open(logFile, 'a') as f:
		w = csv.writer(f)
		w.writerow(simOutput)

print("Done!")

