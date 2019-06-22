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

class imageStabilizer(object):
	def __init__(self, frameWidth, frameHeight, maxGoodPoints=100):
		self.goodPts = []
		self.flagPts = False # We do not have good points yet
		self.frameCounter = 0
		self.maxGoodPoints = maxGoodPoints
		self.timing = []
		self.frameWidth = frameWidth
		self.frameHeight = frameHeight
		self.prev_pts = []

	def fixBorder(self, frame):
		s = frame.shape
		# Scale the image 4% without moving the center
		T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
		frame = cv2.warpAffine(frame, T, (s[1], s[0]))
		return frame

	def extractMatrix(self, m):
		# Extract translation
		dx = m[0,2]
		dy = m[1,2]

		# Extract rotation angle
		da = np.arctan2(m[1,0], m[0,0])
		return [dx, dy, da]

	def stabilizeFrame(self, previousBwFrame, currentBwFrame):

		# Detect feature points in previous frame
		t0 = time.perf_counter()
		if not self.flagPts:
			self.prev_pts = cv2.goodFeaturesToTrack(previousBwFrame, maxCorners=self.maxGoodPoints, qualityLevel=0.01, minDistance=30, blockSize=3)
			self.flagPts=True
		t1 = time.perf_counter()

		t2 = time.perf_counter()
		# Calculate optical flow (i.e. track feature points) using the same pts, as they should not have moved after image stabilization.
		curr_pts, status, err = cv2.calcOpticalFlowPyrLK(previousBwFrame, currentBwFrame, self.prev_pts, None, maxLevel=10, winSize=(21, 21))
		t3 = time.perf_counter()

		# Sanity check
		assert self.prev_pts.shape == curr_pts.shape 
		#print(status)

		# Keep only pts for which the flow was found
		idx = np.where(status==1)[0]
		self.prev_pts = self.prev_pts[idx]
		curr_pts = curr_pts[idx]
		if len(self.prev_pts) < self.maxGoodPoints/2:
			self.flagPts = False # We have lost too many points, need to redo


		#Find transformation matrix to go from curr_pts -> prev_pts
		t4 = time.perf_counter()
		m, _ = cv2.estimateAffinePartial2D(curr_pts, self.prev_pts)
		t5 = time.perf_counter()

		if m is not None:
			# stabilize the frame
			t6 = time.perf_counter()
			frameStabilized = cv2.warpAffine(currentBwFrame, m, (self.frameWidth, self.frameHeight))
			t7 = time.perf_counter()
			#frameStabilized = self.fixBorder(frameStabilized)
			self.timing.append([t1-t0, t3-t2, t5-t4, t7-t6])
			return m, frameStabilized

	def detailedTiming(self):
		estTiming = np.array(self.timing.copy())
		print("[INFO] Timing: {:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(np.mean(estTiming[:,0]), np.mean(estTiming[:,1]), np.mean(estTiming[:,2]), np.mean(estTiming[:,3])))
		print("[INFO] Total stabilization time: {:.6f}".format(np.sum(np.mean(estTiming, axis=0))))
		print("[INFO] {}/{} pts left in prev_pts".format(len(self.prev_pts), self.maxGoodPoints))


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

def createPolygons():
	# Poly 1 is everything but the landing pad
	polygon1 = [[300, 280],[400, 220], [620, 185],[620, 220],[430, 270], [320, 330]]
	polyObject1 = mp.Path(polygon1)
	pts1 = np.array(polygon1).reshape((-1, 1, 2))
	# Polygon 2 is the whole area
	polygon2 = [[200, 340],[400, 220], [620, 185],[620, 220],[430, 270], [225, 380]]
	polyObject2 = mp.Path(polygon2)
	pts2 = np.array(polygon2).reshape((-1, 1, 2))
	return pts1, pts2, polyObject1, polyObject2

def showFeed(s, threshFeed, deltaFrame, currentFrame):
	if s[0] == '1':
		cv2.imshow("Thresh", threshFeed)
	if s[1] == '1':
		cv2.imshow("currentFrame Delta", deltaFrame)
	if s[2] == '1':
		cv2.imshow("Security Feed", currentFrame)
	


KPIs = []

vs = cv2.VideoCapture("/gdrive/My Drive/landing_unstable.mp4", CAP_FFMPEG)
# Check if camera opened successfully
if (vs.isOpened()== False): 
	print("Error opening video stream or file")

#------------------
# CACHE THE VIDEO
#------------------
vs_cache = []
while True:
	frame = vs.read()[1]
	if frame is not None:
		vs_cache.append(frame)
	else:
		break
vs = vs_cache
nFrames = len(vs)
(frameHeight, frameWidth, _) = vs[0].shape
print("[INFO] Cached {} frames with shape x-{} y-{}".format(nFrames, frameWidth, frameHeight))
bbHelicopter = []
with open("/gdrive/My Drive/frameLocations.csv", 'r') as f:
	r = csv.reader(f)
	for entry in r:
		bbHelicopter.append(tuple([int(v) for v in entry]))

#--------------------------
# Pre-optimization setting
#--------------------------
"""REAL TIME
# Get frame count
nFrames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT)) 
# Get width and height of video stream
frameWidth = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)) 
frameHeight = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
"""
# Create a log file
with open("log.csv", 'w') as f:
	w = csv.writer(f)
	w.writerow(["gaussWindow", "mgp", "realFps", "avNbBoxes", "avNbFilteredBoxes", "avNbHeliBox", "percentHeliTotalFiltered", "percentFrameWithHeli"])

modif = np.zeros((nFrames, 3))
timing = np.zeros((nFrames, 4))
#minArea = args["min_area"]

# Hyperparam
# Stabilization
flagStabilization = True
#mgp = 100
# Gaussian blurringq
#gaussWindow = 5 # px of side
#assert gaussWindow%2 == 1
#minArea = 10

text=0
nbBoxes = []

maxFrames = nFrames
# When is the helicopter moving to area 2?
section2Frames = 525
distanceToHelico = 15 # acceptable distance to actual BBox center

# DEBUG - OPTIMIZATION
# 1.Create polygons for BB classification
# Polygon 1 is the first area where it travels

pts1, pts2, polyObject1, polyObject2 = createPolygons()
iterations = []

# Build the iteration table
for gaussWindow in range(1, 12, 2):#tqdm.tqdm(range(1, 12, 2)): # 11
	for mgp in range (25, 51, 25):#range (25, 151, 25): # 6
		for minArea in [x**2 for x in range(1, 2)]:#[x**2 for x in range(1, 6)]: # 5
			for diffMethod in ['dilate', 'open']: # 2
				iterations.append([gaussWindow, mgp, minArea, diffMethod])
print("[INFO] Starting {} iterations".format(len(iterations)))

for simulation in tqdm.tqdm(iterations):
	gaussWindow, mgp, minArea, diffMethod = simulation

	# initialize the first currentFrame in the video stream

	previousGaussFrame = None
	previousGrayFrame = None
	fps = FPS().start()

	#frameCounter = 0

	if flagStabilization:
		iS = imageStabilizer(frameWidth, frameHeight, maxGoodPoints=mgp)
		padding = 10 # px

	# ----------------------------
	# FRAME PROCESSING
	#-----------------------------
	for frameCounter, frame in enumerate(vs):
	#while True:
		if frameCounter > maxFrames-2:
			#time.sleep(10)
			break

		# I. Grab the current in color space
		currentFrame = frame
		#currentFrame = vs.read()[1] # [0] is a bool
		#if currentFrame is None:
			#break
		#frameCounter += 1

		#currentFrame = imutils.resize(currentFrame, width=500)
		
		# II. Convert to gray scale
		currentGrayFrame = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)
		if previousGrayFrame is None:
			previousGrayFrame = currentGrayFrame
			continue

		# III. Stabilize the image in the gray space, fwd to color space
		if flagStabilization:
			m, currentGrayFrame = iS.stabilizeFrame(previousGrayFrame, currentGrayFrame)
			currentFrame = cv2.warpAffine(currentFrame, m, (frameWidth, frameHeight))
			#currentFrame = currentFrame[int(cropPerc*frameHeight):int((1-cropPerc)*frameHeight), int(cropPerc*frameWidth):int((1-cropPerc)*frameWidth)]
			#modif[frameCounter-1] = iS.extractMatrix(m)
		
		# IV. Gaussian Blur
		currentGaussFrame = cv2.GaussianBlur(currentGrayFrame, (gaussWindow, gaussWindow), 0)
		if previousGaussFrame is None:
			previousGaussFrame = currentGaussFrame
			continue
		
		# V. Differentiation in the Gaussian space
		diffFrame = cv2.absdiff(currentGaussFrame, previousGaussFrame)
		deltaFrame = diffFrame.copy()

		# VI. BW space manipulations
		diffFrame = cv2.threshold(diffFrame, 25, 255, cv2.THRESH_BINARY)[1]
		# dilate the thresholded image to fill in holes, then find contours
		if diffMethod == 'dilate':
			diffFrame = cv2.dilate(diffFrame, None, iterations=2)
		elif diffMethod == 'open':
			diffFrame = cv2.morphologyEx(diffFrame, cv2.MORPH_OPEN, None)
		threshFeed = diffFrame.copy() # DEBUG
		cnts = cv2.findContours(diffFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		# TBR #cv2.polylines(currentFrame,[pts1],True,(0,0,255)) # BGR format
		# TBR #cv2.polylines(currentFrame,[pts2],True,(0,0,255))
		# Cirle around the actual corner of the helicoBBox
		# Obtained via manual CSRT tracker
		#cv2.circle(currentFrame, bbHelicopter[frameCounter], distanceToHelico, (0,0,255), -1)
		
		largeBox = 0
		heliBB = 0
		#assert np.array_equal(test, currentFrame)

		# VII. Process the BB and classify them
		for c in cnts:
			"""
			plt.figure()
			plt.imshow(currentFrame)
			plt.figure()
			plt.imshow(currentFrame-test)
			plt.show()
			assert np.array_equal(test, currentFrame)
			"""
			# A. Filter out useless BBs
			# 1. if the contour is too small, ignore it
			if cv2.contourArea(c) < minArea:
				continue
			# compute the bounding box for the contour, draw it on the currentFrame,
			# and update the text
			#(x, y, w, h) = cv2.boundingRect(c)
			(x, y, s) = boundingSquare(c)
			
			# 2. Box partially out of the frame
			if x < 0 or x+s > frameWidth or y < 0 or y+s > frameHeight:
				continue
			# 3. Box center in the padding area
			if flagStabilization:
				if not(padding < x+s//2 < frameWidth-padding and padding < y+s//2 < frameHeight-padding):
					continue

			# B. Classify BBs
			largeBox += 1
			# Check if the corner is within range of the actual corner
			# That data was obtained by running a CSRT tracker on the helico
			xHelico, yHelico = bbHelicopter[frameCounter]
			if np.sqrt((x-xHelico)**2+(y-yHelico)**2) < distanceToHelico:
				heliBB += 1

			# C. Generate a square BB
			#cv2.rectangle(currentFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)
			#cv2.rectangle(currentFrame, (x, y), (x + s, y + s), (0, 255, 0), 2)


		
		

		# VIII. draw the text and timestamp on the currentFrame
		"""REAL TIME
		if frameCounter%10 == 0:
			text = len(cnts)
		cv2.putText(currentFrame, "Helicopter: {} found".format(text), (10, 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		cv2.putText(currentFrame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
			(10, currentFrame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
		"""

		# IX. show the currentFrame and record if the user presses a key
		showFeed('000', threshFeed, deltaFrame, currentFrame)

		"""REAL TIME
		key = cv2.waitKey(1) & 0xFF
		# if the `q` key is pressed, break from the loop
		if key == ord("q"):
			break
		"""

		# X. Save frames & track KPI
		previousGaussFrame = currentGaussFrame.copy()
		previousGrayFrame = currentGrayFrame.copy()
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
	print("[INFO] approx. FPS: {:.2f} \t real FPS: {:.2f}\tRatio (approx/real): {:.2f}".format(predictedFps, realFps, ratio))
	if flagStabilization:
		#print(iS.detailedTiming())
		pass


	#Impact of stabilization on number of boxes
	bb=np.array(nbBoxes)
	bb = bb[1:] # Delete first frame which is not motion controlled

	# KPI
	# per frame
	avNbBoxes = np.mean(bb[:, 0])
	avNbFilteredBoxes = np.mean(bb[:, 1])
	avNbHeliBox = np.mean(bb[:, 2])
	percentHeliTotalFiltered = avNbHeliBox/avNbFilteredBoxes
	percentFrameWithHeli = np.sum(bb[:, 3])/frameCounter

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

	# Output results
	KPIs.append([gaussWindow, mgp, realFps, avNbBoxes, avNbFilteredBoxes, avNbHeliBox, percentHeliTotalFiltered, percentFrameWithHeli])
	with open("log.csv", 'a') as f:
		w = csv.writer(f)
		w.writerow([gaussWindow, mgp, realFps, avNbBoxes, avNbFilteredBoxes, avNbHeliBox, percentHeliTotalFiltered, percentFrameWithHeli])

print("Done!")

