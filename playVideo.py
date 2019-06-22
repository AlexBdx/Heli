import argparse
import cv2
from imutils.video import FPS
import imutils
import time
import numpy as np


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()#
ap.add_argument("-v", "--video", help="path to the video file")
args = vars(ap.parse_args())

vs = cv2.VideoCapture(args["video"])
# Get frame count
nFrames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT)) 
# Get width and height of video stream
frameWidth = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)) 
frameHeight = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("This feed is {} px by {} px".format(frameWidth, frameHeight))

frameCounter =-1
fps = FPS().start()
"""
bytes = ''
while True:
	bytes += vs.read(1024)
	a = bytes.find('\xff\xd8')
	b = bytes.find('\xff\xd9')
	if a != -1 and b != -1:
		jpg = bytes[a:b+2]
		bytes = bytes[b+2:]
		i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.CV_LOAD_IMAGE_COLOR)
		cv2.imshow('i', i)
		print(i.shape)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
"""
blank = np.zeros((3280, 500))
while True:
	frame = vs.read()[1]
	frameCounter += 1
	if frame is not None:
		print("Frame shape is {}".format(frame.shape))
		#print(frame[100, 100, :])
		frame = imutils.resize(frame, width=1024) # Otherwise this will get out of hand
		print("Resized shape is {}".format(frame.shape))
		cv2.putText(frame, "Frame: {}/{}".format(frameCounter, nFrames), (10, 20),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		cv2.imshow("Real time feed", frame)
		
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
	else:
		break
	#time.sleep(0.2)
	fps.update()

"""
bytes=''
while True:
	bytes+=vs.read(1024)[1]
	a = bytes.find('\xff\xd8') # JPEG start
	b = bytes.find('\xff\xd9') # JPEG end
	if a!=-1 and b!=-1:
		jpg = bytes[a:b+2] # actual image
		bytes= bytes[b+2:] # other informations

		# decode to colored image ( another option is cv2.IMREAD_GRAYSCALE )
		img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.IMREAD_COLOR) 
		cv2.imshow('Window name',img) # display image while receiving data
		if cv2.waitKey(1) ==27: # if user hit esc
		    exit(0) # exit program
"""
fps.stop()
print("Average: {} fps".format(fps.fps()))
print("Number of frames: {}".format(frameCounter+1))
vs.release()
cv2.destroyAllWindows()

