import numpy as np
import cv2
import copy
import time

class image_stabilizer(object):
    # function/variables needs to be renamed to support PEP 8
    def __init__(self, frameWidth, frameHeight, maxGoodPoints=100, maxLevel=3, winSize=21):
        self.goodPts = []
        self.flagPts = False # We do not have good points yet
        self.frameCounter = 0
        self.maxGoodPoints = maxGoodPoints
        self.timing = []
        self.frameWidth = frameWidth
        self.frameHeight = frameHeight
        self.prev_pts = []
        self.maxLevel = maxLevel
        self.winSize = winSize


    def fix_border(self, frame):
        s = frame.shape
        # Scale the image 4% without moving the center
        T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
        frame = cv2.warpAffine(frame, T, (s[1], s[0]))
        return frame


    def extract_matrix(self, m):
        # Extract translation
        dx = m[0,2]
        dy = m[1,2]

        # Extract rotation angle
        da = np.arctan2(m[1,0], m[0,0])
        return [dx, dy, da]


    def stabilize_frame(self, previousBwFrame, currentBwFrame):

        # Detect feature points in previous frame
        t0 = time.perf_counter()
        if not self.flagPts:
            self.prev_pts = cv2.goodFeaturesToTrack(previousBwFrame, maxCorners=self.maxGoodPoints, qualityLevel=0.01, minDistance=30, blockSize=3)
            self.flagPts=True
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        # Calculate optical flow (i.e. track feature points) using the same pts, as they should not have moved after image stabilization.
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(previousBwFrame, currentBwFrame, self.prev_pts, None, maxLevel=self.maxLevel, winSize=(self.winSize, self.winSize))
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


    def detailed_timing(self):
        estTiming = np.array(self.timing.copy())
        print("[INFO] Timing: {:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(np.mean(estTiming[:,0]), np.mean(estTiming[:,1]), np.mean(estTiming[:,2]), np.mean(estTiming[:,3])))
        print("[INFO] Total stabilization time: {:.6f}".format(np.sum(np.mean(estTiming, axis=0))))
        print("[INFO] {}/{} pts left in prev_pts".format(len(self.prev_pts), self.maxGoodPoints))
