"""
190618 - HelicoCapture.py
This script captures a video of given length/framerate/sensor_mode/resolution.
The goal is to record a helicopter landing/take-off with the RPi camera.
Resolution above full HD might be severely limited in fps due to the increased weight of using
mjpeg and the resolution of the images.
"""

import time
import picamera
import numpy as np
import sys
import psutil
import os
import cv2
import subprocess as sp
import argparse

def argumentChecking(sensor_mode, res, framerate, duration):
	# Check max resolution
	if res[0] > 3820:
		print("[WARNING] Max width is 3820, set to 3820")
		res[0] = 3820
	if res[1] > 2464:
		print("[WARNING] Max height is 2464, set to 2464")
		res[1] = 2464
	# Test h264/mjpeg cases
	if max(res) > 1920:
		if sensor_mode is not 2:
			print("[WARNING][MJPEG] sensor_mode set to 2")
			sensor_mode = 2
		if not 0.1 < framerate < 15:
			print("[WARNING][MJPEG] Framerate set between 0.1 and 15 fps")
			framerate = 15 if framerate>15 else framerate
			framerate = 0.1 if framerate < 0.1 else framerate
		if duration < 1:
			duration = 1
			print("[WARNING][MJPEG] Duration is too short, set to 1 s")
	else:
		if sensor_mode is not 1:
			print("[WARNING][H264] sensor_mode set to 1")
			sensor_mode = 1
	return sensor_mode, res, framerate, duration

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()#
ap.add_argument("-d", "--duration", type=int, default=10, help="recording duration")
ap.add_argument("-fps", "--fps", type=int, default=25, help="fps")
ap.add_argument("-s", "--sensor_mode", type=int, default=1, help="sensor_mode")
ap.add_argument("-r", "--resolution", type=str, default='1920x1080', help="Resolution in the format wxh. Example: 1920x1080")
args = vars(ap.parse_args())


# MAIN
if __name__ == '__main__':
	# 1. Collect arguments
	sensor_mode = args["sensor_mode"]
	res = tuple(int(x) for x in args["resolution"].split('x'))
	framerate = args["fps"]
	duration = args["duration"]
	sensor_mode, res, framerate, duration = argumentChecking(sensor_mode, res, framerate, duration)
	# 2. Create a specific folder that will only contain the latest movie, delete the previous one.
	folderName = 'newVideo'
	if os.path.isdir(folderName):
		sp.run(['rm', '-r', folderName])
	sp.run(['mkdir', folderName])
	assert os.path.isdir(folderName)
	# 3. Name the file - ex: helico_1920x1080_45s_25fps
	fileName = 'helico_'+str(res[0])+'x'+str(res[1])+'_'+str(duration)+'s_'+str(framerate)+'fps.'
	
	with picamera.PiCamera() as camera:
		# 1. Camera configuration
		camera.hflip=True 
		camera.vflip=True
		camera.sensor_mode = sensor_mode
		camera.resolution = res # WARNING: defaults to 1280x720 when the tv off!
		camera.framerate = framerate
		# WARNING: H264 IS ONLY SUPPORTED UP TO 1920x1080, MJPEG BEYOND
		codec = 'h264' if max(res) <= 1920 else 'mjpeg'
		camera.sensor_mode = 1 if  max(res) <= 1920 else 2
		fileName += codec # h264 | mjpeg depending on the above
		time.sleep(2) # let the camera warm up and set gain/white balance
		
		# 2. Recording the movie
		camera.start_recording(fileName)
		print("[RPi] Recording {}".format(fileName))
		camera.wait_recording(duration)
		camera.stop_recording()
		print('[RPi] Generated '+codec+' file')

		# 3. File post processing
		if codec == 'h264':
			print('[RPi] Converting to mp4')
			convertedFile = fileName.split('.')[0]+'.mp4'
			ffmpegRequest = ['ffmpeg','-y', '-an', '-i', fileName, '-framerate', str(framerate), '-c', 'copy', convertedFile, '-hide_banner', '-loglevel', 'panic']
			sp.run(ffmpegRequest)
			sp.run(['rm', fileName]) # Delete h264 file
		else:
			convertedFile = fileName
		sp.run(['mv', convertedFile, folderName+'/'+convertedFile])
		print('[RPi] Generated {} ({:.1f} Mb)'.format(convertedFile, os.path.getsize(folderName+'/'+convertedFile)//2**20))

# Read the mjpeg file: vlc --demux=mjpeg --mjpeg-fps=5 video.mjpeg 
