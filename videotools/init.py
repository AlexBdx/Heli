import os
import psutil
import cv2
import numpy as np
import time

def check_ram_use(unit='Mb'):
    """
    Check and display the current RAM used by the script.
    :return: RAM use in different units
    """
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0]
    if unit == 'kb':
        divider = 2**10
    elif unit=='Mb':
        divider = 2**20
    elif unit=='Gb':
        divider = 2**30
    else:
        print("[WARNING] Unit not understood, defaulted to Mb")
        divider = 2**20

    return memory_use//divider


def import_stream(video_stream_path=None, verbose=False):
    """
    Connect to /dev/video0 or a given file.
    :param video_stream_path:
    :param verbose: more prints
    :return: stream, nb frames, width, height
    """
    # if the video argument is None, then we are reading from webcam
    if video_stream_path is None:
        video_stream = cv2.VideoCapture("/dev/video0")
        time.sleep(2.0)
    # otherwise, we are reading from a video file
    else:
        video_stream = cv2.VideoCapture(video_stream_path)

    # Stream properties
    nb_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get width and height of video stream
    frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if verbose:
        print("[INFO] Imported {} frames with shape x-{} y-{}".format(nb_frames, frame_width, frame_height))
    return video_stream, nb_frames, frame_width, frame_height
    
    
def cache_video(video_stream, method, gray_scale=False):
    """
    Loads in RAM a video_stream as a list or numpy array.
    :param video_stream: the local video file to cache
    :param method: currently, numpy array or list
    :param gray_scale: When True loads all the data as gray images
    :return: the cached video
    """
    nb_frames = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    # Get width and height of video stream
    frame_width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Populate a numpy array
    if method == 'numpy':
        vs_cache = np.zeros((nb_frames, frame_height, frame_width, 3), dtype=np.uint8)
        for i in range(nb_frames):
            frame = video_stream.read()[1]
            vs_cache[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if gray_scale else frame
    # Appends the frames in a list
    elif method == 'list':
        vs_cache = []
        while True:
            frame = video_stream.read()[1]
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if gray_scale else frame
                vs_cache.append(frame)
            else:
                break
    else:
        raise TypeError('This caching method is not supported')
    print("[INFO] Cached {} frames with shape x-{} y-{}".format(nb_frames, frame_width, frame_height))
    return vs_cache
