import extract_ROI
import glob
import matplotlib.pyplot as plt
import time
import numpy as np

folder='/home/alex/Desktop/Helico/0_Database/RPi_import/190622_201853/190622_201853_NN_crops/nnSizeCrops/*'
output = '/home/alex/Desktop/Helico/0_Database/RPi_import/190622_201853/190622_201853_NN_crops/Test/'
images = [img for img in glob.glob(folder)]

BLUR = 3
CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 200
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (0.0,0.0,1.0) # In BGR format
max_area = 80*40
list_max_area = []
extractor = extract_ROI.extract(BLUR, CANNY_THRESH_1, CANNY_THRESH_2, MASK_DILATE_ITER, MASK_ERODE_ITER, max_area)
timing = []
for index, img in enumerate(images):
    img_array = plt.imread(img)
    t0 = time.perf_counter()
    list_max_area.append(extractor.extract_helico(img_array, output+str(index)))
    t1 = time.perf_counter()
    timing.append(t1-t0)
print("Average time: {:.3f} per image ({:.3f} img/s)".format(np.mean(timing), 1/np.mean(timing)))

