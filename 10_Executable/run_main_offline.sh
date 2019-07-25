#!/bin/bash

python3 main_offline.py \
-v ../0_Database/RPi_import/190622_201853/190622_201853_helico_1920x1080_45s_25fps_L.mp4 \
-bb ../0_Database/RPi_import/190622_201853/190622_201853_extrapolatedBB.pickle \
-p md_params.csv \
-ma ../4_CNN/190723_Update_Negative/190723.json \
-mw ../4_CNN/190723_Update_Negative/190723.h5
