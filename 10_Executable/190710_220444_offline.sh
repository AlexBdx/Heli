#!/bin/bash

python3 main_offline.py \
-v ../0_Database/RPi_import/190710_220444/190710_220444_helico_1920x1080_38s_25fps_T.mp4 \
-bb ../0_Database/RPi_import/190710_220444/190710_220444_extrapolatedBB.pickle \
-p md_params.csv \
-ma ../4_CNN/190727_104133/190727_104133.json \
-mw ../4_CNN/190727_104133/190727_104133.h5
