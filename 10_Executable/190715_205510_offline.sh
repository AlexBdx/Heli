#!/bin/bash

python3 main_offline.py \
-v ../0_Database/RPi_import/190715_205510/190715_205510_helico_1920x1080_40s_25fps_L.mp4 \
-bb ../0_Database/RPi_import/190715_205510/190715_205510_extrapolatedBB.pickle \
-p md_params.csv \
-ma ../4_CNN/190727_104133/190727_104133.json \
-mw ../4_CNN/190727_104133/190727_104133.h5
