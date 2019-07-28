#!/bin/bash

# This script asks the RPi to record a video. The user can input a desired framerate
# and duration. If that is not the case, default values are used.

HOST=pi@rpi
FILE_PATH='/home/pi/Desktop/newVideo/'

# 1. Set some defaults values
duration=45 # [s]
fps=25 # [-]
sensor_mode=1 # [-]
resolution='1920x1080' # [-]
quality=30 # [-] default, otherwise 10 to 40
# 2. Get positional arguments
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -d|--duration)
    duration="$2"
    shift # past argument
    shift # past value
    ;;
    -fps|--fps)
    fps="$2"
    shift # past argument
    shift # past value
    ;;
    -s|--sensor_mode)
    sensor_mode="$2"
    shift # past argument
    shift # past value
    ;;
    -r|--resolution)
    resolution="$2"
    shift # past argument
    shift # past value
    ;;
    -q|--quality)
    quality="$2"
    shift # past argument
    shift # past value
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

# 3. Send recording request to RPi
echo 'Requesting '$duration' s of ['$resolution'] at '$fps' fps - sensor_mode='$sensor_mode', quality '$quality
sshpass -p 'raspberree' ssh $HOST 'cd Desktop/ && python3 HelicoCapture.py -d '$duration' -fps '$fps' -s '$sensor_mode' -r '$resolution' -q '$quality

# 4. Download mp4 file from RPi
if sshpass -p 'raspberree' ssh $HOST stat $FILE_PATH \> /dev/null 2\>\&1
then
    echo "Retrieving converted file..."
    sshpass -p 'raspberree' scp -r $HOST:$FILE_PATH 0_Database/RPi_import/
    echo "Done"

    else
        echo "Converted file does not exist! Please check and retrieve manually."
fi
