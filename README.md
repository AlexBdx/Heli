# Helicopter Tracking

## Goal

Reliably detect motion, identify and track helicopters flying in the urban sky using the limited resources offered by a RPi 3B+.

## Steps

### Motion detection
Based on gaussian blur differences on gray scales images. This is still a work in progress and a few files in the motion detection folder attempt to optimize the f1_score of my motion detection parameters.

### Identification
Transfer learning on MobileNetV2. Top layers retrained on a dataset manually collected over many weeks in order to classify bounding boxes surrounding detected motion as helicopter/non helicopter.

### Tracking
Tracking of detected helicopters using OpenCV built in trackers.
