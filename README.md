# DroneDeploy-Coding-Challenge

## Challenge Description

This zip file https://www.dropbox.com/s/g67dmolq79ko4jk/Camera%20Localization.zip?dl=0 contains a number of images taken from different positions and orientations with an iPhone 6. Each image is the view of a pattern on a flat surface. The original pattern that was photographed is 8.8cm x 8.8cm and is included in the zip file. Write a Python program that will visualize (i.e. generate a graphic) where the camera was when each image was taken and how it was posed, relative to the pattern.

You can assume that the pattern is at 0,0,0 in some global coordinate system and are thus looking for the x, y, z and yaw, pitch, roll of the camera that took each image. Please submit a link to a Github repository contain the code for your solution. Readability and comments are taken into account too. You may use 3rd party libraries like OpenCV and Numpy. 

## Prerequisites

- python-2.7.11
- python packages - cv2, matplotlib, numpy

## How to Run

- Clone the repository
- Change the camera calibration matrix with respect to your camera (Default : Iphone 6 camera parameters)
- Store the images in Images folder
- Run the `visulalize.py` file
- 3D graphs stored in Output folder

## Algorithm

- Found the QR code vertices in the image using Canny edge detector and Contours
- Used opencv SolvePnP function to calculate the rotation and translation vector
- Used rotation and translation vector to plot camera position with respect to image

