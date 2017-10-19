#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 14:14:19 2017

@author: Lakshya Kejriwal
"""

import cv2
import numpy as np
import glob
import util
import matplotlib.pyplot as plt

# =============================================================================
# Refer this link: http://ksimek.github.io/2013/08/13/intrinsic/
# The camera intrinsic matrix is given as
#  matrix = ( (fx, s , x0) , (0 , fy, y0), (0, 0, 1) )
# For assume iphone6 fx, fy = 35mm, s = 0, CCD width = 24mm
# The camera calibration matrix from opencv can be used to calculate intrinsic matrix
# =============================================================================

camera_matrix = np.array([[2800, 0, 1200], [0, 2800, 1600], [0 , 0 , 1]], dtype = 'float64')

#Assume QR code to be at some point in world coordinate system
QR_world_coordinates = np.array([[-20, 20, 0], [-20, -20, 0], [20, -20, 0], [20, 20, 0]], dtype = 'float32')

#The point used to plot camera frame in figure
camera_frame = [[0, 0, 0], [30, 0, 0], [30, 30, 0], [0, 30, 0], [0, 0, 0]]

#Assume camera distortion to be 0 (can be changed later w.r.t to change in camera)
camera_distortion = np.zeros(4)

#Go through the list to compute each location of camera
for files in glob.glob("Images/*.JPG"):
	
    #Read the image using the filename
    image = cv2.imread(files)
	
    #Convert the image to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # =============================================================================
    #     Step 1:
    #     Finding QR Code in the image (mainly 3 vertices of the QR code - square boxes) - top_left, top_right, bottom_left
    #     Used this link for referecnce: https://github.com/bharathp666/opencv_qr/blob/master/video.cpp
    # =============================================================================
    
    #Apply canny edge detection to find edges in the image
    edges = cv2.Canny(image_gray, 100, 200, 3)
	
    #Find square contours in image
    image_gray , contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Start processing the contour data
    #Finding three squares by iterating over all contours
    mark = 0
    for contour in range(0,len(contours)):
        
        k = contour
        c = 0
        
        #Find the approximated polygon of the contour we are examining
        epsilon = 0.1 * cv2.arcLength(contours[k], True)
        approx = cv2.approxPolyDP(contours[k] , epsilon, True)
        
        #Only quadrilateral contours to be considered
        if len(approx) == 4:
            while(hierarchy[0][k][2] != -1):
    			k = hierarchy[0][k][2]
    			c = c + 1
    		
            if c == 5:
                if mark == 0:
    			A = contour
                #A has been found assign next contour to B
                elif mark == 1:
    			B = contour
                #A and B have been found assign contour to C
                elif mark == 2:
    			C = contour
    		#Track which of A,B,C have been found
                mark = mark+1
	
    all_contours = np.concatenate((contours[A],contours[B],contours[C]))
	
    #Find approximated polygon of 3 square contours
    epsilon = 0.1 * cv2.arcLength(all_contours, True)
    approx = cv2.approxPolyDP(all_contours, epsilon, True)
    
    #3 namely A,B,C 'Alignment Markers' discovered
    A=approx[0][0]
    B=approx[1][0]
    C=approx[2][0]
    	
    #Vertex of the triangle NOT involved in the longest side is the outlier or the fourth vertex
    #Mainly finding the fourth vertex of QR code w.r.t 3 alignment markers
    
    AB = util.cv2Distance(A , B)
    BC = util.cv2Distance(B , C)
    AC = util.cv2Distance(C , A)
    
    #Find the topleft vertex using three distances
    if(AB>BC and AB>AC):
        top = C
        bottom = A
        right = B
    elif(AC>AB and AC>BC):
        top = B
        bottom = A 
        right = C 
    elif(BC>AB and BC>AC):
        top = A 
        bottom = B
        right = C	
    
    #Determine the correct orientation of 'right' and 'bottom' markers
    slope, align = util.cv2Slope(right, bottom)
    perpendicular = np.cross(bottom - top, right - top)
    
    if perpendicular > 0:
        bottom, right = right, bottom
    	     
    #Given 3 alignment marker vertices find 4th vertex point of the QR code
    bottom_right_x = bottom[0] + right[0] - top[0]
    bottom_right_y = bottom[1] + right[1] - top[1]
    bottom_right = [bottom_right_x, bottom_right_y]
    	
    #Step 1 finished, finding the QR code coordinates in the image
    QR_image_coordinates = np.float32([top,bottom,bottom_right,right])
    
    # =============================================================================
    # Step 2: Calculate the relative position of the camera w.r.t to object
    # Use SolvePnP from opencv to calculate rotation and translation vector
    # Refer this link: https://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
    # =============================================================================

    ret, rotation, translation = cv2.solvePnP(QR_world_coordinates, QR_image_coordinates, camera_matrix, camera_distortion)
    	
    rotation, _ = cv2.Rodrigues(rotation)
    	
    #Change rotation and translation from camera coordinate to world coordinate
    rotation = rotation.transpose()
    translation = -np.dot(rotation, translation)
    
    #Plotting 3d graph for visualizing
    figure = plt.figure()
    axis = figure.gca(projection='3d')
    
    #Plot the QR code in the figure
    x=[]
    y=[]
    z=[]
    
    for point in QR_world_coordinates:
        x.append(point[0])
        y.append(point[1])
        z.append(point[2])
    
    #Matplot lib wants the 5th point to make a complete figure, so give back the starting point
    x.append(QR_world_coordinates[0][0])
    y.append(QR_world_coordinates[0][1])
    z.append(QR_world_coordinates[0][2])
    
    axis.plot(x, y, z, label='QR code')
    
    x = []
    y = []
    z = []
    
    for point in camera_frame:
        point=np.array(point)
        out = np.dot(rotation, point.transpose()) + translation.transpose()
        out=out[0]
        x.append(out[0])
        y.append(out[1])
        z.append(out[2])
    
    axis.plot(x, y, z, label=files[-8:-4])
    
    plt.savefig("Output/"+files[-12:])
    
    #Calculating euler angles from rotation matrix
    angle = util.rotationMatrixToEulerAngles(rotation)
    	
    print "For " + files[-8:-4] + " x = %.2f cm, y = %.2f cm, z = %.2f cm, Pitch = %.2f degrees, Yaw = %.2f degrees, Roll = %.2f degrees." % ( translation[0], translation[1], translation[2], angle[0], angle[1], angle[2])
    
