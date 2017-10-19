#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:36:49 2017

@author: lakshya
"""

import math

# =============================================================================
# Utility functions used in the main routine
# =============================================================================
    
#Function to give slope of line, given two points
def cv2Slope(point_1, point_2):
    dx = point_2[0] - point_1[0]
    dy = point_2[1] - point_1[1]
    
    #To make sure not dividing by zero and passing 0 or 1 as flag value
    if dy != 0:
        return dy/dx, 1
    else:
        return 0.0, 0

#Function to get Distance between two points
def cv2Distance(point_1, point_2):
    return math.sqrt( math.pow(point_1[0] - point_2[0], 2) + math.pow(point_1[1] - point_2[1], 2) )

#Refered to this link : http://nghiaho.com/?page_id=846
def rotationMatrixToEulerAngles(rotation):
    
    # Angles of each axis
    x = math.atan2(rotation[2][1] , rotation[2][2])
    y = math.atan2(-rotation[2][0] , math.sqrt(math.pow(rotation[2][1], 2) + math.pow(rotation[2][2], 2)))
    z = math.atan2(rotation[1][0] , rotation[0][0])
    
    #Convert each angle to degree format
    x = x * (180/math.pi)
    y = y * (180/math.pi)
    z = z * (180/math.pi)
    
    return [x,y,z]