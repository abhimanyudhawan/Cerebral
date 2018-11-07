import argparse
import imutils
import time
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from google.cloud import vision

import sys
##reload (sys)
##sys.setdefaulencoding('utf8')

image_file_path = "final.jpg"
firstFrame = None

min_area = 20000
erc1 = cv2.text.loadClassifierNM1("/home/pi/Documents/project1/trained_classifierNM1.xml")
er1 = cv2.text.createERFilterNM1(erc1, 80, 0.00015, 0.13, 0.8, True, 0.1)
erc2 = cv2.text.loadClassifierNM2("/home/pi/Documents/project1/trained_classifierNM2.xml")
er2 = cv2.text.createERFilterNM2(erc2, 0.99)

def start():
    # construct the argument parser and parse the arguments
    camera = cv2.VideoCapture(1)
    return camera

def stop(camera):
    camera.release()
    cv2.destroyAllWindows()

def save(frame_original):
    global image_path
    if(cv2.imwrite(image_path, frame_original)):
        return(1)

def motion_detection(gray):
    global firstFrame
    motion_detected = False
    cv2.imshow("first_frame",firstFrame)
    
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta,120,255,cv2.THRESH_BINARY)[1]
    
    thresh = cv2.dilate(thresh, None, iteration
