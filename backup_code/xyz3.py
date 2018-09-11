#!/usr/bin/python

import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
from google.cloud import vision
import io
import os
import time
from PIL import Image, ImageEnhance, ImageFilter
import random


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/pi/Downloads/My First Project.json"
"""Detects text in the file."""
client = vision.ImageAnnotatorClient()
# encoding=utf8
##import sys
##reload(sys)
##sys.setdefaultencoding('utf8')
##
image_path = "final.jpg"
firstFrame = None

min_area = 8000
erc1 = cv2.text.loadClassifierNM1('/home/pi/Documents/project1/trained_classifierNM1.xml')
er1 = cv2.text.createERFilterNM1(erc1, 80, 0.00015, 0.13, 0.5, True, 0.1)
erc2 = cv2.text.loadClassifierNM2('/home/pi/Documents/project1/trained_classifierNM2.xml')
er2 = cv2.text.createERFilterNM2(erc2, 0.8)

motion_detected = False
text_detected = False
text_recognised = False

def start():

    # construct the argument parser and parse the arguments
    camera = cv2.VideoCapture(0)
    # initialize the first frame in the video stream
    return camera

def stop(camera):
    camera.release()
    cv2.destroyAllWindows()

def save(gray_original):
    global image_path
    if(cv2.imwrite(image_path, gray_original)):
##        cv2.imshow('my_frame2', cv2.imread(image_path))
        return(1)

def motion_detection(gray):
    global firstFrame, motion_detected
    cv2.imshow("2first_frame2",firstFrame)
    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 90, 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=1)
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
            
    # loop over the contours
    for c in cnts:
        if cv2.contourArea(c) > min_area:
                motion_detected = True
    # if the contour is too small, ignore it
    # compute the bounding box for the contour, draw it on the frame,
    # and update the text
    # (x, y, w, h) = cv2.boundingRect(c)
    # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    if motion_detected:
        print ("motion is there", random.random())
        firstFrame = gray

def captures(camera):
    global firstFrame
    (grabbed, frame_original) = camera.read()
    # if the frame could not be grabbed, then we have reached the end
    # of the video

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame_original, width=500)
    gray_original = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
    
    cv2.imshow("2output2", frame_original)

    return gray, frame_original,gray_original

def text_detection(gray_original,er1,er2):
    global text_detected
##    im = Image.open(image_path) # the second one
####    im = im.filter(ImageFilter.BLUR())
##    im = im.filter(ImageFilter.MedianFilter())
##    enhancer = ImageEnhance.Contrast(im)
##    im = enhancer.enhance(2)
##    print(type(im))
##    im = im.convert('1')
##    im.save(image_path)
##    gray_original = cv2.imread(image_path)
##    gray_original = cv2.cvtColor(np.array(im),cv2.COLOR_2GRAY)
##    cv2.imshow('converted',gray_new)
##    gray_original = cv2.GaussianBlur(gray_original, (21, 21), 0)
    cv2.imshow('2converted',gray_original)
    regions = cv2.text.detectRegions(gray_original,er1,er2)
    if(len(regions)>=3):
        text_detected = True
        # print (rects)
        # for rect in rects:
        # 	# cv2.rectangle(frame,rect[0:2],(rect[0]+rect[2],rect[1]+rect[3],(0,0,0),2))
        # 	text_detected = True

        # for rect in rects:
        # 	# cv2.rectangle(frame, rect[0:2], (rect[0]+rect[2],rect[1]+rect[3]), (255, 255, 255), 1)
        # 	text_detected = True

    if(text_detected):
        print("text is there")
        return 1

def text_recognition(image_path):
    global text_recognised
##    im = Image.open(image_path) # the second one
##    im = im.filter(ImageFilter.BLUR())
##    im = im.filter(ImageFilter.MedianFilter())
##    enhancer = ImageEnhance.Contrast(im)
##    im = enhancer.enhance(2)
##    im = im.convert('1')
##    im.save(image_path)
    
    # [START migration_text_detection]
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)
    
    texts = []
    response = client.text_detection(image=image)
    texts = response.text_annotations
    print("texte")
    for text in texts:
        # print("hello")
        # text.encode('utf-8').strip()
        print('\n"{}"'.format(text.description).encode('utf-8'))

        # vertices = (['({},{})'.format(vertex.x, vertex.y)
        #             for vertex in text.bounding_poly.vertices])

        # print('bounds: {}'.format(','.join(vertices)))
    if len(texts)>0:
        text_recognised = True

if __name__=='__main__':
    camera = start()
    global motion_detected, text_detected, text_recognised
    i=0
    while True:
        gray, frame_original,gray_original = captures(camera)
        motion_detection(gray)
        
        if(motion_detected):
            motion_detected = False
            text_detection (gray_original,er1,er2)   
            if(text_detected):
                text_detected = False
                if(save(frame_original)==1):
                    text_recognition(image_path)
                    if(text_recognised):
                        i=0
                        text_recognised = False
                        print("done")
                    elif (i<5):
                        i=i+1
                        motion_detected =True
                        text_detected = True
                    else:
                        i=0
                        motion_detected =False
                        text_detected = False
                        
##                    else:
##                        for i in range(0,3):
##                            gray, frame_original,gray_original = captures(camera)
##                            save(frame_original)
##                            text_recognition(image_path)                   
        
##                            while(save(frame_original)!=1):
##                                        print("not done")

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
                break
    
    stop(camera)


