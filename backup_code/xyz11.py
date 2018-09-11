#!/usr/bin/python

import argparse
import datetime
import imutils
import time
import cv2
import numpy as np
from google.cloud import vision
import sys
import codecs
import io
import os
import time
from PIL import Image, ImageEnhance, ImageFilter
import random
import threading
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
global text_detected, i,t5,t6,er1,er2,frame
global text_recognised,image_path
global firstFrame,gray, gray_original, frame, frame_original,t1,t2

firstFrame = None

choices = ["Polaroids and Excitors", "The Fermi Surface", "Ultrasonic Methods in Solid Physics", "Wert and Thomson Physics of solids", "Acoustic fields and waves in solids","Light Scattering in Solids", "Fundamentals of Adhesion and Interactions","ill-condensed matter","Glassy Materials and disordered solids"]

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Cerebral-24ef0ec93035.json"
"""Detects text in the file."""
client = vision.ImageAnnotatorClient()
# encoding=utf8
##import sys
##reload(sys)
##sys.setdefaultencoding('utf8')
##
image_path = "final.png"

firstFrame = None
gray=None
gray_original = None
frame = None
frame_original = None

i =0

t0 = threading.Thread()
t1 = threading.Thread()
t2 = threading.Thread()
t3 = threading.Thread()
t4 = threading.Thread()
t5 = threading.Thread()
t6 = threading.Thread()
##    Create an Extremal Region Filter for the 1st stage classifier of N&M algorithm
##    Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012
##    The component tree of the image is extracted by a threshold increased step by step
##    from 0 to 255, incrementally computable descriptors (aspect_ratio, compactness,
##    number of holes, and number of horizontal crossings) are computed for each ER
##    and used as features for a classifier which estimates the class-conditional
##    probability P(er|character). The value of P(er|character) is tracked using the inclusion
##    relation of ER across all thresholds and only the ERs which correspond to local maximum
##    of the probability P(er|character) are selected (if the local maximum of the
##    probability is above a global limit pmin and the difference between local maximum and
##    local minimum is greater than minProbabilityDiff).
##    \param  cb                Callback with the classifier.
##                              default classifier can be implicitly load with function loadClassifierNM1()
##                              from file in samples/cpp/trained_classifierNM1.xml
##    \param  thresholdDelta    Threshold step in subsequent thresholds when extracting the component tree
##    \param  minArea           The minimum area (% of image size) allowed for retrieved ER's
##    \param  minArea           The maximum area (% of image size) allowed for retrieved ER's
##    \param  minProbability    The minimum probability P(er|character) allowed for retrieved ER's
##    \param  nonMaxSuppression Whenever non-maximum suppression is done over the branch probabilities
##    \param  minProbability    The minimum probability difference between local maxima and local minima ERs
##*/createERFilterNM1(const Ptr<ERFilter::Callback>& cb, int thresholdDelta,
##                                float minArea, float maxArea, float minProbability,
##bool nonMaxSuppression, float minProbabilityDiff)
min_area = 4000
erc1 = cv2.text.loadClassifierNM1('trained_classifierNM1.xml')
er1 = cv2.text.createERFilterNM1(erc1, 90, 0.00015, 0.013, 0.6, True, 0.1)
erc2 = cv2.text.loadClassifierNM2('trained_classifierNM2.xml')
er2 = cv2.text.createERFilterNM2(erc2, 0.8)

motion_detected = False
text_detected = False
text_recognised = False

def start():
    # construct the argument parser and parse the arguments
    camera = cv2.VideoCapture("Cerebral.mp4")
    # initialize the first frame in the video stream
    return camera

def stop(camera):
    camera.release()
    cv2.destroyAllWindows()

def save(frame):
    global image_path
   ## time.sleep(0.0001)
    if(cv2.imwrite(image_path, frame)):
##        cv2.imshow('my_frame2', cv2.imread(image_path))
        return(1)

def motion_detection():
    global firstFrame, motion_detected,gray,t3,t4

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
        else:
            motion_detected = False

    if motion_detected:
        print("motion detected", random.random())
        
##        if(t3.isAlive()==False): 
##            t3 = threading.Thread(target=text_detection, args=())
##            t3.start()
##        
##        elif(t4.isAlive()==False): 
##            t4 = threading.Thread(target=text_detection, args=())
##            t4.start()
      
        firstFrame = gray
        return True

def captures(camera):
    global text_detected, i,t5,t6,er1,er2,frame
    global text_recognised,image_path
    global firstFrame,gray, gray_original, frame, frame_original,t1,t2
    global firstFrame
    grabbed = 0
    while(grabbed == 0 ):
        (grabbed, frame_original) = camera.read()
    # if the frame could not be grabbed, then we have reached the end
    # of the video

    # resize the frame, convert it to grayscale, and blur it
    frame = cv2.resize(frame_original,(300,300))
    gray_original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
    
    cv2.imshow("output2", frame)
    if(t1.isAlive()==False): 
        t1 = threading.Thread(target=motion_detection, args=())
        t1.start()
        
    elif(t2.isAlive()==False): 
        t2 = threading.Thread(target=motion_detection, args=())
        t2.start()

##    return gray, frame_original,gray_original

def text_detection():
    global text_detected, i,t5,t6,er1,er2,frame
    global text_recognised,image_path,camera
    global firstFrame,gray, gray_original, frame, frame_original,t1,t2
    
    img  = frame
    M = cv2.getRotationMatrix2D((300/2,300/2),90,1)
    img = cv2.warpAffine(img,M,(300,300))
    vis = img.copy()
    
    # Extract channels to be processed individually
    channels = cv2.text.computeNMChannels(img)
    # Append negative channels to detect ER- (bright regions over dark background)
    cn = len(channels)-1
    for c in range(0,cn):
        channels.append((255-channels[c]))

      # Apply the default cascade classifier to each independent channel (could be done in parallel)
    print("Extracting Class Specific Extremal Regions from "+str(len(channels))+" channels ...")
    print("    (...) this may take a while (...)")
    for channel in channels:
        ##erc1 = cv2.text.loadClassifierNM1('/home/abhimanyu/c++/opencv_contrib-master/modules/text/samples/trained_classifierNM1.xml')
        er1 = cv2.text.createERFilterNM1(erc1,90,0.00015,0.013,0.6,True,0.1)

        ##erc2 = cv2.text.loadClassifierNM2('/home/abhimanyu/c++/opencv_contrib-master/modules/text/samples/trained_classifierNM2.xml')
        er2 = cv2.text.createERFilterNM2(erc2,0.8)

        regions = cv2.text.detectRegions(channel,er1,er2)
        
    if(regions):
        rects = cv2.text.erGrouping(img,channel,[r.tolist() for r in regions])
    else:
        rects = 0
        #Visualization
    for r in range(0,np.shape(rects)[0]):
        rect = rects[r]
        cv2.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (0, 0, 0), 2)
        cv2.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (255, 255, 255), 1)
         #Visualization   
    
    if(np.shape(rects)[0]>=1):
        vis = cv2.warpAffine(vis,M,(300,300))
        vis = cv2.warpAffine(vis,M,(300,300))
        vis = cv2.warpAffine(vis,M,(300,300))
        cv2.imshow("Text detection result", vis) 
        text_detected = True
        print("text_detected")
    else:
        text_detected = False
        
    if(text_detected):
        if(i<3):
            if(t5.isAlive()==False):
                print("working",threading.activeCount())
                t5 = threading.Thread(target=text_recognition, args=())
                t5.start()
                i=i+1
                
            elif(t6.isAlive()==False):
                print("working",threading.activeCount())
                t6 = threading.Thread(target=text_recognition, args=())
                t6.start()
                i=i+1

def text_recognition():
    global text_detected, i,t5,t6,er1,er2,frame
    global text_recognised,image_path,camera
    global firstFrame,gray, gray_original, frame, frame_original,t1,t2
    if(text_recognised == False):
    
        # [START migration_text_detection]
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.types.Image(content=content)
        
        texts = []
        response = client.text_detection(image=image)
        texts = response.text_annotations
    ##    print("texte")
        for text in texts:
            # print("hello")
            #text.encode('utf-8').strip()
            print(format(texts[0].description))
            #print(process.extractOne('\n"{}"'.format(texts[0].description),choices, scorer=fuzz.token_sort_ratio))

            # vertices = (['({},{})'.format(vertex.x, vertex.y)
            #             for vertex in text.bounding_poly.vertices])

            # print('bounds: {}'.format(','.join(vertices)))
        if len(texts)>0:
            text_recognised = True
        else:
            text_recognised = False


if __name__=='__main__':
    camera = start()
    while True:
        if(t0.isAlive()==False):
            t0 = threading.Thread(target=captures, args=(camera,))
            t0.start()
                        
        if(text_recognised):
            i = 0
            repeat = False
            stopped = False
            text_recognised = False
            text_detected = False
            motion_detected = False
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
                break
    
    stop(camera)

