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

choices = ["Polaroids and Excitors", "The Fermi Surface", "Ultrasonic Methods in Solid Physics", "Wert and Thomson Physics of solids", "Acoustic fields and waves in solids","Light Scattering in Solids", "Fundamentals of Adhesion and Interactions","ill-condensed matter","Glassy Materials and disordered solids"]

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/pi/Downloads/indoorbuddy.json"

client = vision.ImageAnnotatorClient()


image_path = "final.png"

firstFrame = None
gray=None
gray_original = None
frame = None
frame_original = None
camera = None
vis = None

min_area = 10000
##erc1 = cv2.text.loadClassifierNM1('/home/pi/Documents/project1/trained_classifierNM1.xml')
##er1 = cv2.text.createERFilterNM1(erc1, 90, 0.00015, 0.013, 0.6, True, 0.1)
##erc2 = cv2.text.loadClassifierNM2('/home/pi/Documents/project1/trained_classifierNM2.xml')
##er2 = cv2.text.createERFilterNM2(erc2, 0.8)


motion_detected = False
text_detected = False
text_recognised = False
captured = False
i=0
def start():
    global camera 
    # construct the argument parser and parse the arguments
    camera = cv2.VideoCapture(0)
    # initialize the first frame in the video stream
    return camera

def stop(camera):
    camera.release()
    cv2.destroyAllWindows()

def save():
    global image_path,gray_original,vis
    if(cv2.imwrite(image_path, gray_original)):
        return(1)

def motion_detection():
    global firstFrame, motion_detected,gray
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 90, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=1)
    (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
            
    # loop over the contours
    for c in cnts:
        if cv2.contourArea(c) > min_area:
            motion_detected = True
        else:
            moion_detected = False
    if motion_detected:
        print("motion detected", random.random())
        firstFrame = gray
        return True

def captures(camera):
    global firstFrame,gray, gray_original, frame, frame_original,captured
    (grabbed, frame_original) = camera.read()
    if(grabbed):
        captured = True
        frame = imutils.resize(frame_original, width=500)
        gray_original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # if the first frame is None, initialize it
        if firstFrame is None:
            firstFrame = gray
        
        cv2.imshow("output2", gray_original)
    else:
        captured = False


def text_detection():
    th = threading.Thread()
    global text_detected,frame,vis,image_path
    img  = cv2.resize(frame,(150,150))
    ##  rows,cols = img.shape
    M = cv2.getRotationMatrix2D((150/2,150/2),90,1)
    img = cv2.warpAffine(img,M,(150,150))
    vis      = img.copy()


    # Extract channels to be processed individually
    channels = cv2.text.computeNMChannels(img)
    # Append negative channels to detect ER- (bright regions over dark background)
    cn = len(channels)-1
    for c in range(0,cn):
        channels.append((255-channels[c]))

    for channel in channels:
        erc1 = cv2.text.loadClassifierNM1('/home/pi/Documents/project1/trained_classifierNM1.xml')
        er1 = cv2.text.createERFilterNM1(erc1,30,0.00015,0.13,0.6,True,0.1)

        erc2 = cv2.text.loadClassifierNM2('/home/pi/Documents/project1/trained_classifierNM2.xml')
        er2 = cv2.text.createERFilterNM2(erc2,0.88)

        regions = cv2.text.detectRegions(channel,er1,er2)
        if(regions):
          rects = cv2.text.erGrouping(img,channel,[r.tolist() for r in regions])
        #rects = cv2.text.erGrouping(img,channel,[x.tolist() for x in regions], cv2.text.ERGROUPING_ORIENTATION_ANY,'../../GSoC2014/opencv2_contrib/modules/text/samples/trained_classifier_erGrouping.xml',0.5)

    #Visualization
    for r in range(0,np.shape(rects)[0]):
      rect = rects[r]
      cv2.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (0, 0, 0), 2)
      cv2.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (255, 255, 255), 1)

##    #Visualization
##    vis = cv2.warpAffine(vis,M,(200,200))
##    vis = cv2.warpAffine(vis,M,(200,200))
##    vis = cv2.warpAffine(vis,M,(200,200))
##    cv2.imshow("Text detection result", vis)
##    frame = vis
##    cv2.imshow("output1", vis)
    #cv2.imshow("Text detection result", vis)
    if(np.shape(rects)[0]>=1):
        vis = cv2.warpAffine(vis,M,(150,150))
        vis = cv2.warpAffine(vis,M,(150,150))
        vis = cv2.warpAffine(vis,M,(150,150))
        cv2.imshow("Text detection result", vis)
##        global image_path,gray_original,vis
##        if(cv2.imwrite(image_path, gray_original)):
##            return(1)
##        print("working",threading.activeCount())
##        th = threading.Thread(target=text_recognition, args=(image_path,))
##        th.start()
        text_detected = True
        print("text_detected")
    else:
        text_detected = False
        

def text_recognition(image_path):
    global text_recognised,i
    i=i+1
    print(i)
    if(text_recognised == False):
        # [START migration_text_detection]
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.types.Image(content=content) 
        texts = []
        response = client.text_detection(image=image)
        texts = response.text_annotations
        for text in texts:
            print(format(texts[0].description))
        if len(texts)>0:
            text_recognised = True        
            
if __name__=='__main__':
    camera = start()
    global motion_detected, text_detected, text_recognised, image_path
    global gray, gray_original, frame, frame_original,i
    t1 = threading.Thread()
    t2 = threading.Thread()
    t3 = threading.Thread()
    t4 = threading.Thread()
    t5 = threading.Thread()
    t6 = threading.Thread()
    repeat = False
    stopped = False
    
    while True:
##        print(threading.activeCount())
        captures(camera)
        if(t1.isAlive()==False): 
            t1 = threading.Thread(target=motion_detection, args=())
            t1.start()
        
        if(motion_detected or repeat):
            repeat = False
            if(t3.isAlive()==False): 
                t3 = threading.Thread(target=text_detection, args=())
                t3.start()
            elif(t4.isAlive()==False): 
                t4 = threading.Thread(target=text_detection, args=())
                t4.start()
                
        if(text_detected):
            stopped=True
        else:
            stopped = False
                
        if(stopped== True and text_recognised == False):
            print("here")
            stopped = False
            motion_detected = False
            text_detected = False
            if(save()==1):
                if (i<1):
                    repeat = True
                    if(t5.isAlive()==False):
                        print("working",threading.activeCount())
                        t5 = threading.Thread(target=text_recognition, args=(image_path,))
                        t5.start()
                        i=i+1
                    elif(t6.isAlive()==False):
                        print("working",threading.activeCount())
                        t6 = threading.Thread(target=text_recognition, args=(image_path,))
                        t6.start()
                        i=i+1                            
                    
                else:
                    text_detected = False
                    motion_detected = False
                    stopped = False
                    repeat = False
                    
                        
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
