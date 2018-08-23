#!/usr/bin/python

import sys
import os

import cv2 as cv
import numpy as np

print('\ndetect_er_chars.py')
print('       A simple demo script using the Extremal Region Filter algorithm described in:')
print('       Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012\n')

cap = cv.VideoCapture(0)

pathname = "image.png"
while(True):
  img  = cv.resize(cap.read()[1],(150,150))
##  rows,cols = img.shape
  M = cv.getRotationMatrix2D((150/2,150/2),90,1)
  img = cv.warpAffine(img,M,(150,150))
  vis      = img.copy()


  # Extract channels to be processed individually
  channels = cv.text.computeNMChannels(img)
  # Append negative channels to detect ER- (bright regions over dark background)
  cn = len(channels)-1
  for c in range(0,cn):
    channels.append((255-channels[c]))

  # Apply the default cascade classifier to each independent channel (could be done in parallel)
  print("Extracting Class Specific Extremal Regions from "+str(len(channels))+" channels ...")
  print("    (...) this may take a while (...)")
  for channel in channels:
    erc1 = cv.text.loadClassifierNM1('/home/pi/Documents/project1/trained_classifierNM1.xml')
    er1 = cv.text.createERFilterNM1(erc1,90,0.00015,0.13,0.6,True,0.1)
    
    erc2 = cv.text.loadClassifierNM2('/home/pi/Documents/project1/trained_classifierNM2.xml')
    er2 = cv.text.createERFilterNM2(erc2,0.8)

    regions = cv.text.detectRegions(channel,er1,er2)
    if(regions):
      rects = cv.text.erGrouping(img,channel,[r.tolist() for r in regions])
    #rects = cv.text.erGrouping(img,channel,[x.tolist() for x in regions], cv.text.ERGROUPING_ORIENTATION_ANY,'../../GSoC2014/opencv_contrib/modules/text/samples/trained_classifier_erGrouping.xml',0.5)

    #Visualization
    for r in range(0,np.shape(rects)[0]):
      rect = rects[r]
      cv.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (0, 0, 0), 2)
      cv.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (255, 255, 255), 1)

    #Visualization
  vis = cv.warpAffine(vis,M,(150,150))
  vis = cv.warpAffine(vis,M,(150,150))
  vis = cv.warpAffine(vis,M,(150,150))
  cv.imshow("Text detection result", vis)
  if cv.waitKey(1) & 0xFF == ord('q'):
        break
  
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
