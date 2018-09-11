#!/usr/bin/python

import sys
import os
import time
import cv2 as cv


camera = cv.VideoCapture(0)
image_path = "final.png"
time.sleep(0.1)
while(True):
    gb, frame = camera.read()
##    cv.imwrite(image_path,frame)
    cv.imshow("Original", frame)
    img  = frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    erc1 = cv.text.loadClassifierNM1('/home/pi/Documents/project1/trained_classifierNM1.xml')
    er1 = cv.text.createERFilterNM1(erc1, 90, 0.00045, 0.13, 0.9, True, 0.1)

    erc2 = cv.text.loadClassifierNM2('/home/pi/Documents/project1/trained_classifierNM2.xml')
    er2 = cv.text.createERFilterNM2(erc2, 0.99)

    regions = cv.text.detectRegions(gray,er1,er2)

    #Visualization
    rects = [cv.boundingRect(p.reshape(-1, 1, 2)) for p in regions]
    for rect in rects:
      cv.rectangle(img, rect[0:2], (rect[0]+rect[2],rect[1]+rect[3]), (0, 0, 0), 2)
    for rect in rects:
      cv.rectangle(img, rect[0:2], (rect[0]+rect[2],rect[1]+rect[3]), (255, 255, 255), 1)
    cv.imshow("Text detection result", img)
    time.sleep(0.1)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
camera.release()
cv.destroyAllWindows()