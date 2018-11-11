# USAGE
# python video_client.py --video Cerebral.mp4

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import time
import cv2.cv2 as cv2
from google.cloud import vision
import os
import io
import base64
import threading
import text_detection_v10

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-east", "--east", type=str, required=False,
	help="path to input EAST text detector")
ap.add_argument("-v", "--video", type=str,
	help="path to optinal input video file")
ap.add_argument("-c", "--min-confidence", type=float, default=0.7,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=192,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=192,
	help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

min_area = 1000
adjustment_factor_x = 0.1
adjustment_factor_y = 0.1

# initialize the original frame dimensions, new frame dimensions,
# and ratio between the dimensions
(W, H) = (None, None)
(newW, newH) = (args["width"], args["height"])
(rW, rH) = (None, None)
encoded = None

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# start the FPS throughput estimator
fps = FPS().start()
rects_out = []
confidences_out = []
	
# loop over frames from the video stream
while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame

	# check to see if we have reached the end of the stream
	if frame is None:
		break

	# resize the frame, maintaining the aspect ratio
	frame = imutils.resize(frame, width=300)
	orig = frame.copy()
	# if our frame dimensions are None, we still need to compute the
	# ratio of old frame dimensions to new frame dimensions
	if W is None or H is None:
		(H, W) = frame.shape[:2]
		rW = W / float(newW)
		rH = H / float(newH)

	# resize the frame, this time ignoring aspect ratio
	frame = cv2.resize(frame, (newW, newH))

	# encoding image to base64 format
	_,encoded_image =cv2.imencode('.jpg',frame)
	encoded_byte = base64.b64encode(encoded_image)
	# f = open('new_text2.txt','wb')
	# f.write(encoded_byte)

	## Calling the actual API
	boxes = text_detection_v10.imageProcessor(encoded_byte, args["min_confidence"], 
													min_area, adjustment_factor_x, adjustment_factor_y, authorization_token='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRhIjoiNWJkZjRhNmJiZDg3ZjMxY2U5MDdiMmMzX180NTk2NzciLCJpYXQiOjE1NDE4MjA2ODgsImV4cCI6MTU0NDQxMjY4OH0.TdOPISz-FEFHePwp-UetWnda6_6Xo2Iv6drJVp8rlz4')
    
    # loop over the bounding boxes
	if(np.shape(boxes)!=None):
		for (startX, startY, endX, endY) in boxes:
			startX = int(startX * rW)
			startY = int(startY * rH)
			endX = int(endX * rW)
			endY = int(endY * rH)

			# draw the bounding box on the frame
			cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
			cv2.imshow("Text Detection", orig)

	# update the FPS counter
	fps.update()
	# show the output frame
	# cv2.imshow("Actual", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# if we are using a webcam, release the pointer
if not args.get("video", False):
	vs.stop()

# otherwise, release the file pointer
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()
