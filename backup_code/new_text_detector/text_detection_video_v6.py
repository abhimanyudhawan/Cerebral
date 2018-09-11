# USAGE
# python text_detection_video.py --east frozen_east_text_detection.pb --video Cerebral.mp4

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

min_Area = 900
i=0
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Cerebral-24ef0ec93035.json"
"""Detects text in the file."""
client = vision.ImageAnnotatorClient()

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-east", "--east", type=str, required=True,
	help="path to input EAST text detector")
ap.add_argument("-v", "--video", type=str,
	help="path to optinal input video file")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=192,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=192,
	help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

# initialize the original frame dimensions, new frame dimensions,
# and ratio between the dimensions
(W, H) = (None, None)
(newW, newH) = (args["width"], args["height"])
(rW, rH) = (None, None)

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

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

firstFrame = None
test_text = "hello"
frame = np.zeros((newH, newW, 1), dtype = "uint8")
image_filepath = "test.jpg"

def make_request(frame2):
	frame2.read()
	return{
				"image":{
	    				"content": image
	    			},
				"features": [
      					{
      						"type":"TEXT_DETECTION",
      						"maxResults": 10
      					}
      				]
			}
	
def text_recognition_video():
	response = []	
	texts = []
	cv2.imwrite(image_filepath,orig)
	with io.open(image_filepath, "rb") as imageFile:
		frame2 = imageFile.read()
		
	image = vision.types.Image(content=frame2)

	
	response = client.text_detection(image = image)
	texts = response.text_annotations
	# print (texts)
	for text in texts:
		print(format(texts[0].description))

	# body = vision.types.AsyncAnnotateFileRequest() 
	# body = make_request(frame2)
	# response = client.async_batch_annotate_files(body)
	# response.add_done_callback(show_results)
	# response = client.text_detection(image=image)
	

	# for text in texts:
	# 	print(format(texts[0].description))


def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < args["min_confidence"]:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# increase or decrease size of detection boxes
			startX = startX - orig.shape[0]/5
			startY = startY - orig.shape[1]/50
			endX = endX + orig.shape[0]/5
			endY = endY + orig.shape[1]/50

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

def motion_detection(frame):
	global firstFrame
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray,(21,21),0)
	if firstFrame is None:
		firstFrame = gray.copy()
		print("firstFrame assigned gray")
	else:
		frameDelta = cv2.absdiff(firstFrame,gray)
		thresh = cv2.threshold(frameDelta,90,255,cv2.THRESH_BINARY)[1]
		thresh = cv2.dilate(thresh, None, iterations =1)
		(_,cnts,_) = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		for c in cnts:
			if cv2.contourArea(c)>min_Area:
				firstFrame = gray.copy()
				return True
	return False
	
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
	frame = imutils.resize(frame, width=500)
	orig = frame.copy()
	# if our frame dimensions are None, we still need to compute the
	# ratio of old frame dimensions to new frame dimensions
	if W is None or H is None:
		(H, W) = frame.shape[:2]
		rW = W / float(newW)
		rH = H / float(newH)

	# resize the frame, this time ignoring aspect ratio
	frame = cv2.resize(frame, (newW, newH))

	##Check for motion
	if (motion_detection(frame) == True):
		# print ("motion detected")
		# construct a blob from the frame and then perform a forward pass
		# of the model to obtain the two output layer sets
		blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
			(123.68, 116.78, 103.94), swapRB=True, crop=False)
		net.setInput(blob)
		(scores, geometry) = net.forward(layerNames)

		# decode the predictions, then  apply non-maxima suppression to
		# suppress weak, overlapping bounding boxes
		(rects, confidences) = decode_predictions(scores, geometry)
		
		boxes = non_max_suppression(np.array(rects), probs=confidences)

		# loop over the bounding boxes
		for (startX, startY, endX, endY) in boxes:
			# scale the bounding box coordinates based on the respective
			# ratios
			startX = int(startX * rW)
			startY = int(startY * rH)
			endX = int(endX * rW)
			endY = int(endY * rH)
			# clip coordinates between min and max
			np.clip(startX,0,orig.shape[1])
			np.clip(startY,0,orig.shape[0])
			np.clip(endX,0,orig.shape[1])
			np.clip(endY,0,orig.shape[0])

			# Select region of interest
			if(abs(startY-startX)*abs(endX-endY)>10):
				imcrop = orig[startY: endY ,startX: endX]
				if(np.size(imcrop)>10):
					cv2.imshow(str(i),imcrop)
					cv2.imwrite(str(i),imcrop)
					i = i+1
					cv2.destroyWindow(str(i-3))

			# t0 = threading.Thread(target=text_recognition_video, args=())
			# t0.start()
			# draw the bounding box on the frame
			# cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
		cv2.imshow("Text Detection", orig)
		# text_recognition_video()

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
