
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import imutils
import time
import cv2.cv2 as cv2
from google.cloud import vision
import os
import io
import base64
import threading
from PIL import Image
import math

min_Area = 200
min_Confidence = 0.2
adjustment_Factor_x = 0.3
adjustment_Factor_y = 0.6
offline_Detection = True
i=0

save_file_path = "extracted_images"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "Cerebral-24ef0ec93035.json"
"""Detects text in the file."""
client = vision.ImageAnnotatorClient()

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

# start the FPS throughput estimator
fps = FPS().start()

firstFrame = None
frame = None

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
	
def text_recognition_video(frame):
	response = []	
	texts = []
	# cv2.imwrite(save_file_path + "//"+"cropped.png",frame)
	frame2 = cv2.imencode(".jpg",frame)[1].tostring()		
	image = vision.types.Image(content=frame2)
	
	response = client.text_detection(image = image)
	texts = response.text_annotations
	
	for text in texts:
		print(format(texts[0].description))

	return texts

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
	# print(scores[0][0][0][0])
	# print("---")
	
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
			if scoresData[x] < min_Confidence:
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
			startX = startX - int(numCols*adjustment_Factor_x)
			startY = startY - int(numRows*adjustment_Factor_y)
			endX = endX + int(numCols*adjustment_Factor_x)
			endY = endY + int(numRows*adjustment_Factor_y)

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

def text_detection(frame):
	# construct a blob from the frame and then perform a forward pass
	# of the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(frame, 1.0, (np.shape(frame)[1], np.shape(frame)[0]),
			(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	return(net.forward(layerNames))

def decode_frame(encoded):
	decoded_byte = base64.b64decode(encoded)
	decoded = np.frombuffer(decoded_byte, dtype=np.uint8)
	return(cv2.imdecode(decoded, flags=1))

def resize_frame(frame):
	global rW,rH
	(H, W) = frame.shape[:2]
	newW = 320
	newH = 320
	rW = W / float(newW)
	rH = H / float(newH)

	# resize the image and grab the new image dimensions
	return(cv2.resize(frame, (newW, newH)))

def crop_save(frame, boxes):
	global i
	for (startX, startY, endX, endY) in boxes:
			if(abs(startY-startX)*abs(endX-endY)>10):
					# print(startX, startY, endX, endY)
					imcrop = frame[startY: endY ,startX: endX]

					if(np.size(imcrop)>10):
						# imcrop2 = cv2.resize(imcrop, (0,0), fx=5, fy=5)
						# imcrop2 = cv2.cvtColor(imcrop2,cv2.COLOR_BGR2GRAY)
						# imcrop2 = cv2.adaptiveThreshold(imcrop2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,6)
						# # imcrop2 = cv2.threshold(imcrop2, 0, 255, cv2.THRESH_OTSU)[1]
						
						# cv2.imwrite(save_file_path + "//" + str(i)+".png",imcrop)
						# i = i+1
						t0 = threading.Thread(target=text_recognition_video, args=(imcrop,))
						t0.start()
						return imcrop	

def resized_boxes(boxes):
	boxes[:,0] = boxes[:,0] * rW
	boxes[:,1] = boxes[:,1] * rH
	boxes[:,2] = boxes[:,2] * rW
	boxes[:,3] = boxes[:,3] * rH
	return boxes.astype(int)

# Main Algorithm
def imageProcessor(encoded, min_confidence = min_Confidence, min_area = min_Area, adjustment_factor_x = adjustment_Factor_x, adjustment_factor_y = adjustment_Factor_y, offline_detection = offline_Detection):
	global frame,i, min_Area, min_Confidence, adjustment_Factor_x, adjustment_Factor_y
	adjustment_Factor_x = adjustment_factor_x
	adjustment_Factor_y = adjustment_factor_y
	min_Area= min_area
	min_Confidence = min_confidence
	offline_Detection = offline_detection
	# Decoding frame
	frame = decode_frame(encoded)

	# resizing frame
	frame = resize_frame(frame)
	
	if(offline_Detection == True):
		## Check for motion
		if (motion_detection(frame) == True or min_area==0):
			# print ("motion detected")
			
			(scores, geometry) = text_detection(frame)

			# decode the predictions, then  apply non-maxima suppression to
			# suppress weak, overlapping bounding boxes
			(rects, confidences) = decode_predictions(scores, geometry)
			
			boxes = non_max_suppression(np.array(rects), probs=confidences)	
			if(np.size(boxes)>1):
				crop_save(frame,boxes)

				return(resized_boxes(boxes))
	else:
		t0 = threading.Thread(target=text_recognition_video, args=(frame,))
		t0.start()
	
	return []