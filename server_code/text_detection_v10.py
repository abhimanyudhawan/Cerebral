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
import requests
import json

url = "http://35.199.46.130/books/search"

min_Area = 50
min_Confidence = 0.1
adjustment_Factor_x = 0.3
adjustment_Factor_y = 0.6
offline_Detection = False
x_Coordinate = 0 
y_Coordinate = 0
z_Coordinate = 0
authorization_Token = '0'
thread_number = 0

save_file_path = "extracted_images\\"
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
# print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

# start the FPS throughput estimator
fps = FPS().start()

firstFrame = {}
recognised_text = {}
output_text = {}

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
	
def text_recognition_video(frame, x_coordinate, y_coordinate, z_coordinate, authorization_token):
	global recognised_text
	# cv2.imwrite(save_file_path + "//"+"cropped.png",frame)
	#frame = imutils.resize(frame, width=200, inter=cv2.INTER_CUBIC)
	#cv2.imwrite("text.jpg",frame)
	frame2 = cv2.imencode(".jpg",frame)[1].tostring()		
	image = vision.types.Image(content=frame2)
	
	response = client.text_detection(image = image)
	texts = response.text_annotations
	
	if(len(texts)>0):
		if(texts[0].description is not None):
			code = texts[0].description.replace("\n", " ")
			print(code.encode("utf-8"))
			recognised_text[authorization_token]=code
			
			#headers = {'Content-Type': "application/json",'authorization': "Bearer "+ str(authorization_token)}
			#payload = {"id" : "5bdf4a6bbd87f31ce907b2c3",
			#			"code" : code,
			#			"isCodeTypeLCClassification" : "true",
			#			"limit" : 10,
			#			"xCordinate" : x_coordinate,
			#			"yCordinate" : y_coordinate,
			#			"zCordinate" : z_coordinate}
			#response = requests.request("POST", url, data=json.dumps(payload), headers=headers)
			# print(response.text.encode("utf-8"))
	return recognised_text[authorization_token]


def decode_predictions(scores, geometry,frame,adjustment_factor_x,adjustment_factor_y,min_confidence):
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
			if scoresData[x] < min_confidence:
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

			# # increase or decrease size of detection boxes
			startX = startX - int(numCols*adjustment_factor_x)
			startY = startY - int(numRows*adjustment_factor_y)
			endX = endX + int(numCols*adjustment_factor_x)
			endY = endY + int(numRows*adjustment_factor_y)

			# add the bounding box coordinates and probability score
			# to our respective lists
			startX = max(startX,0)
			endX = min(endX,np.shape(frame)[0])
			startY = max(startY,0)
			endY = min(endY,np.shape(frame)[1])

			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

def motion_detection(frame,min_area, authorization_token):
	global firstFrame
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray,(21,21),0)
	if authorization_token not in firstFrame:
		firstFrame[authorization_token] = gray.copy()
		# print("firstFrame assigned gray")
		return True		
	else:
		frameDelta = cv2.absdiff(firstFrame[authorization_token],gray)
		thresh = cv2.threshold(frameDelta,90,255,cv2.THRESH_BINARY)[1]
		thresh = cv2.dilate(thresh, None, iterations =1)
		(_,cnts,_) = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		for c in cnts:
			if cv2.contourArea(c)>min_area:
				firstFrame[authorization_token] = gray.copy()
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
	(H, W) = frame.shape[:2]
	frame = imutils.resize(frame, height=500, inter=cv2.INTER_CUBIC)
	# newH = H - H%32
	# newW = W - W%32
	newH = frame.shape[0] - frame.shape[0]%32
	newW = frame.shape[1] - frame.shape[1]%32
	rW = W / float(newW)
	rH = H / float(newH)

	# resize the image and grab the new image dimensions
	return(cv2.resize(frame, (newW, newH)),rW,rH)

def crop_save(frame, boxes, x_coordinate, y_coordinate, z_coordinate, authorization_token):
	global thread_number
	final_image = None
	final_boxes = []
	distance_center_x = np.shape(frame)[1]/5
	distance_bottom_y = np.shape(frame)[0]
	for (startX, startY, endX, endY) in boxes:		
		imcrop = frame[startY: endY ,startX: endX]
		#cv2.imwrite("cropped.jpg",imcrop)
		if(np.size(imcrop)>1):	
			if (1 or abs(np.shape(frame)[1]/2 - abs(startX + endX)/2) < distance_center_x and abs((np.shape(frame)[0]/2 - abs(startY + endY)/2) < distance_bottom_y)):
				distance_bottom_y = abs((np.shape(frame)[0]/2 - abs(startY + endY)/2))
				# print(distance_bottom_y)
				final_boxes= [np.array([startX,startY,endX,endY])]
				final_image = imcrop
						# final_image = imcrop
					# # hsv_image = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
					# # #extract dominant color 
					# # # (aka the centroid of the most popular k means cluster)
					# # dom_color = get_dominant_color(hsv_image, k=3)
					# # dom_color = cv2.cvtColor(np.uint8([[dom_color]]), cv2.COLOR_HSV2BGR)
					# # if(dom_color[0][0][0]>120 and dom_color[0][0][1]>180 and dom_color[0][0][2]>180):
						# distance_center_x = abs(np.shape(frame)[0] - abs(startX + endX)/2)
			# thread_number = thread_number + 1
			# 			# cv2.imwrite(save_file_path + str(thread_number) + ".jpg", imcrop) 
			
						# cv2.imshow('res' + str(thread_number),res) 
	# print ( abs((np.shape(frame)[0]/2 - abs(final_boxes[0][1] + final_boxes[0][3])/2)))
	if(np.size(final_image)>1):
		threading.Thread(target=text_recognition_video, args=(final_image, x_coordinate, y_coordinate, z_coordinate, authorization_token)).start()
	return np.asarray(final_boxes)

def resized_boxes(boxes,rW,rH):
	if(len(np.shape(boxes))==2):
		boxes[:,0] = boxes[:,0] * rW
		boxes[:,1] = boxes[:,1] * rH
		boxes[:,2] = boxes[:,2] * rW
		boxes[:,3] = boxes[:,3] * rH
		return boxes.astype(int)
	return []

# Main Algorithm
def imageProcessor(encoded, min_confidence = min_Confidence, min_area = min_Area, adjustment_factor_x = adjustment_Factor_x, adjustment_factor_y = adjustment_Factor_y, offline_detection = offline_Detection, x_coordinate = x_Coordinate, y_coordinate = x_Coordinate, z_coordinate = z_Coordinate, authorization_token = authorization_Token ):
	global firstFrame,recognised_text,output_text
	
	if authorization_token not in recognised_text:
		recognised_text[authorization_token] = None
		output_text[authorization_token] = None

	if recognised_text[authorization_token] is None and output_text[authorization_token] is not None:
		output_text[authorization_token] = None

	if(len(firstFrame)>100):
		firstFrame = {}
		recognised_text={}

	boxes = []	
	# Decoding frame
	frame = decode_frame(encoded)
	# frame = cv2.cvtColor(frame,cv2.COLOR_YCrCb2RGB)
	# resizing frame
	#frame = imutils.resize(frame, width=600, inter=cv2.INTER_CUBIC)
	frame, rW, rH = resize_frame(frame)
	#cv2.imwrite("resized.jpg",frame)
	if(offline_detection == False):
		## Check for motion
		if (motion_detection(frame,min_area, authorization_token) == True or min_area==0):
			# print ("motion detected")
			
			(scores, geometry) = text_detection(frame)

			# decode the predictions, then  apply non-maxima suppression to
			# suppress weak, overlapping bounding boxes
			(rects, confidences) = decode_predictions(scores, geometry,frame,adjustment_factor_x,adjustment_factor_y,min_confidence)
			boxes = non_max_suppression(np.array(rects), probs=confidences)	
			if(np.size(boxes)>1):
				boxes = crop_save(frame,boxes, x_coordinate, y_coordinate, z_coordinate, authorization_token)
	else:
		threading.Thread(target=text_recognition_video, args=(frame, x_coordinate, y_coordinate, z_coordinate, authorization_token)).start()
	
	if recognised_text[authorization_token] is not None:
		output_text[authorization_token] = recognised_text[authorization_token]
		recognised_text[authorization_token] = None
	
	return resized_boxes(boxes,rW,rH), output_text[authorization_token]
