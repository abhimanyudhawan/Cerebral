# USAGE
# python image_client.py


import base64
import cv2.cv2 as cv2
import text_detection_v10
import numpy as np
import imutils

frame = cv2.imread('test_image.jpg')

(H, W) = (None, None)
(newH, newW) = (192,192)
(rH, rW) = (None, None)

# resize the frame, maintaining the aspect ratio
# frame = imutils.resize(frame, width=300)
orig = frame.copy()

if W is None or H is None:
	(H, W) = frame.shape[:2]
	rW = W / float(newW)
	rH = H / float(newH)

# resize the frame, this time ignoring aspect ratio
# frame = cv2.resize(frame, (newW, newH))

_,encoded_image =cv2.imencode('.jpg',frame)
encoded_byte = (base64.b64encode(encoded_image))
# f = open('new_text2.txt','wb')
# f.write(encoded_byte)
# f.close()

min_area = 0
adjustment_factor_x = 0.08
adjustment_factor_y = 0.5

boxes, output_text = text_detection_v10.imageProcessor(encoded_byte, 0.9,min_area, adjustment_factor_x, adjustment_factor_y, authorization_token='eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJkYXRhIjoiNWJkZjRhNmJiZDg3ZjMxY2U5MDdiMmMzX180NTk2NzciLCJpYXQiOjE1NDE4MjA2ODgsImV4cCI6MTU0NDQxMjY4OH0.TdOPISz-FEFHePwp-UetWnda6_6Xo2Iv6drJVp8rlz4')
if(np.shape(boxes)!=None):
	for (startX, startY, endX, endY) in boxes:
		# startX = int(startX * rW)
		# startY = int(startY * rH)
		# endX = int(endX * rW)
		# endY = int(endY * rH)
		# draw the bounding box on the frame
		# if output_text is not None:
		# 	print (output_text)
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
cv2.imshow("Text Detection", orig)
key = cv2.waitKey(0)