import base64
import cv2.cv2 as cv2
import text_detection_v8
import numpy as np
import imutils

frame = cv2.imread('new_book.jpg')

# resize the frame, maintaining the aspect ratio
frame = imutils.resize(frame, width=500)
orig = frame.copy()

_,encoded_image =cv2.imencode('.jpg',frame)
encoded_byte = (base64.b64encode(encoded_image))

boxes = text_detection_v8.imageProcessor(encoded_byte, 0.999, 0)
boxes = text_detection_v8.imageProcessor(encoded_byte, 0.98, 0)
# loop over the bounding boxes
if(np.shape(boxes)!=None):
    # print (boxes)
    for (startX, startY, endX, endY) in boxes:
        # draw the bounding box on the frame
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.imshow("Text Detection", frame)

# show the output frame
# cv2.imshow("Actual", frame)
key = cv2.waitKey(0)
