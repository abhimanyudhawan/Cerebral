import base64
import cv2.cv2 as cv2
import text_detection_v8
import numpy as np
import imutils

frame = cv2.imread('initial3.jpeg')

# resize the frame, maintaining the aspect ratio
frame = imutils.resize(frame, width=300)
orig = frame.copy()

_,encoded_image =cv2.imencode('.jpg',frame)
encoded_byte = (base64.b64encode(encoded_image))
# f = open('new_text2.txt','wb')
# f.write(encoded_byte)
# f.close()

boxes = text_detection_v8.imageProcessor(encoded_byte, 0.999, 0,adjustment_factor_y=0.1)
print (boxes)
for i in range (0,len(boxes)):
    # print(boxes[i][1])

    # if(np.shape(boxes[i][1])!=None):
    #     for (startX, startY, endX, endY) in boxes[i][1]:
    startX = int(boxes[i][1][0][0])
    startY = int(boxes[i][1][0][1])
    endX = int(boxes[i][1][0][2])
    endY = int(boxes[i][1][0][3])
#         # draw the bounding box on the frame
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

cv2.imshow("Text Detection", orig)

# for i in range(0, len(boxes)):
#     if (i % 2 == 0):
#        cnt = boxes[i]
#        #mask = np.zeros(im2.shape,np.uint8)
#        #cv2.drawContours(mask,[cnt],0,255,-1)
#        x,y,w,h = cv2.boundingRect(cnt)
#        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
#        cv2.imshow('Features', frame)
#        cv2.imwrite(str(i)+'.png', frame)

# boxes = text_detection_v8.imageProcessor(encoded_byte, 0.98, 0)
# loop over the bounding boxes
# if(np.shape(boxes)!=None):
#     # print (boxes)
#     for (startX, startY, endX, endY) in boxes:
#         new_frame = frame[startY:endY, startX:endX]
#         new_frame = cv2.resize(new_frame,None,1,1,cv2.INTER_CUBIC)
#         gray = cv2.cvtColor(new_frame,cv2.COLOR_BGR2GRAY)
#         gray = cv2.bitwise_not(gray)

#         thresh = cv2.threshold(gray, 0, 255,
# 	            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#         coords = np.column_stack(np.where(thresh > 0))
#         angle = cv2.minAreaRect(coords)[-1]
        
#         # the `cv2.minAreaRect` function returns values in the
#         # range [-90, 0); as the rectangle rotates clockwise the
#         # returned angle trends to 0 -- in this special case we
#         # need to add 90 degrees to the angle
#         if angle < -45:
#             angle = -(90 + angle)
        
#         # otherwise, just take the inverse of the angle to make
#         # it positive
#         else:
#             angle = -angle

#         print(angle)

#         (h, w) = new_frame.shape[:2]
#         center = (w // 2, h // 2)
#         M = cv2.getRotationMatrix2D(center, angle, 1.0)
#         rotated = cv2.warpAffine(new_frame, M, (w, h),
#             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

#         cv2.imshow("Before", new_frame)
#         cv2.imshow("After", rotated)

# show the output frame
# cv2.imshow("Actual", frame)
key = cv2.waitKey(0)
