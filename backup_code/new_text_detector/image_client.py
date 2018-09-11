import base64
import cv2.cv2 as cv2
import text_detection_video_test

frame = cv2.imread('new_book.jpg')
_,encoded_image =cv2.imencode('.jpg',frame)
f = open('new_text2.txt','wb')
encoded_byte = (base64.b64encode(encoded_image))
f.write(encoded_byte)
f.close()

boxes = text_detection_video_test.imageProcessor(encoded_byte, 0.7, 200, 0.8, 0.02)

print (boxes)