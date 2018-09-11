import cv2

capture = cv2.VideoCapture(1)

while True:
    (ret, frame) = capture.read()
    
    
    if cv2.waitKey(1)== 13:
        break

