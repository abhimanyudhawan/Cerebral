import cv2.cv2 as cv2
import numpy as np
import os
def apply_threshold(img, argument):
    switcher = {
        1: cv2.threshold(cv2.GaussianBlur(img, (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        2: cv2.threshold(cv2.GaussianBlur(img, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        3: cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        4: cv2.threshold(cv2.medianBlur(img, 5), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        5: cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        6: cv2.adaptiveThreshold(cv2.GaussianBlur(img, (1, 1), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 17),
        7: cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
        8: cv2.adaptiveThreshold(cv2.GaussianBlur(img, (1, 1), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15),
    }
    return switcher.get(argument, "Invalid method")

def split_images(img):
    b,g,r = cv2.split(img)
    # cv2.imshow("Blue" ,b)
    # cv2.imshow("Green" ,g)
    # cv2.imshow("Red" ,r)
    # my_img = np.zeros((b.shape[0], b.shape[1], 3), 
    #         dtype = b.dtype) 
    my_img = cv2.merge((b, g, r-195))
    cv2.imwrite(os.path.join("./", "colored" +".jpg"), my_img)
    return my_img

    # for values, color, channel in zip((r, g, b), 
    #         ('red', 'green', 'blue'), (2,1,0)):
        
    #     my_img[:,:,channel] = values
    #     cv2.imwrite(os.path.join("./", color +".jpg"), my_img)
        # cv2.imwrite("blue.jpg",myimg)
        # cv2.imwrite("green.jpg",g)
        # cv2.imwrite("red.jpg",r)




for i in range (2,3):
    actual_img = cv2.imread(str(i)+".jpeg")
    # cv2.imshow("input"+str(i),actual_img)
    # actual_img=cv2.resize(actual_img,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(actual_img,cv2.COLOR_BGR2GRAY)
    print(np.size(img,1))
    # upper_white = np.array([255,255,255], dtype = "uint16")
    # lower_white = np.array([120,120,120], dtype = "uint16")
    # white_mask = cv2.inRange(img, lower_white, upper_white)
    # # _,white_mask = cv2.invert(black_mask)
    # cv2.imshow('mask0',white_mask)
    # img = white_mask
    # img = split_images(img)
    # 
    # cv2.imshow("output"+str(i),img)
    # cv2.imwrite("final.jpg", img)

# Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    # img = cv2.GaussianBlur(img, (33,33), 0)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    # new_image = cv2.threshold(img,100,255,cv2.THRESH_BINARY)[1]
    # new_image = cv2.add(new_image,new_image)
    # __,contours, hier = cv2.findContours(new_image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    # for i in range(0,len(contours)):
    #     if(cv2.contourArea(contours[i])>5000):
    #         # print("found"+str(i))
    #         cv2.drawContours(actual_img,contours,i,(0,0,0))
            
    # cv2.imwrite("i_am_done.jpg",actual_img)
    # cv2.imshow("output" + str(i) ,actual_img)
# # # #  Apply threshold to get image with only black and white
    for j in range(7,9):
        # new_image = img
        new_image = apply_threshold(img, j)
        # new_image = cv2.dilate(new_image, kernel, iterations=1)
        # new_image = cv2.erode(new_image, kernel, iterations=1)
        # added1 = cv2.add(new_image,new_image)
        cv2.imshow("output"+str(i) + "_" + str(j),new_image)
        cv2.imwrite("final" + str(i) + "_" + str(j) + ".jpg", new_image)


# save_path = os.path.join(output_path, file_name + "_filter_" + str(method) + ".jpg")
# cv2.imwrite(save_path, img)


cv2.waitKey(0)
cv2.destroyAllWindows()