import cv2
import numpy as np

img=cv2.imread('C:/Users/varsh/Downloads/fruit/images (3).jpg')
img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

mask=cv2.inRange(img_hsv,np.array((20,100,100)),np.array((35,255,255)))

res_img=cv2.bitwise_and(img,img,mask=mask)

cv2.imshow('Original Image',img)
cv2.imshow('HSV Image',img_hsv)
cv2.imshow('resultant Image',res_img)

cv2.waitKey()
cv2.destroyAllWindows()
