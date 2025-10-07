import cv2
import numpy as np

img = cv2.imread('D:/New Volume(X;)/HCL/opencv/MVdecAD/leather/test/cut/003.png', cv2.IMREAD_GRAYSCALE)

_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((15,12), np.uint8)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
defect_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

cv2.imshow('fabric', img)
cv2.imshow('fabric_defects_mask', defect_mask)
cv2.imwrite('fabric_defects_mask.png', defect_mask)
