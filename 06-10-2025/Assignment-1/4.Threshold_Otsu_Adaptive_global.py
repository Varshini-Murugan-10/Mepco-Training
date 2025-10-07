import cv2
import numpy as np
import matplotlib.pyplot as plt

def threshold_metal_surface(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Global thresholding (fixed value, e.g., 127)
    _, global_thresh = cv2.threshold(img, 53, 127, cv2.THRESH_BINARY)
    _, otsu_thresh = cv2.threshold(img, 0, 10, cv2.THRESH_OTSU)
    adaptive_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=11, C=10)
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(global_thresh, cmap='gray')
    plt.title('Global Thresholding')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(otsu_thresh, cmap='gray')
    plt.title("Otsu's Thresholding")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(adaptive_thresh, cmap='gray')
    plt.title('Adaptive Thresholding')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    return global_thresh, otsu_thresh, adaptive_thresh
global_mask, otsu_mask, adaptive_mask = threshold_metal_surface('D:/New Volume(X;)/HCL/opencv/MVdecAD/tile/test/crack/000.png')
cv2.imwrite('global_threshold_mask.png', global_mask)
cv2.imwrite('otsu_threshold_mask.png', otsu_mask)
cv2.imwrite('adaptive_threshold_mask.png', adaptive_mask)
