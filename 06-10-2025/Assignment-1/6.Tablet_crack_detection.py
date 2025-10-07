import cv2
import numpy as np

def detect_tablet_defects(image_path, min_area=100, max_area=1000):
    # Load and preprocess image
    img = cv2.imread(image_path)
    cv2.imshow('input_img', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 125, cv2.THRESH_BINARY + cv2.THRESH_OTSU )

    # Connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=4)

    defect_boxes = []
    output_img = img.copy()

    for i in range(1, num_labels):  
        x, y, w, h, area = stats[i]
        # Check area for defect (missing, broken, or extra)
        if area < min_area or area > max_area:
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            defect_boxes.append((x, y, w, h, area))

    return output_img, defect_boxes

result_img, defects = detect_tablet_defects('D:/New Volume(X;)/HCL/opencv/MVdecAD/capsule/test/crack/001.png', min_area=500, max_area=1000)
cv2.imwrite('tablet_defects_detected.png', result_img)
cv2.imshow('tablet_defects_detected.png', result_img)
