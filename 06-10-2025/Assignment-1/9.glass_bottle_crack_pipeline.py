import cv2
import numpy as np

def detect_cracks_or_missing_parts(image_path, edge_method="Canny", morph_kernel_size=5):
    img = cv2.imread(image_path)
    cv2.imshow("Original Image",img)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    if edge_method == "Sobel":
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobelx, sobely)
        edges = cv2.convertScaleAbs(edges)
        _, edges = cv2.threshold(edges, 20, 255, cv2.THRESH_BINARY)
    elif edge_method == "Laplacian":
        edges = cv2.Laplacian(blurred, cv2.CV_64F)
        edges = cv2.convertScaleAbs(edges)
        _, edges = cv2.threshold(edges, 20, 255, cv2.THRESH_BINARY)
    else:  
        edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    defect_mask = opened

    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    localized_img = img.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:  # filter tiny noise
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(localized_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.drawContours(localized_img, [cnt], -1, (0, 255, 0), 1)

    return defect_mask, localized_img

defect_mask, localized_img = detect_cracks_or_missing_parts('D:/New Volume(X;)/HCL/opencv/MVdecAD/bottle/test/broken_large/014.png', edge_method="Canny", morph_kernel_size=5)
cv2.imshow('glass_bottle_defect_mask.png', defect_mask)
cv2.imshow('glass_bottle_defect_localization.png', localized_img)
