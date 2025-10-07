import cv2
import numpy as np
def detect_pcb_defects(image_path, edge_method="Canny", morph_kernel_size=3):
    gray = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    if edge_method == "Sobel":
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobelx, sobely)
        edges = cv2.convertScaleAbs(edges)
        _, edges = cv2.threshold(edges, 40, 255, cv2.THRESH_BINARY)
    elif edge_method == "Laplacian":
        edges = cv2.Laplacian(blurred, cv2.CV_64F)
        edges = cv2.convertScaleAbs(edges)
        _, edges = cv2.threshold(edges, 40, 255, cv2.THRESH_BINARY)
    else:  # Canny
        edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    defect_mask = opened
    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated_img = img.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 30:  # filter small noise
            x, y, w, h = cv2.boundingRect(cnt)
            # Annotate defects with bounding boxes and label
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(annotated_img, "Defect", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return defect_mask, annotated_img, gray
defect_mask, annotated_img, image = detect_pcb_defects('download.jpg', edge_method='Canny', morph_kernel_size=3)
cv2.imshow('pcb_image.png',image)
cv2.imshow('pcb_defect_mask.png', defect_mask)
cv2.imshow('pcb_defect_annotated.png', annotated_img)
cv2.imwrite('pcb_defect_mask.png', defect_mask)
cv2.imwrite('pcb_defect_annotated.png', annotated_img)

