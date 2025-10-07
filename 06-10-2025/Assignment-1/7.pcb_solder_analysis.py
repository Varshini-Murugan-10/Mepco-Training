import cv2
import numpy as np

def analyze_solder_joints(image_path, min_area=50, max_area=500):
    img = cv2.imread(image_path)
    cv2.imshow("Original Image",img)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    good_count = 0
    defective_count = 0
    output_img = img.copy()
    for i in range(1, num_labels):  
        x, y, w, h, area = stats[i]
        if min_area <= area <= max_area:
            good_count += 1
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0,255,0), 2)
        else:
            defective_count += 1
            cv2.rectangle(output_img, (x, y), (x+w, y+h), (0,0,255), 2)
    
    stats_dict = {
        "total_joints": num_labels - 1,
        "good_joints": good_count,
        "defective_joints": defective_count,
        "defective_percentage": 100.0 * defective_count / (num_labels - 1) if num_labels > 1 else 0,
        "good_percentage": 100.0 * good_count / (num_labels - 1) if num_labels > 1 else 0,
    }
    
    return stats_dict, output_img

stats, result_img = analyze_solder_joints('download.jpg', min_area=80, max_area=400)
cv2.imshow('pcb_solder_analysis.png', result_img)
print("Solder joint statistics:", stats)
