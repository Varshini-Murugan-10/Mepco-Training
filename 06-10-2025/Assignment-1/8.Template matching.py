import cv2
import numpy as np
def sift_orb_feature_matching(product_img_path, template_img_path, method="SIFT"):
    # Load images
    product_img = cv2.imread(product_img_path, cv2.IMREAD_GRAYSCALE)
    template_img = cv2.imread(template_img_path, cv2.IMREAD_GRAYSCALE)
    if method.upper() == "SIFT":
        detector = cv2.SIFT_create()
    elif method.upper() == "ORB":
        detector = cv2.ORB_create()
    else:
        raise ValueError("Method must be 'SIFT' or 'ORB'")

    kp1, des1 = detector.detectAndCompute(template_img, None)
    kp2, des2 = detector.detectAndCompute(product_img, None)

    if method.upper() == "SIFT":
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append([m])
        matched_img = cv2.drawMatchesKnn(template_img, kp1, product_img, kp2, good_matches, None,
                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        num_good_matches = len(good_matches)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:20]  # top 20 matches
        matched_img = cv2.drawMatches(template_img, kp1, product_img, kp2, good_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        num_good_matches = len(good_matches)

    template_match_img = product_img.copy()
    res = cv2.matchTemplate(product_img, template_img, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    template_match_val = max_val
    h, w = template_img.shape
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(template_match_img, top_left, bottom_right, (255, 0, 0), 2)
    return matched_img, template_match_img, num_good_matches, template_match_val

matched, template_matched, good_count, tm_val = sift_orb_feature_matching(
     "product.jpg", "logo_template.jpg", method="SIFT"
 )
img=cv2.imread("product.jpg")
cv2.imshow("Original Image",img)
cv2.imshow('Matched Image',matched)
cv2.imshow('Template Match image',template_matched)
print(f"Number of good feature matches: {good_count}")
print(f"Template matching max correlation: {tm_val:.3f}")

