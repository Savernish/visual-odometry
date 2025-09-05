import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from feature_detection import feature_extraction

def lowe_ratio_test(matches, threshold):
    # Test to find good matches using Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < threshold * n.distance:  # If best match is significantly better than second best
            good_matches.append(m)
    return sorted(good_matches, key=lambda x: x.distance)

def feature_matching(img1, img2):
    # Get keypoints and descriptors for both images
    keyp1, des1 = feature_extraction(img1)
    keyp2, des2 = feature_extraction(img2)
    # Use brute force matcher with KNN for lowe's ratio test
    bf = cv.BFMatcher(cv.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = lowe_ratio_test(matches, 0.8)

    return keyp1, keyp2, good_matches

    # dpi = plt.rcParams['figure.dpi']
    # height, width = img3.shape[:2]
    # figsize = (width/dpi*1.5, height/dpi*1.5)
    
    # plt.figure(figsize=figsize)
    # plt.imshow(img3)
    # plt.axis('off')
    # plt.tight_layout(pad=0)
    # plt.show()

# feature_matching()
