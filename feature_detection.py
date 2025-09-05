import cv2 as cv
import numpy as np
import os 

def feature_extraction(img):
    '''Extracts features from an image. 
    Returns keypoints and descriptors.'''
    if isinstance(img, str):
        # If a path is provided, load the image
        img = cv.imread(img, cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error loading image from path: {img}")
            return None, None

    # Feature detection
    # Using ORB
    orb = cv.ORB_create(nfeatures=1000)

    # Use the ORB object to detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(img, None)

    # print(f"Detected {len(keypoints)} keypoints in the image.")

    return keypoints, descriptors
