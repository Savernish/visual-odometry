import cv2 as cv
import numpy as np

def feature_extraction():
    '''Extracts features from two consecutive images. 
    Returns keypoints and descriptors and the images.'''
    # Data setup
    image_dir = 'data/sequences/00/image_0' 
    
    # Load two consecutive images
    img_path1 = f'{image_dir}/000000.png'
    img_path2 = f'{image_dir}/000001.png'
    
    # Load images in grayscale
    img1 = cv.imread(img_path1, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(img_path2, cv.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print(f"Error loading images. Check the path: {image_dir}")
        return

    # Feature detection
    # Using ORB
    orb = cv.ORB_create(nfeatures=1000)

    # Use the ORB object to detect keypoints and compute descriptors for both images.
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    print(f"Detected {len(keypoints1)} keypoints in the first image.")
    print(f"Detected {len(keypoints2)} keypoints in the second image.")

    # Visualization
    # Draw the keypoints on the images.
    img1_keypoints = cv.drawKeypoints(img1, keypoints1, None, color=(0, 255, 0), flags=0)
    img2_keypoints = cv.drawKeypoints(img2, keypoints2, None, color=(0, 255, 0), flags=0)

    # Combine the two images side-by-side for display
    combined_image = np.hstack((img1_keypoints, img2_keypoints))

    return img1, (keypoints1, descriptors1), img2, (keypoints2, descriptors2)
