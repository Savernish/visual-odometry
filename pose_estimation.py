import cv2 as cv
import numpy as np
from feature_matching import feature_matching

def estimate_pose(prev_image, curr_image, good_matches):
    # Get keypoints from the feature matching function
    kp1, kp2, _ = feature_matching(prev_image, curr_image)

    # Camera Intrinsics (Provided by KITTI dataset for sequence 00)
    # Must use these values for the geometry to be correct
    focal_length = 718.8560
    principal_point = (607.1928, 185.2157)
    K = np.array([[focal_length, 0, principal_point[0]],
                  [0, focal_length, principal_point[1]],
                  [0, 0, 1]])

    # Exctract point coordinates
    # TODO: Get the coordinates of the good matches from both images.
    # The keypoints `kp1` and `kp2` are lists of KeyPoint objects.
    # You need to extract the `pt` attribute (which is a (x,y) tuple) for each match.
    # The result should be two NumPy arrays, `points1` and `points2`.
    points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    # Calculate the Essential Matrix
    # TODO: Use cv.findEssentialMat to calculate the Essential Matrix.
    # Required inputs: points2, points1, and the camera matrix K.
    # Use the RANSAC method (cv.RANSAC) for robustness.
    E, mask = cv.findEssentialMat(points2, points1, K, method=cv.RANSAC, prob=0.999, threshold=1.0)

    # TODO: Use cv.recoverPose to get the rotation (R) and translation (t).
    # Required inputs: The Essential Matrix E, points2, points1, and K.
    # The 'mask' from findEssentialMat can be passed to filter outliers.
    _, R, t, mask = cv.recoverPose(E, points2, points1, K)

    return R, t

if __name__ == "__main__":
    estimate_pose()