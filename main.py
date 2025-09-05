import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from pose_estimation import estimate_pose, feature_matching

def get_ground_truth_poses(file_path):
    """Loads ground truth poses from the KITTI dataset file."""
    # TODO: Read the ground truth poses file.
    # Each line in the file represents a pose as a flattened 3x4 transformation matrix.
    # Return a list of these 3x4 matrices.
    poses = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            T = np.fromstring(line, dtype=np.float64, sep=' ')
            T = T.reshape(3, 4)
            poses.append(T)
    return poses

def get_absolute_scale(ground_truth_poses, frame_id):
    """
    Calculates the absolute scale of motion from ground truth.
    This is a simplification for monocular VO. In a real system, you would get
    scale from other sensors (like stereo vision or IMU).
    """
    # TODO: Calculate the Euclidean distance between the translation vectors
    # of the current pose and the previous pose from the ground truth.
    # The translation vector is the last column of the 3x4 pose matrix.
    # This distance is our 'scale'.
    if frame_id < 1:
        return 1.0 # No motion for the first frame
    
    prev_pose = ground_truth_poses[frame_id - 1]
    curr_pose = ground_truth_poses[frame_id]
    
    dist = np.sqrt(np.sum((curr_pose[:, 3] - prev_pose[:, 3])**2))
    return dist


def main():
    # Setup
    # Set the paths
    image_dir = 'data/sequences/01/image_1/'
    pose_file = 'data/poses/01.txt'

    # Load camera intrinsics and ground truth
    K = np.array([[718.8560, 0, 607.1928],
                  [0, 718.8560, 185.2157],
                  [0, 0, 1]])
    ground_truth_poses = get_ground_truth_poses(pose_file)

    # --- 2. Initialization ---
    # TODO: Initialize your global pose variables.
    # The global rotation matrix should be a 3x3 identity matrix.
    # The global translation vector should be a 3x1 zero vector.
    R_global = np.eye(3)
    t_global = np.zeros((3, 1))

    # List to store the trajectory points for plotting
    trajectory = [t_global.flatten()]
    
    # Load the first image
    prev_image = cv.imread(f'{image_dir}/000000.png', cv.IMREAD_GRAYSCALE)

    # Main loop - process up to frame 4000
    num_frames = min(1100, len(ground_truth_poses))
    print(f"\nStarting visual odometry on {num_frames} frames...")
    print("Every '.' represents 10 frames processed:")
    
    for i in range(1, num_frames):
        # Load the current image
        curr_image = cv.imread(f'{image_dir}/{i:06d}.png', cv.IMREAD_GRAYSCALE)

        # 1. Match features between prev_image and curr_image
        kp1, kp2, good_matches = feature_matching(prev_image, curr_image)

        # 2. Estimate relative pose (R_rel, t_rel)
        R_rel, t_rel = estimate_pose(prev_image, curr_image, good_matches)

        # 3. Get the absolute scale from ground truth
        scale = get_absolute_scale(ground_truth_poses, i)

        # 4. Update the global pose
        if scale > 0.1: # Only update if there is significant motion
            t_global = t_global + scale * (R_global @ t_rel)
            R_global = R_global @ R_rel

        # 5. Store the new position
        trajectory.append(t_global.flatten())

        # 6. Update the previous image for the next iteration
        prev_image = curr_image
        
        # Print progress
        if i % 10 == 0:
            print('.', end='', flush=True)
        if i % 200 == 0:
            print(f' {i}', flush=True)

    # --- 4. Visualization ---
    # TODO: Plot your calculated trajectory and the ground truth trajectory.
    trajectory_points = np.array(trajectory)
    ground_truth_points = np.array([T[:3, 3] for T in ground_truth_poses])

    plt.figure(figsize=(10, 10))
    # We plot X (axis 0) vs Z (axis 2) for a top-down view
    plt.plot(trajectory_points[:, 0], trajectory_points[:, 2], label='Estimated Trajectory', color='red')
    plt.plot(ground_truth_points[:1100, 0], ground_truth_points[:1100, 2], label='Ground Truth', color='blue')
    
    plt.title('Monocular Visual Odometry')
    plt.xlabel('X position (m)')
    plt.ylabel('Z position (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Important for correct aspect ratio
    plt.show()

if __name__ == "__main__":
    main()
