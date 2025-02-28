import glm
import random
import numpy as np
from calibration import extract_frames, get_manual_corners, interpolate_with_homography, calibrate_camera, inner_objp, square_size
from background_subtraction import build_background_model_mog2, get_foreground_mask

block_size = 1.0

# --- Calibration Section ---
import os
from calibration import extract_frames, get_manual_corners, interpolate_with_homography, calibrate_camera, inner_objp, square_size

# Function that processes each camera folder’s intrinsics.avi using manual corner selection and interpolation,
# then performs calibration to compute intrinsics and extrinsics
def calibrate_cameras(data_path, camera_folders, num_frames=25):
    cam_params = {}
    for cam in camera_folders:
        video_path = os.path.join(data_path, cam, "intrinsics.avi")
        print(f"Processing calibration for {video_path}")
        frames = extract_frames(video_path, num_frames=num_frames)
        outer_corners_list = []
        saved_frames = []
        for frame in frames:
            corners_4 = get_manual_corners(frame)
            if corners_4 is not None:
                outer_corners_list.append(corners_4)
                saved_frames.append(frame)
            else:
                print("Skipping frame: not enough corners selected.")
        if not outer_corners_list:
            print(f"No valid frames for calibration in {cam}. Skipping.")
            continue
        object_points_all = []
        image_points_all = []
        for corners_4 in outer_corners_list:
            full_corners_2d = interpolate_with_homography(corners_4, (10, 8))
            inner_corners_2d = full_corners_2d[1:-1, 1:-1, :].reshape(-1, 2)
            object_points_all.append(inner_objp) 
            image_points_all.append(inner_corners_2d)
        h, w = saved_frames[0].shape[:2]
        img_size = (w, h)
        camMat, distCoeffs, err, rvecs, tvecs = calibrate_camera(object_points_all, image_points_all, img_size)
        print(f"[Camera: {cam}] Calibration complete. Mean Error: {err}")
        cam_params[cam] = {
            "cameraMatrix": camMat,
            "distCoeffs": distCoeffs,
            "rvec": rvecs[0],
            "tvec": tvecs[0]
        }
    return cam_params


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    data, colors = [], []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if random.randint(0, 1000) < 5:
                    data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
                    colors.append([x / width, z / depth, y / height])
    return data, colors


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    return [[-64 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, 63 * block_size],
            [63 * block_size, 64 * block_size, -64 * block_size],
            [-64 * block_size, 64 * block_size, -64 * block_size]], \
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations
