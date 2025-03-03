import glm
import random
import numpy as np
import cv2
import os

from calibration import extract_frames, get_manual_corners, interpolate_with_homography, determine_grid_size, calibrate_camera, inner_objp, reorder_corners, square_size
from background_subtraction import build_background_model_mog2, get_foreground_mask
from scipy.ndimage import label
from scipy.spatial import cKDTree

block_size = 1.0

# --- Calibration Section ---
import os
from calibration import extract_frames, get_manual_corners, interpolate_with_homography, calibrate_camera, inner_objp, square_size

# Function that processes each camera folder's intrinsics.avi using manual corner selection and interpolation,
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
            grid_size = determine_grid_size(corners_4, horizontal_grid_size=10, vertical_grid_size=8)
            full_corners_2d = interpolate_with_homography(corners_4, grid_size)
            inner_corners_2d = full_corners_2d[1:-1, 1:-1, :].reshape(-1, 2)
            object_points_all.append(inner_objp) 
            image_points_all.append(inner_corners_2d)
        h, w = saved_frames[0].shape[:2]
        img_size = (w, h)
        camMat, distCoeffs, err, rvecs, tvecs = calibrate_camera(object_points_all, image_points_all, img_size)
        print(f"[Camera: {cam}] Calibration complete. Mean Error: {err}")
        cam_params[cam] = {
            "cameraMatrix": camMat,
            "distCoeffs": distCoeffs
        }
    return cam_params

# Function that calculates the extrinsics for a camera based on the chessboard.avi video frame
def calibrate_extrinsics_for_camera(cam_folder, intrinsics, calib_video="checkerboard.avi"):
    cameraMatrix = intrinsics["cameraMatrix"]
    distCoeffs = intrinsics["distCoeffs"]
    
    video_path = os.path.join("data", cam_folder, calib_video)
    cap = cv2.VideoCapture(video_path)
    ret, test_img = cap.read()
    cap.release()
    if not ret or test_img is None:
        print(f"Error: Could not load a frame from {video_path}")
        return None
    
    corners = get_manual_corners(test_img)
    if not corners:
        print(f"No corners selected for {cam_folder}; extrinsics calibration aborted.")
        return None
    ordered_corners = reorder_corners(corners)
    grid_size = (10, 8)
    full_grid = interpolate_with_homography(ordered_corners, grid_size)
    inner_grid = full_grid[1:-1, 1:-1, :].reshape(-1, 2)
    image_points = inner_grid.reshape(-1, 1, 2).astype(np.float32)
    
    success, rvec, tvec = cv2.solvePnP(inner_objp, image_points, cameraMatrix, distCoeffs)
    if not success:
        print(f"solvePnP failed for {cam_folder}")
        return None
    
    projected, _ = cv2.projectPoints(inner_objp, rvec, tvec, cameraMatrix, distCoeffs)
    error = np.mean(np.linalg.norm(image_points.reshape(-1,2) - projected.reshape(-1,2), axis=1))
    print(f"[{cam_folder}] Extrinsics reprojection error: {error:.4f}")
    print(f"[{cam_folder}] rvec:\n{rvec}\n tvec:\n{tvec}")

    # Visualize by projecting a 3D axis.
    axis_3D = np.float32([
        [3 * square_size, 0, 0],
        [0, 3 * square_size, 0],
        [0, 0, -3 * square_size]
    ])
    imgpts, _ = cv2.projectPoints(axis_3D, rvec, tvec, cameraMatrix, distCoeffs)
    corner_origin = tuple(map(int, inner_grid[0].ravel()))
    x_axis = tuple(map(int, imgpts[0].ravel()))
    y_axis = tuple(map(int, imgpts[1].ravel()))
    z_axis = tuple(map(int, imgpts[2].ravel()))
    overlay_img = test_img.copy()
    cv2.line(overlay_img, corner_origin, x_axis, (255, 0, 0), 2)
    cv2.line(overlay_img, corner_origin, y_axis, (0, 255, 0), 2)
    cv2.line(overlay_img, corner_origin, z_axis, (0, 0, 255), 2)
    cv2.namedWindow("3D Axes Overlay", cv2.WINDOW_NORMAL)
    cv2.imshow("3D Axes Overlay", overlay_img)
    cv2.waitKey(0)
    cv2.destroyWindow("3D Axes Overlay")
    
    return {"rvec": rvec, "tvec": tvec}

# Function that saves intrinsics and extrinsics for all cameras
def calibrate_all_cameras(data_path, camera_folders, intrinsics_results, calib_video="checkerboard.avi"):
    full_calib = {}
    for cam in camera_folders:
        if cam not in intrinsics_results:
            print(f"Skipping {cam} because intrinsics were not computed.")
            continue

        print(f"\n--- Calibrating extrinsics for {cam} ---")
        extrinsics = calibrate_extrinsics_for_camera(cam, intrinsics_results[cam], calib_video)
        if extrinsics is None:
            print(f"Extrinsics calibration for {cam} failed.")
            continue

        # Merge intrinsics and extrinsics for this camera
        full_calib[cam] = {
            "cameraMatrix": intrinsics_results[cam]["cameraMatrix"],
            "distCoeffs": intrinsics_results[cam]["distCoeffs"],
            "rvec": extrinsics["rvec"],
            "tvec": extrinsics["tvec"]
        }

        # Create the XML string using your helper function.
        xml_string = create_config_xml(
            full_calib[cam]["cameraMatrix"],
            full_calib[cam]["distCoeffs"],
            full_calib[cam]["rvec"],
            full_calib[cam]["tvec"]
        )
        # Save the XML file into the camera's folder
        out_path = os.path.join(data_path, cam, "config.xml")
        with open(out_path, "w") as f:
            f.write(xml_string)
        print(f"Calibration data for {cam} saved to {out_path}")

    return full_calib

# Function that creates xml files for the extrinsic and intrinsic values
def create_config_xml(cameraMatrix, distCoeffs, rvec, tvec):
    """
    Creates an XML string in OpenCV format that stores the intrinsic and extrinsic parameters.
    """
    cm = cameraMatrix.flatten()
    dc = distCoeffs.flatten()
    rv = rvec.flatten()
    tv = tvec.flatten()
    xml = f"""<?xml version="1.0"?>
    <opencv_storage>
    <CameraMatrix type_id="opencv-matrix">
        <rows>3</rows>
        <cols>3</cols>
        <dt>f</dt>
        <data>
        {" ".join([f"{x:.8f}" for x in cm])}
        </data>
    </CameraMatrix>
    <DistortionCoeffs type_id="opencv-matrix">
        <rows>5</rows>
        <cols>1</cols>
        <dt>f</dt>
        <data>
        {" ".join([f"{x:.8f}" for x in dc])}
        </data>
    </DistortionCoeffs>
    <RotationVector type_id="opencv-matrix">
        <rows>3</rows>
        <cols>1</cols>
        <dt>f</dt>
        <data>
        {" ".join([f"{x:.8f}" for x in rv])}
        </data>
    </RotationVector>
    <TranslationVector type_id="opencv-matrix">
        <rows>3</rows>
        <cols>1</cols>
        <dt>f</dt>
        <data>
        {" ".join([f"{x:.8f}" for x in tv])}
        </data>
    </TranslationVector>
    </opencv_storage>
    """
    return xml

# Function that determined whether calibration is needed
def run_calibration_interface(data_path, camera_folders, num_frames=25, calib_video="checkerboard.avi"):
    full_calib = {}
    missing = []
    for cam in camera_folders:
        config_path = os.path.join(data_path, cam, "config.xml")
        if not os.path.exists(config_path):
            print(f"[{cam}] Config file not found. Calibration needed.")
            missing.append(cam)
        else:
            print(f"[{cam}] Config file found. Loading calibration data...")
            fs = cv2.FileStorage(config_path, cv2.FILE_STORAGE_READ)
            if fs.isOpened():
                cameraMatrix = fs.getNode("CameraMatrix").mat()
                distCoeffs = fs.getNode("DistortionCoeffs").mat()
                rvec = fs.getNode("RotationVector").mat()
                tvec = fs.getNode("TranslationVector").mat()
                fs.release()
                full_calib[cam] = {"cameraMatrix": cameraMatrix,
                                   "distCoeffs": distCoeffs,
                                   "rvec": rvec,
                                   "tvec": tvec}
            else:
                print(f"[{cam}] Failed to open {config_path}. Calibration needed.")
                missing.append(cam)
    if missing:
        print("Running calibration for missing cameras:", missing)
        intrinsics_results = calibrate_cameras(data_path, missing, num_frames=num_frames)
        extrinsics_results = calibrate_all_cameras(data_path, missing, intrinsics_results, calib_video=calib_video)
        for cam in missing:
            if cam in extrinsics_results:
                full_calib[cam] = extrinsics_results[cam]
                xml_string = create_config_xml(full_calib[cam]["cameraMatrix"],
                                               full_calib[cam]["distCoeffs"],
                                               full_calib[cam]["rvec"],
                                               full_calib[cam]["tvec"])
                out_path = os.path.join(data_path, cam, "config.xml")
                with open(out_path, "w") as f:
                    f.write(xml_string)
                print(f"[{cam}] Calibration data saved to {out_path}")
            else:
                print(f"[{cam}] Calibration failed. No config saved.")
    return full_calib

def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [1, 1, 1])
    return data, colors

def remove_small_clusters(active_voxels_mm, roi_origin, voxel_size, roi_extent, min_voxel_count=50):
    """
    Convert the list of active voxel positions (in mm) to a 3D occupancy grid,
    perform connected component labeling, and remove clusters smaller than min_voxel_count.
    Return the cleaned voxel positions in mm.
    """
    # Define grid shape based on the ROI
    xs = np.arange(roi_origin[0], roi_origin[0] + roi_extent[0], voxel_size, dtype=np.float32)
    ys = np.arange(roi_origin[1], roi_origin[1] + roi_extent[1], voxel_size, dtype=np.float32)
    zs = np.arange(roi_origin[2], roi_origin[2] + roi_extent[2], voxel_size, dtype=np.float32)
    grid_shape = (len(xs), len(ys), len(zs))
    
    occupancy = np.zeros(grid_shape, dtype=np.uint8)
    
    # For each active voxel, compute its index in the occupancy grid.
    # Since the voxel positions come from the grid, the indices can be computed by:
    # index = round((position - roi_origin) / voxel_size)
    indices = np.rint((active_voxels_mm - roi_origin) / voxel_size).astype(int)
    for idx in indices:
        i, j, k = idx
        if 0 <= i < grid_shape[0] and 0 <= j < grid_shape[1] and 0 <= k < grid_shape[2]:
            occupancy[i, j, k] = 1

    # Label connected components in the 3D occupancy grid
    labeled, num_features = label(occupancy)
    # Build a new occupancy grid that keeps only large clusters.
    cleaned = np.zeros_like(occupancy)
    for comp in range(1, num_features + 1):
        comp_size = np.sum(labeled == comp)
        if comp_size >= min_voxel_count:
            cleaned[labeled == comp] = 1

    # Convert the cleaned occupancy grid back to voxel positions (in mm)
    kept_indices = np.argwhere(cleaned == 1)
    # Each index corresponds to: position = roi_origin + index * voxel_size
    cleaned_voxels = kept_indices.astype(np.float32) * voxel_size + roi_origin
    return cleaned_voxels

def set_voxel_positions(dummy_width, dummy_height, dummy_depth):
    """
    Silhouette-based voxel carving in mm with multi-camera voting, noise removal,
    and coloring of the voxel model with occlusion reasoning.
    
    Additional improvements:
      - A step size (subsampling) is applied to the final set of active voxels,
        to create a sparser (point cloud) effect.
      - Voxels with no color samples are now given a fallback color (white)
        instead of black.
    
    Steps:
      1) Define an ROI in mm around the person.
      2) Generate a voxel grid in mm.
      3) For each camera (cam1, cam2, cam3, cam4):
             - Load calibration data.
             - Build background model and grab one frame.
             - Compute the foreground mask.
             - Project voxel centers onto the 2D image and increment a vote if the projection falls on foreground.
      4) Select voxels with votes >= threshold.
      5) Run a 3D connected component analysis to remove small noisy clusters.
      6) (Optional) Subsample the active voxels using a step size.
      7) For each active voxel, determine its color via occlusion reasoning (using a per-camera z-buffer),
         and average colors from all cameras where the voxel is visible.
         If a voxel is not visible in any camera, assign it a fallback color (white).
      8) Convert the active voxel coordinates from mm to viewer coordinates:
             - Multiply by 0.01 (mm → cm)
             - Flip Y and Z axes as in get_cam_positions.
             - Add a manual vertical offset if needed.
      9) Return the transformed positions and colors.
    """
    # --- ROI in mm (adjust to your scene) ---
    roi_origin = np.array([-400, -800, -1400], dtype=np.float32)
    roi_extent = np.array([1000, 1400, 2000], dtype=np.float32)
    voxel_size = 10.0  # mm per voxel (adjust for resolution vs. performance)
    
    # Build the voxel grid in mm
    xs = np.arange(roi_origin[0], roi_origin[0] + roi_extent[0], voxel_size, dtype=np.float32)
    ys = np.arange(roi_origin[1], roi_origin[1] + roi_extent[1], voxel_size, dtype=np.float32)
    zs = np.arange(roi_origin[2], roi_origin[2] + roi_extent[2], voxel_size, dtype=np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    voxels_mm = np.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=-1)  # shape: (N, 3)
    N = voxels_mm.shape[0]
    print(f"[VoxelCarving] ROI-based grid in mm: {N} voxels")
    
    # Initialize vote counter for each voxel
    votes = np.zeros((N,), dtype=int)
    data_path = "data"
    camera_folders = ["cam1", "cam2", "cam3", "cam4"]
    
    # Loop over all cameras for multi-camera voting
    for cam in camera_folders:
        config_path = os.path.join(data_path, cam, "config.xml")
        fs = cv2.FileStorage(config_path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            print(f"Could not open {config_path}")
            continue
        cameraMatrix = fs.getNode("CameraMatrix").mat()
        distCoeffs = fs.getNode("DistortionCoeffs").mat()
        rvec = fs.getNode("RotationVector").mat()
        tvec = fs.getNode("TranslationVector").mat()
        fs.release()
        
        bg_video_path = os.path.join(data_path, cam, "background.avi")
        bg_model = build_background_model_mog2(bg_video_path, num_frames=30)
        if bg_model is None:
            print(f"[VoxelCarving] Could not build BG model for {cam}")
            continue
        
        video_path = os.path.join(data_path, cam, "video.avi")
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"[VoxelCarving] Could not read frame from {video_path}")
            continue
        
        fg_mask = get_foreground_mask(frame, bg_model, fixed_thresh=(30,30,30), min_blob_area=500)
        h_img, w_img = fg_mask.shape
        
        projected, _ = cv2.projectPoints(voxels_mm.astype(np.float32), rvec, tvec, cameraMatrix, distCoeffs)
        projected = projected.reshape(-1, 2)
        proj_int = np.rint(projected).astype(int)
        
        valid = (proj_int[:,0] >= 0) & (proj_int[:,0] < w_img) & (proj_int[:,1] >= 0) & (proj_int[:,1] < h_img)
        valid_indices = np.where(valid)[0]
        for idx in valid_indices:
            x_img, y_img = proj_int[idx]
            if fg_mask[y_img, x_img] == 255:
                votes[idx] += 1
        print(f"[VoxelCarving] Processed {cam}")
    
    vote_threshold = 4
    active_mask = votes >= vote_threshold
    active_voxels_mm = voxels_mm[active_mask]
    print(f"[VoxelCarving] Active voxels after voting: {active_voxels_mm.shape[0]} of {N}")
    
    # --- Remove small noisy clusters using connected component analysis ---
    active_voxels_mm = remove_small_clusters(active_voxels_mm, roi_origin, voxel_size, roi_extent, min_voxel_count=50)
    print(f"[VoxelCarving] Active voxels after noise removal: {active_voxels_mm.shape[0]}")
    
    # --- Optional: Subsample (step size) the active voxels for a sparser point cloud ---
    step_size = 1  # Use every 2nd voxel; increase to make it sparser.
    active_voxels_mm = active_voxels_mm[::step_size]
    print(f"[VoxelCarving] Active voxels after subsampling: {active_voxels_mm.shape[0]}")
    
    
    # --- Coloring: Determine Voxel Colors with Occlusion Reasoning ---
    voxel_colors = np.zeros((active_voxels_mm.shape[0], 3), dtype=np.float32)
    visibility_counts = np.zeros((active_voxels_mm.shape[0],), dtype=np.int32)

    for cam in camera_folders:
        config_path = os.path.join(data_path, cam, "config.xml")
        fs = cv2.FileStorage(config_path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            print(f"Could not open {config_path}")
            continue
        cameraMatrix = fs.getNode("CameraMatrix").mat()
        distCoeffs = fs.getNode("DistortionCoeffs").mat()
        rvec = fs.getNode("RotationVector").mat()
        tvec = fs.getNode("TranslationVector").mat()
        fs.release()
        
        video_path = os.path.join(data_path, cam, "video.avi")
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Could not read frame from {video_path}")
            continue
        frame = frame.astype(np.float32)
        h_img, w_img, _ = frame.shape
        
        projected, _ = cv2.projectPoints(active_voxels_mm.astype(np.float32), rvec, tvec, cameraMatrix, distCoeffs)
        projected = projected.reshape(-1,2)
        proj_int = np.rint(projected).astype(int)
        
        R, _ = cv2.Rodrigues(rvec)
        X_cam = (R @ active_voxels_mm.T + tvec).T  # shape: (num_active, 3)
        depths = X_cam[:, 2]  # Depth values in mm
        
        # Build a simple depth map (z-buffer) for this camera:
        depth_map = np.full((h_img, w_img), np.inf, dtype=np.float32)
        for i, (p, d) in enumerate(zip(proj_int, depths)):
            x, y = p
            if 0 <= x < w_img and 0 <= y < h_img:
                if d < depth_map[y, x]:
                    depth_map[y, x] = d
        
        tolerance = 5.0  # mm tolerance for occlusion
        for i, (p, d) in enumerate(zip(proj_int, depths)):
            x, y = p
            if 0 <= x < w_img and 0 <= y < h_img:
                if abs(d - depth_map[y, x]) < tolerance:
                    color = frame[y, x, :]  # BGR
                    voxel_colors[i] += color
                    visibility_counts[i] += 1
        print(f"[Coloring] Processed {cam}")

    # For voxels that did not receive any color sample, fill in using neighboring voxels.
    colored_idx = np.where(visibility_counts > 0)[0]
    if colored_idx.size > 0:
        colored_positions = active_voxels_mm[colored_idx]
        colored_colors = voxel_colors[colored_idx]
        tree = cKDTree(colored_positions)
        
        uncolored_idx = np.where(visibility_counts == 0)[0]
        for i in uncolored_idx:
            pos = active_voxels_mm[i]
            # Try to find neighbors within a radius (e.g., 2*voxel_size)
            neighbors = tree.query_ball_point(pos, r=2 * voxel_size)
            if len(neighbors) > 0:
                avg_color = np.mean(colored_colors[neighbors], axis=0)
                voxel_colors[i] = avg_color
                visibility_counts[i] = 1
            else:
                # If no neighbor within the radius, use the nearest neighbor.
                distance, neighbor_index = tree.query(pos)
                voxel_colors[i] = colored_colors[neighbor_index]
                visibility_counts[i] = 1
    else:
        voxel_colors[:] = 255

    # Average the colors for each voxel (for those with direct samples, this divides by the count)
    for i in range(voxel_colors.shape[0]):
        if visibility_counts[i] > 0:
            voxel_colors[i] /= visibility_counts[i]
        else:
            voxel_colors[i] = [255, 255, 255]  # Fallback to white

    # Normalize colors to [0,1]
    voxel_colors /= 255.0


    
    # --- Convert active voxel positions to viewer coordinates ---
    positions_viewer = []
    scale_factor = 0.01  # mm -> cm
    viewer_y_offset = 2  # cm offset
    for (Xmm, Ymm, Zmm) in active_voxels_mm:
        Xv = Xmm * scale_factor
        Yv = -Zmm * scale_factor + viewer_y_offset
        Zv = Ymm * scale_factor
        positions_viewer.append([Xv, Yv, Zv])
    
    colors = voxel_colors.tolist()
    return positions_viewer, colors






def get_cam_positions():
    # Returns the camera positions in world coordinates based on calibration data.
    data_path = "data"
    camera_folders = ["cam1", "cam2", "cam3", "cam4"]
    cam_positions = []
    
    cam_colors = [
        [1.0, 0.0, 0.0],  
        [0.0, 1.0, 0.0],  
        [0.0, 0.0, 1.0],  
        [1.0, 1.0, 0.0]   
    ]
    
    for i, cam in enumerate(camera_folders):
        try:
            fs = cv2.FileStorage(os.path.join(data_path, cam, "config.xml"), cv2.FILE_STORAGE_READ)
            if not fs.isOpened():
                raise Exception(f"Failed to open config.xml for {cam}")
            rvec = fs.getNode("RotationVector").mat()
            tvec = fs.getNode("TranslationVector").mat()
            fs.release()
                
            R, _ = cv2.Rodrigues(rvec)
            
            # Calculate camera position in world coordinates
            camera_position = -np.matmul(R.T, tvec).flatten()
            
            # Scale down the positions to fit in the visualization space
            # Divide by 100 to convert from centimeters to meters or to scale appropriately
            scale_factor = 0.01  # 1/100
            
            # OpenCV to OpenGL coordinate conversion:
            # OpenCV: Y down, Z forward
            # OpenGL: Y up, Z backward
            # Need to flip Y and Z axes
            scaled_position = [
                camera_position[0] * scale_factor,
                -camera_position[2] * scale_factor,  # Flip Y axis
                camera_position[1] * scale_factor   # Flip Z axis
            ]
            
            cam_positions.append(scaled_position)
            print(f"Loaded camera position for {cam}: {scaled_position}")
                
        except Exception as e:
            print(f"Error loading camera position for {cam}: {e}")
            # Use default position if loading fails
            default_positions = [
                [-64, 64, 63],
                [63, 64, 63],
                [63, 64, -64],
                [-64, 64, -64]
            ]
            cam_positions.append(default_positions[i])
            print(f"Using default position for {cam}: {default_positions[i]}")
    
    return cam_positions, cam_colors

def get_cam_rotation_matrices():
    # Returns the camera rotation matrices in world coordinates based on calibration data.
    data_path = "data"
    camera_folders = ["cam1", "cam2", "cam3", "cam4"]
    cam_rotations = []
    
    for cam in camera_folders:
        try:
            fs = cv2.FileStorage(os.path.join(data_path, cam, "config.xml"), cv2.FILE_STORAGE_READ)
            if not fs.isOpened():
                raise Exception(f"Failed to open config.xml for {cam}")
            rvec = fs.getNode("RotationVector").mat()
            fs.release()
            rot_mat, _ = cv2.Rodrigues(rvec)
            
            # Convert OpenCV rotation matrix to OpenGL rotation matrix
            # Need to flip Y and Z axes
            flip_mat = np.array([
                [1, 0, 0],
                [0, 1, 0],  # Flip Y axis
                [0, 0, 1]   # Flip Z axis
            ])
            rot_mat_gl = np.matmul(flip_mat, rot_mat)
            
            rot_mat_4x4 = glm.mat4(1)
            for i in range(3):
                for j in range(3):
                    rot_mat_4x4[i][j] = rot_mat_gl[i, j]
            
            cam_rotations.append(rot_mat_4x4)
            print(f"Loaded camera rotation for {cam}: {rvec.flatten()}")
                
        except Exception as e:
            print(f"Error loading camera rotation for {cam}: {e}")
            # Use default rotation if loading fails
            cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
            rot_mat = glm.mat4(1)
            idx = len(cam_rotations)
            rot_mat = glm.rotate(rot_mat, cam_angles[idx % 4][0] * np.pi / 180, [1, 0, 0])
            rot_mat = glm.rotate(rot_mat, cam_angles[idx % 4][1] * np.pi / 180, [0, 1, 0])
            rot_mat = glm.rotate(rot_mat, cam_angles[idx % 4][2] * np.pi / 180, [0, 0, 1])
            cam_rotations.append(rot_mat)
    
    return cam_rotations