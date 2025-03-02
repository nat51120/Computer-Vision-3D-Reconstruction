import cv2
import numpy as np
import random

# Setting the expected chessboard pattern size (number of inner corners)
pattern_size = (8, 6)
square_size  = 115
full_grid_size = (10, 8)

# Preparing the object points (3D points in the chessboard coordinate system)
inner_objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
inner_objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
inner_objp *= square_size

# Function that extract frames from the chessboard.avi videos allowing the user to select which frames to extract
def extract_frames(video_path, num_frames=25):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = list(range(total_frames))
    random.shuffle(frame_indices)
    frames = []
    idx_pointer = 0
    while idx_pointer < total_frames and len(frames) < num_frames:
        frame_idx = frame_indices[idx_pointer]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            idx_pointer += 1
            continue
        cv2.imshow("Frame Selection (Press 'y'=accept, 'n'=reject, 'q'=quit)", frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('y'):
            frames.append(frame.copy())
            print(f"Accepted frame {frame_idx}. Total accepted: {len(frames)}")
        elif key == ord('n'):
            print(f"Rejected frame {frame_idx}")
        elif key == ord('q'):
            print("User requested to quit early.")
            break
        idx_pointer += 1
    cap.release()
    cv2.destroyAllWindows()
    return frames

# Function that provides an interactive manual annotation interface
def get_manual_corners(img):
    clicked_points = []
    img_copy = img.copy()
    original_img = img.copy()

    # Function to update the display of the manual annotation window
    def update_display():
        nonlocal img_copy
        img_copy = original_img.copy()
        for pt in clicked_points:
            cv2.circle(img_copy, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
        cv2.imshow("Manual Annotation", img_copy)

    # Function that displays a zoomed-in view to help the user click more precisely
    def zoom_click_refinement(img, point, zoom_factor=3, window_size=400):
        x, y = int(point[0]), int(point[1])
        h, w = img.shape[:2]
        half = window_size // 2
        x1, y1 = max(x - half, 0), max(y - half, 0)
        x2, y2 = min(x + half, w), min(y + half, h)

        roi = img[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]

        zoomed_width  = int(roi_w * zoom_factor)
        zoomed_height = int(roi_h * zoom_factor)
        zoomed = cv2.resize(roi, (zoomed_width, zoomed_height))

        scale_x = roi_w / zoomed_width
        scale_y = roi_h / zoomed_height

        win_name = "Zoomed Refinement"
        cv2.imshow(win_name, zoomed)
        refined_point = None

        # Callback function for the zoomed-in refinement window
        def zoom_callback(event, zx, zy, flags, param):
            nonlocal refined_point
            if event == cv2.EVENT_LBUTTONDOWN:
                rx = x1 + zx * scale_x
                ry = y1 + zy * scale_y
                refined_point = (rx, ry)
                cv2.destroyWindow(win_name)

        cv2.setMouseCallback(win_name, zoom_callback)

        while refined_point is None:
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                refined_point = (x, y)
                cv2.destroyWindow(win_name)
                break

        return refined_point

    #  Mouse callback function to capture clicks using zoomed-in refinement
    def click_event(event, x, y, flags, params):
        nonlocal clicked_points
        if event == cv2.EVENT_LBUTTONDOWN:
            refined = zoom_click_refinement(img, (x, y))
            clicked_points.append(refined)
            update_display()
            print(f"Point selected: {refined}")

    cv2.namedWindow("Manual Annotation", cv2.WINDOW_NORMAL)
    cv2.imshow("Manual Annotation", img_copy)
    cv2.setMouseCallback("Manual Annotation", click_event)

    print("Please click on the 4 OUTER corners (10×8 boundary) in this order:")
    print("   (1) Top-Left")
    print("   (2) Top-Right")
    print("   (3) Bottom-Right")
    print("   (4) Bottom-Left")
    print("Press 'u' to undo, 'a' to accept after 4 points, or ESC to cancel.")

    while True:
        cv2.imshow("Manual Annotation", img_copy)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('u'):
            if clicked_points:
                clicked_points.pop()
                update_display()
                print("Undid the last click.")

        if len(clicked_points) == 4:
            print("4 points selected. Press 'a' to accept, or 'u' to undo the final click.")
            key_confirm = cv2.waitKey(0) & 0xFF
            if key_confirm == ord('a'):
                break
            elif key_confirm == ord('u'):
                if clicked_points:
                    clicked_points.pop()
                    update_display()
                    print("Undid the last click. Please click again.")

        if key == 27:  # ESC
            print("Manual annotation canceled.")
            break

    cv2.destroyWindow("Manual Annotation")

    if len(clicked_points) == 4:
        return clicked_points
    else:
        return None

# Function that linearly interpolates all chessboard points from the given outer corners
def interpolate_with_homography(corners_4, grid_size):
    num_cols, num_rows = grid_size
    src_points = np.array([
        [0, 0],
        [num_cols - 1, 0],
        [num_cols - 1, num_rows - 1],
        [0, num_rows - 1]
    ], dtype=np.float32)
    dst_points = np.array(corners_4, dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_points, dst_points)
    grid = np.array([[c, r] for r in range(num_rows) for c in range(num_cols)], dtype=np.float32)
    grid = grid.reshape(num_rows, num_cols, 2)
    grid_reshaped = grid.reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(grid_reshaped, H)
    warped = warped.reshape(num_rows, num_cols, 2)
    return warped

# Function that determines whether the chessboard is placed vertically or horizontally
def determine_grid_size(corners, horizontal_grid_size=11, vertical_grid_size=8):
    tl = np.array(corners[0], dtype=np.float32)
    tr = np.array(corners[1], dtype=np.float32)
    bl = np.array(corners[3], dtype=np.float32)
    width = np.linalg.norm(tr - tl)
    height = np.linalg.norm(bl - tl)
    if width >= height:
        return (horizontal_grid_size, vertical_grid_size)
    else:
        return (vertical_grid_size, horizontal_grid_size)

# Function that calibrates the cameras using openCV function
def calibrate_camera(object_points, image_points, img_size):
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, img_size, None, None
    )
    return cameraMatrix, distCoeffs, ret, rvecs, tvecs