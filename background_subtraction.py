import cv2
import numpy as np

# Function that builds a background model using MOG2 from the given video
def build_background_model_mog2(bg_video_path, num_frames=30):
    cap = cv2.VideoCapture(bg_video_path)
    backSub = cv2.createBackgroundSubtractorMOG2(history=num_frames, varThreshold=16, detectShadows=True)
    count = 0
    while count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        backSub.apply(frame)
        count += 1
    cap.release()
    bg_bgr = backSub.getBackgroundImage()
    return bg_bgr

# Function that computes a foreground mask by comparing the current frame to the background model
def get_foreground_mask(frame, bg_bgr, fixed_thresh=(30, 30, 30), min_blob_area=500):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bg_hsv = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV)
    diff = cv2.absdiff(frame_hsv, bg_hsv)
    _, mask_h = cv2.threshold(diff[:, :, 0], fixed_thresh[0], 255, cv2.THRESH_BINARY)
    _, mask_s = cv2.threshold(diff[:, :, 1], fixed_thresh[1], 255, cv2.THRESH_BINARY)
    _, mask_v = cv2.threshold(diff[:, :, 2], fixed_thresh[2], 255, cv2.THRESH_BINARY)
    combined_mask = cv2.bitwise_or(mask_h, mask_s)
    combined_mask = cv2.bitwise_or(combined_mask, mask_v)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    processed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(processed_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_blob_area:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)
    return filtered_mask
