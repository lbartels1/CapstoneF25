import cv2
import numpy as np

# Load camera calibration data
camera_matrix = np.load('camera_matrix.npy')
dist_coeffs = np.load('dist_coeffs.npy')

# Define blue color range in HSV
LOWER_BLUE = np.array([100, 150, 50])
UPPER_BLUE = np.array([140, 255, 255])

# ArUco dictionary and marker size (in meters)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
MARKER_LENGTH = 0.05  # 5 cm

# Define floor plane homography or rvec/tvec if known
# For now, assume flat floor with homography approach

def detect_blue_tape(frame):
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

    # Morphological clean-up
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask

def compute_real_world_length(cnt, H):
    # Convert contour to homogeneous coordinates
    pts = cnt.reshape(-1, 2)
    pts_homog = cv2.perspectiveTransform(np.array([pts], dtype=np.float32), H)[0]

    # Compute length as max distance between any two points
    max_length = 0
    for i in range(len(pts_homog)):
        for j in range(i + 1, len(pts_homog)):
            dist = np.linalg.norm(pts_homog[i] - pts_homog[j])
            max_length = max(max_length, dist)
    return max_length

def detect_aruco_marker(frame):
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, ARUCO_DICT)
    if ids is not None:
        frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs)
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
        return frame, ids, tvecs
    return frame, None, None

def main():
    cap = cv2.VideoCapture(0)

    # Define world points for homography (assume z=0 plane)
    # Manually define 4 known points in the image and their corresponding real-world coordinates
    # You must calibrate these once
    image_points = np.array([
        [x1, y1],
        [x2, y2],
        [x3, y3],
        [x4, y4]
    ], dtype=np.float32)

    world_points = np.array([
        [X1, Y1],
        [X2, Y2],
        [X3, Y3],
        [X4, Y4]
    ], dtype=np.float32)

    # Compute homography from image to world plane
    H, _ = cv2.findHomography(image_points, world_points)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Undistort frame
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # Detect blue tape
        contours, mask = detect_blue_tape(frame)

        for cnt in contours:
            if cv2.contourArea(cnt) < 500:
                continue

            # Draw contour
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

            # Compute real-world length
            length_m = compute_real_world_length(cnt, H)
            cv2.putText(frame, f"Length: {length_m:.2f} m", tuple(cnt[0][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Detect ArUco marker and estimate position
        frame, ids, tvecs = detect_aruco_marker(frame)

        # Show result
        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
