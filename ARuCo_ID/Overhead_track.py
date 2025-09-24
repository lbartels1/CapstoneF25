# Lars Bartels - Capstone
# ArUco Marker Pose Estimation with World Coordinate Conversion

import cv2
import cv2.aruco as aruco
import numpy as np
import time

# ==== CONFIGURATION ====
video = ""  # Leave blank for live webcam
camId = 1
markerLength = 0.47625  # In meters
estimatePose = True
showRejected = False

# Paths to camera calibration and pose
mtx = np.loadtxt(r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\camera_mtx.txt", delimiter=',')
dist = np.loadtxt(r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\camera_dist.txt", delimiter=',')
cam_tvec = np.load(r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\camera_tvec.npy")
cam_rvec = np.load(r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\camera_rvec.npy")

# ==== FUNCTIONS ====

def pose_to_matrix(rvec, tvec):
    """Convert rotation vector and translation vector into 4x4 transformation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

# ==== SETUP ====

camMatrix = mtx
distCoeffs = dist

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detectorParams = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, detectorParams)

inputVideo = cv2.VideoCapture(video if video else camId)
waitTime = 0 if video else 10

# Define marker corners (object points)
objPoints = np.array([
    [-markerLength / 2,  markerLength / 2, 0],
    [ markerLength / 2,  markerLength / 2, 0],
    [ markerLength / 2, -markerLength / 2, 0],
    [-markerLength / 2, -markerLength / 2, 0]
], dtype=np.float32).reshape((4, 1, 3))

# ==== POSE TRACKING ====

all_rvecs = []
all_tvecs = []
translated_tvecs = []

previous_time = time.time()
previous_tvecs = {}

frame_count = 0
cam_T = pose_to_matrix(cam_rvec, cam_tvec)

# ==== KALMAN FILTER ====
class KalmanFilter3D:
    def __init__(self):
        # State: [x, y, z, vx, vy, vz]
        self.x = np.zeros((6, 1))
        self.P = np.eye(6) * .75
        self.F = np.eye(6)
        dt = 1  # Assume 1 frame per step
        for i in range(3):
            self.F[i, i+3] = dt
        self.Q = np.eye(6) * 0.001
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.R = np.eye(3) * 1

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        z = z.reshape((3, 1))
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def get_position(self):
        return self.x[:3].flatten()

kf = KalmanFilter3D()
kf_initialized = False

# ==== MAIN LOOP ====
while inputVideo.isOpened():
    ret, image = inputVideo.read()
    if not ret:
        break

    start_time = time.time()
    frame_count += 1

    corners, ids, rejected = detector.detectMarkers(image)

    rvecs_frame = []
    tvecs_frame = []

    if estimatePose and ids is not None:
        for i in range(len(ids)):
            success, rvec, tvec = cv2.solvePnP(
                objPoints, corners[i], camMatrix, distCoeffs
            )
            if not success:
                continue

            rvecs_frame.append(rvec)
            tvecs_frame.append(tvec)

            # Transform to world coordinate
            robot_T_cam = pose_to_matrix(rvec, tvec)
            robot_T_world = cam_T @ robot_T_cam
            robot_pos_world = robot_T_world[:3, 3]

            # Kalman filter
            if not kf_initialized:
                kf.x[:3] = robot_pos_world.reshape((3, 1))
                kf_initialized = True
            kf.predict()
            kf.update(robot_pos_world)
            filtered_pos = kf.get_position()
            translated_tvecs.append(filtered_pos)

    all_rvecs.extend(rvecs_frame)
    all_tvecs.extend(tvecs_frame)

    # ==== DRAW ====
    display_image = image.copy()
    if ids is not None:
        aruco.drawDetectedMarkers(display_image, corners, ids)
        for rvec, tvec in zip(rvecs_frame, tvecs_frame):
            cv2.drawFrameAxes(display_image, camMatrix, distCoeffs, rvec, tvec, markerLength * 1.5)

    if showRejected and rejected is not None and len(rejected) > 0:
        aruco.drawDetectedMarkers(display_image, rejected, borderColor=(100, 0, 255))

    # ==== UI ====
    cv2.putText(display_image, f"Frame: {frame_count} | Markers: {len(ids) if ids is not None else 0}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Pose Estimation", display_image)

    key = cv2.waitKey(waitTime)
    if key == 27:  # ESC
        break

inputVideo.release()
cv2.destroyAllWindows()

# ==== SAVE OUTPUT ====

if translated_tvecs:
    np.save("translated_tvec.npy", np.array(translated_tvecs))
    print(f"\n✅ Saved {len(translated_tvecs)} robot world positions to 'translated_tvec.npy'")
    np.save("translated_rvec.npy", np.array(all_rvecs))
    print(f"\n✅ Saved {len(all_rvecs)} robot world positions to 'translated_rvec.npy'")
else:
    print("\n⚠️ No robot poses were detected. Check marker visibility and camera calibration.")
