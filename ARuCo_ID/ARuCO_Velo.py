# Lars Bartels
# Capstone
# ARuCo marker pose estimation

import cv2
import cv2.aruco as aruco
import numpy as np
import time

# TODO: get this imported to a ROS node so the pose estimation and location of the markers can be found
# TODO: get parallization of multiple cameras (most likely to be done by a master node that detects number of camera's ) 
# TODO: Tune Kalman Filter
# TODO: estimate velocity from kalman filter. 


F = np.array([[1, 1], [0, 1]])
B = np.array([[0.5], [1]])
H = np.array([[1, 0]])
Q = np.array([[0.1, 0], [0, 0.01]])
R = np.array([[0.05]])


class KalmanFilter:
    def __init__(self, F, B, H, Q, R, x0, P0):
        self.F = F
        self.B = B
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x0
        self.P = P0

    def predict(self, u):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x
    
    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.x
    


# ---- Configurations ----
video = ""  # Replace with video path if needed
camId = 1 # 0 is laptop camera, 1 is webcam
markerLength = 0.1  # Marker side length in meters (adjust accordingly)
estimatePose = True
showRejected = False

previous_time = time.time()
previous_tvecs = {}

velo_total = 0

period = 30
window_size = 15
velo_array = [0] * window_size

font = cv2.FONT_HERSHEY_COMPLEX
bottomLeftCornerOfText = (10,30)
fontScale = 1
fontColor = (255,255,255)
thickness = 1 
lineType = 1

# file_name = "./../Camera/cam_calibration_mtx.txt"

mtx = np.loadtxt(r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\camera_mtx.txt", delimiter=',')
dist = np.loadtxt(r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\camera_dist.txt", delimiter=',')

print(mtx, dist)

# Load camera parameters (replace with actual calibration)
camMatrix = mtx  # Dummy identity matrix
distCoeffs = dist  # Dummy zero distortion


# ---- Load ArUco Dictionary and Detector Parameters ----
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detectorParams = aruco.DetectorParameters()

detector = aruco.ArucoDetector(dictionary, detectorParams)

# ---- Open Video/Camera ----
if video:
    inputVideo = cv2.VideoCapture(video)
    waitTime = 0
else:
    inputVideo = cv2.VideoCapture(camId)
    waitTime = 10

# ---- Coordinate system for pose estimation ----
objPoints = np.array([
    [-markerLength / 2,  markerLength / 2, 0],
    [ markerLength / 2,  markerLength / 2, 0],
    [ markerLength / 2, -markerLength / 2, 0],
    [-markerLength / 2, -markerLength / 2, 0]
], dtype=np.float32)

objPoints = objPoints.reshape((4, 1, 3))

# ---- Main Loop ----
totalTime = 0
totalIterations = 0

x0 = np.array([[0], [1]])
P0 = np.array([[1, 0], [0, 1]])

# Initial state: assume position = 0, velocity = 0 for all axes
x0 = np.array([[0], [0]])
P0 = np.eye(2)

kf_x = KalmanFilter(F, B, H, Q, R, x0.copy(), P0.copy())
kf_y = KalmanFilter(F, B, H, Q, R, x0.copy(), P0.copy())
kf_z = KalmanFilter(F, B, H, Q, R, x0.copy(), P0.copy())

raw_tvecs = []
filtered_tvecs = []

while inputVideo.isOpened():
    ret, image = inputVideo.read()
    if not ret:
        break

    startTime = time.time()

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(image)

    current_time = time.time()
    delta_t = current_time - previous_time

    rvecs = []
    tvecs = []

    velocity = 0.00
    velo_total = 0

    kalman_velocity = np.array([[],[],[]])

    # Estimate pose for each detected marker
    if estimatePose and ids is not None:
        for i in range(len(ids)):
            retval, rvec, tvec = cv2.solvePnP(
                objPoints, corners[i], camMatrix, distCoeffs)
            rvecs.append(rvec)

        
            # TODO: Apply the kalman filter her 
            # tvec = np.array([[x], [y], [z]])
            raw_tvec = tvec.reshape(3, 1)  # Ensure shape (3, 1)

            # Use each component as measurement (z) and control (u)
            z_x = np.array([[raw_tvec[0, 0]]])
            z_y = np.array([[raw_tvec[1, 0]]])
            z_z = np.array([[raw_tvec[2, 0]]])

            # Control input (we don’t really have control inputs here, use 0)
            u = np.array([[0]])

            kf_x.update(z_x)
            kf_y.update(z_y)
            kf_z.update(z_z)

            x_filtered = kf_x.predict(u)
            y_filtered = kf_y.predict(u)
            z_filtered = kf_z.predict(u)

            filtered_tvec = np.array([
                [x_filtered[0, 0]],
                [y_filtered[0, 0]],
                [z_filtered[0, 0]]
            ])

            kalman_velocity = np.array([
                [x_filtered[1, 0]],
                [y_filtered[1, 0]],
                [z_filtered[1, 0]]
            ])

            # recorded_tvecs.append(tvec)
            # TODO: Uncomment to switch to filtered data
            # tvecs.append(filtered_tvec)
            tvecs.append(tvec)
            raw_tvecs.append(tvec)
            filtered_tvecs.append(filtered_tvec)

            marker_id = ids[i][0]
            current_tvec = tvecs[i][0]

            if marker_id in previous_tvecs:
                prev_tvec = previous_tvecs[marker_id]
                # Calculate velocity components (vx, vy, vz)
                velocity = (current_tvec - prev_tvec) / delta_t
                velo_array.append(velocity)
                velo_array.pop(0)
                # print(f"Marker ID {marker_id} Velocity: {np.round(velocity,2)} m/s")

            # velo_total[marker_id] += velocity
            previous_tvecs[marker_id] = current_tvec
            # print(tvec)
            

            # previous_tvecs.append(tvec)

    currentTime = time.time() - startTime
    totalTime += currentTime
    totalIterations += 1


    # if totalIterations % period == 0:
        # print(f"velocity = {np.round((velo_total/period), 2)} m/s")
        # print(f"velocity = {kalman_velocity} m/s")

    for item in velo_array:
        velo_total += item
    velo_average = velo_total / len(velo_array)
        
    previous_time = current_time
    imageCopy = image.copy()

    # Draw detections and pose
    cv2.putText(imageCopy, "X velo: " + str(np.round(np.abs(kalman_velocity[0]),2)), bottomLeftCornerOfText,font, fontScale, fontColor, thickness, lineType)
    if ids is not None:
        aruco.drawDetectedMarkers(imageCopy, corners, ids)
        if estimatePose:
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.drawFrameAxes(imageCopy, camMatrix, distCoeffs, rvec, tvec, markerLength)
                

    if showRejected and rejected is not None and len(rejected) > 0:
        aruco.drawDetectedMarkers(imageCopy, rejected, borderColor=(100, 0, 255))

    cv2.imshow("out", imageCopy)
    key = cv2.waitKey(waitTime)

    if key == 27:  # ESC key
        break

np.save("raw_tvecs", raw_tvecs)
np.save("filtered_tvecs", filtered_tvecs)

inputVideo.release()
cv2.destroyAllWindows()
