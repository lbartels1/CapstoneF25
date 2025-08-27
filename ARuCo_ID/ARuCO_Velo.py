# Lars Bartels
# Capstone
# ARuCo marker pose estimation

import cv2
import cv2.aruco as aruco
import numpy as np
import time

# TODO: get this imported to a ROS node so the pose estimation and location of the markers can be found
# TODO: get parallization of multiple cameras (most likely to be done by a master node that detects number of camera's ) 

# ---- Configurations ----
video = ""  # Replace with video path if needed
camId = 0
markerLength = 0.1  # Marker side length in meters (adjust accordingly)
estimatePose = True
showRejected = True

previous_time = time.time()
previous_tvecs = {}

# Load camera parameters (replace with actual calibration)
camMatrix = np.eye(3, dtype=np.float32)  # Dummy identity matrix
distCoeffs = np.zeros((5, 1), dtype=np.float32)  # Dummy zero distortion

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

    # Estimate pose for each detected marker
    if estimatePose and ids is not None:
        for i in range(len(ids)):
            retval, rvec, tvec = cv2.solvePnP(
                objPoints, corners[i], camMatrix, distCoeffs)
            marker_id = ids[i][0]
            current_tvec = tvecs[i][0]

            if marker_id in previous_tvecs:
                prev_tvec = previous_tvecs[marker_id]
                # Calculate velocity components (vx, vy, vz)
                velocity = (current_tvec - prev_tvec) / delta_t
                print(f"Marker ID {marker_id} Velocity: {velocity} m/s")

            previous_tvecs[marker_id] = current_tvec

            rvecs.append(rvec)
            tvecs.append(tvec)
            # previous_tvecs.append(tvec)

    currentTime = time.time() - startTime
    totalTime += currentTime
    totalIterations += 1

    if totalIterations % 30 == 0:
        print(f"Detection Time = {currentTime * 1000:.2f} ms "
              f"(Mean = {1000 * totalTime / totalIterations:.2f} ms)")


    previous_time = current_time

    # Draw detections and pose
    imageCopy = image.copy()
    if ids is not None:
        aruco.drawDetectedMarkers(imageCopy, corners, ids)
        if estimatePose:
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.drawFrameAxes(imageCopy, camMatrix, distCoeffs, rvec, tvec, markerLength * 1.5)

    if showRejected and rejected is not None and len(rejected) > 0:
        aruco.drawDetectedMarkers(imageCopy, rejected, borderColor=(100, 0, 255))

    cv2.imshow("out", imageCopy)
    key = cv2.waitKey(waitTime)
    if key == 27:  # ESC key
        break

inputVideo.release()
cv2.destroyAllWindows()
