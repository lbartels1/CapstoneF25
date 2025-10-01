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
camId = 1 # 0 is laptop camera, 1 is webcam
markerLength = 0.47625  # Marker side length in meters (adjust accordingly)
estimatePose = True
showRejected = False

previous_time = time.time()
previous_tvecs = {}

velo_total = 0

period = 30
window_size = 45
velo_array = [0] * window_size

font = cv2.FONT_HERSHEY_COMPLEX
bottomLeftCornerOfText = (10,30)
fontScale = 1
fontColor = (255,255,255)
thickness = 1 
lineType = 2

# file_name = "./../Camera/cam_calibration_mtx.txt"

mtx = np.loadtxt(r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\camera_mtx.txt", delimiter=',')
dist = np.loadtxt(r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\camera_dist.txt", delimiter=',')
cam_tvec = np.load(r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\camera_tvec.npy")
cam_rvec = np.load(r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\camera_rvec.npy")

print(mtx, dist)

# Load camera parameters (replace with actual calibration)
camMatrix = mtx  # Dummy identity matrix
distCoeffs = dist  # Dummy zero distortion


# ---- Load ArUco Dictionary and Detector Parameters ----
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detectorParams = aruco.DetectorParameters()

detector = aruco.ArucoDetector(dictionary, detectorParams)

def pose_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


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

tvecs = []
translated_tvecs = []
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

    velocity = 0.00
    velo_total = 0

    # Estimate pose for each detected marker
    if estimatePose and ids is not None:
        for i in range(len(ids)):
            retval, rvec, tvec = cv2.solvePnP(
                objPoints, corners[i], camMatrix, distCoeffs)
            rvecs.append(rvec)
            tvecs.append(tvec)

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

    for item in velo_array:
        velo_total += item
    velo_average = velo_total / len(velo_array)
        
    previous_time = current_time
    imageCopy = image.copy()

    # Draw detections and pose
    cv2.putText(imageCopy, str(np.round(np.abs(velo_average),2)), bottomLeftCornerOfText,font, fontScale, fontColor, thickness, lineType)
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

cam_T = pose_to_matrix(cam_rvec, cam_tvec)

translated_tvecs = []

for tvec, rvec in zip(tvecs, rvecs):
    robot_T_cam = pose_to_matrix(rvec, tvec)
    robot_T_world = cam_T @ robot_T_cam
    robot_pos_world = robot_T_world[:3, 3]  # extract x, y, z
    translated_tvecs.append(robot_pos_world)


np.save("translated_tvec", translated_tvecs)
inputVideo.release()
cv2.destroyAllWindows()