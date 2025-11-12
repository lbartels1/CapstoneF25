import cv2
import cv2.aruco as aruco
import numpy as np

# ---- Configuration ----
imagePath = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\ARuCo_ID\test_image2.jpg"  # 🔁 Replace with your actual image path
markerLength = 0.1  # in meters
estimatePose = True
showRejected = True
resizeFactor = 0.15  # 🔍 Scale down for display

# ---- Camera Calibration (example, replace with real calibration if needed) ----
camMatrix = np.array([[1000, 0, 640],
                      [0, 1000, 360],
                      [0, 0, 1]], dtype=np.float32)
distCoeffs = np.zeros((5, 1), dtype=np.float32)

# ---- Load ArUco Dictionary and Detector ----
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detectorParams = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, detectorParams)

# ---- Load the Image ----
image = cv2.imread(imagePath)
if image is None:
    raise FileNotFoundError(f"Image not found: {imagePath}")

# ---- Define Object Points for Marker Corners ----
objPoints = np.array([
    [-markerLength / 2,  markerLength / 2, 0],
    [ markerLength / 2,  markerLength / 2, 0],
    [ markerLength / 2, -markerLength / 2, 0],
    [-markerLength / 2, -markerLength / 2, 0]
], dtype=np.float32).reshape((4, 1, 3))

# ---- Detect Markers ----
corners, ids, rejected = detector.detectMarkers(image)

rvecs, tvecs = [], []

# ---- Pose Estimation ----
if estimatePose and ids is not None:
    for i in range(len(ids)):
        retval, rvec, tvec = cv2.solvePnP(objPoints, corners[i], camMatrix, distCoeffs)
        rvecs.append(rvec)
        tvecs.append(tvec)

# ---- Draw Detected Markers with Thicker Borders ----
imageCopy = image.copy()

if ids is not None:
    for i in range(len(ids)):
        corners_i = corners[i][0].astype(int)  # shape (4, 2)
        # Draw each edge of the marker with thickness
        for j in range(4):
            pt1 = tuple(corners_i[j])
            pt2 = tuple(corners_i[(j + 1) % 4])
            cv2.line(imageCopy, pt1, pt2, (0, 255, 0), thickness=4)

        # Draw the ID at the center
        cX = int(np.mean(corners_i[:, 0]))
        cY = int(np.mean(corners_i[:, 1]))
        cv2.putText(imageCopy, str(ids[i][0]), (cX - 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2)

        # Draw pose axes if needed
        if estimatePose:
            cv2.drawFrameAxes(imageCopy, camMatrix, distCoeffs, rvecs[i], tvecs[i], markerLength * 1.5)

# ---- Draw Rejected Markers (optional) ----
if showRejected and rejected is not None and len(rejected) > 0:
    for r in rejected:
        corners_r = r[0].astype(int)
        for j in range(4):
            pt1 = tuple(corners_r[j])
            pt2 = tuple(corners_r[(j + 1) % 4])
            cv2.line(imageCopy, pt1, pt2, (100, 0, 255), thickness=2)

# ---- Resize Image for Display ----
resizedImage = cv2.resize(imageCopy, (0, 0), fx=resizeFactor, fy=resizeFactor)

# ---- Show Result ----
cv2.imshow("Aruco Detection", resizedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
