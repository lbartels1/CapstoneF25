import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import time

# ---- CONFIG ----
cam_index = 1
marker_length = 0.47625  # meters (same convention as other code)
camera_mtx_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\camera_mtx.txt"
camera_dist_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\camera_dist.txt"
image_points_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\image_points.npy"
world_points_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\world_points.npy"
cam_rvec_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\camera_rvec.npy"
cam_tvec_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\camera_tvec.npy"

# ---- helpers ----
def pose_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

# ---- load calibration & homography ----
camera_matrix = np.loadtxt(camera_mtx_path, delimiter=',')
dist_coeffs = np.loadtxt(camera_dist_path, delimiter=',')
image_points = np.load(image_points_path)    # Nx2 image points
world_points = np.load(world_points_path)    # Nx2 world points (meters)

H, status = cv2.findHomography(image_points, world_points, method=cv2.RANSAC)
if H is None:
    raise RuntimeError("Homography failed. Check image/world correspondences.")
Hw = np.linalg.inv(H)

# try load camera extrinsics (optional)
have_cam_extrinsics = os.path.exists(cam_rvec_path) and os.path.exists(cam_tvec_path)
if have_cam_extrinsics:
    cam_rvec = np.load(cam_rvec_path)
    cam_tvec = np.load(cam_tvec_path)
    cam_T = pose_to_matrix(cam_rvec, cam_tvec)
else:
    cam_rvec = cam_tvec = None
    cam_T = None

# ---- ArUco setup ----
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
params = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, params)

# object points for solvePnP (order matches returned corners: tl, tr, br, bl)
objPoints = np.array([
    [-marker_length / 2,  marker_length / 2, 0],
    [ marker_length / 2,  marker_length / 2, 0],
    [ marker_length / 2, -marker_length / 2, 0],
    [-marker_length / 2, -marker_length / 2, 0]
], dtype=np.float32).reshape((4, 1, 3))

# ---- realtime capture ----
cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open camera {cam_index}")

# setup matplotlib interactive plot
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title("ArUco markers in world plane (meters)")
ax.set_xlabel("X (tiles)")
ax.set_ylabel("Y (tiles)")
ax.axis('equal')
ax.grid(True)

# keep scatter/arrow handles to update
scatter = None
arrow_artists = []
pose_points = []

frame_idx = 0
fps_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    # undistort frame
    frame_und = cv2.undistort(frame, camera_matrix, dist_coeffs)

    # detect markers
    corners, ids, rejected = detector.detectMarkers(frame_und)

    world_markers = []
    if ids is not None and len(ids) > 0:
        for i, c in enumerate(corners):
            id_val = int(ids[i])
            img_corners = c.reshape(4, 2).astype(np.float32)  # tl,tr,br,bl

            # map corners to world plane via homography
            world_corners = cv2.perspectiveTransform(np.array([img_corners]), H)[0]  # (4,2)
            center_world = world_corners.mean(axis=0)

            # compute yaw from first edge (tl->tr)
            edge = world_corners[1] - world_corners[0]
            yaw = math.atan2(edge[1], edge[0])  # radians in world plane

            marker_info = {
                "id": id_val,
                "img_corners": img_corners,
                "world_corners": world_corners,
                "center_world": center_world,
                "yaw": yaw
            }

            # estimate pose in camera frame with solvePnP and transform to world frame if extrinsics available
            try:
                retval, rvec, tvec = cv2.solvePnP(objPoints, img_corners.reshape(4,1,2), camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if retval:
                    marker_info["rvec_cam"] = rvec
                    marker_info["tvec_cam"] = tvec
                    if cam_T is not None:
                        marker_T_cam = pose_to_matrix(rvec, tvec)
                        marker_T_world = cam_T @ marker_T_cam
                        t_world = marker_T_world[:3, 3]
                        Rw = marker_T_world[:3, :3]
                        rvec_world, _ = cv2.Rodrigues(Rw)
                        yaw_world = math.atan2(Rw[1, 0], Rw[0, 0])
                        marker_info["tvec_world"] = t_world.flatten()
                        marker_info["rvec_world"] = rvec_world.flatten()
                        marker_info["yaw_world_from_pose"] = yaw

                        print(yaw)
            except Exception:
                pass

            world_markers.append(marker_info)

    # ---- OpenCV overlay ----
    vis = frame_und.copy()
    for m in world_markers:
        pts = m["img_corners"].astype(int)
        cv2.polylines(vis, [pts.reshape(-1,1,2)], isClosed=True, color=(0,255,0), thickness=2)
        img_center = cv2.perspectiveTransform(np.array([[m["center_world"]]], dtype=np.float32), Hw)[0,0]
        cv2.circle(vis, (int(img_center[0]), int(img_center[1])), 4, (0,0,255), -1)
        cv2.putText(vis, f"id:{m['id']}", (int(img_center[0])+5, int(img_center[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        tip_world = m["center_world"] + np.array([math.cos(m["yaw"]), math.sin(m["yaw"])]) * (marker_length * 0.75)
        tip_img = cv2.perspectiveTransform(np.array([[tip_world]], dtype=np.float32), Hw)[0,0]
        cv2.arrowedLine(vis, (int(img_center[0]), int(img_center[1])), (int(tip_img[0]), int(tip_img[1])), (0,0,255), 2, tipLength=0.2)

    # show frame
    cv2.imshow("ArUco realtime (undistorted)", vis)

    # ---- update matplotlib plot at a lower rate (every N frames) ----
    if frame_idx % 5 == 0:
        ax.clear()
        ax.set_title("ArUco markers in world plane (meters)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.axis('equal')
        ax.grid(True)
        ax.set_xlim((0,10))
        ax.set_ylim((0,10))

        for m in world_markers:
            wc = m["world_corners"]
            xs = [p[0] for p in wc] + [wc[0,0]]
            ys = [p[1] for p in wc] + [wc[0,1]]
            ax.plot(xs, ys, '-o', label=f"id {m['id']}")
            cx, cy = m["center_world"]
            ax.text(cx, cy, f"id {m['id']}", color='k', fontsize=8, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
            ax.arrow(cx, cy, math.cos(m["yaw"])*marker_length*0.75, math.sin(m["yaw"])*marker_length*0.75,
                     head_width=marker_length*0.05, head_length=marker_length*0.05, color='r', length_includes_head=True)

        ax.legend(loc='upper right')
        plt.pause(0.001)

    # handle keys
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show(block=False)