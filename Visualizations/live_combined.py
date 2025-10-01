"""
live_combined.py

Runs ArUco detection (based on Overhead_track.py) in a background thread and
shows a live 3D matplotlib visualization (based on live_robot_visualization.py)
that updates as new poses arrive. Orientation markers are fixed-length and
rotate only about Z (heading), as requested.

This file does not modify the originals; it copies the minimal logic it needs.

Usage: python live_combined.py

Notes:
- Adjust camera/device paths and IDs if necessary.
- By default the camera feed is shown in an OpenCV window; set show_camera=False
  if you don't want it.
"""

import threading
import time
import sys
import os

import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation

# -------------------- Configuration (tweak as needed) --------------------
show_camera = True
video_source = ""  # empty for webcam
cam_id = 1
markerLength = 0.47625  # meters (same as Overhead_track.py)

# Paths (match your repo) - update if your layout differs
base = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25"
cam_mtx_path = os.path.join(base, "camera_mtx.txt")
cam_dist_path = os.path.join(base, "camera_dist.txt")
cam_tvec_path = os.path.join(base, "camera_tvec.npy")
cam_rvec_path = os.path.join(base, "camera_rvec.npy")

# Visualization settings
arrow_length = 2  # fixed length of orientation markers (meters)
plot_interval_ms = 50

# -------------------- Utilities (copied/adapted) --------------------

def pose_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T


def matrix_to_pose(T):
    R = T[:3, :3]
    tvec = T[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    return rvec, tvec


# Simple KalmanFilter3D copied from Overhead_track for smoothing
class KalmanFilter3D:
    def __init__(self):
        # State: [x, y, z, vx, vy, vz]
        self.x = np.zeros((6, 1))
        self.P = np.eye(6) * 0.75
        self.F = np.eye(6)
        dt = 1
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


# -------------------- Shared state between threads --------------------
state_lock = threading.Lock()
path_positions = []  # list of (x,y,z)
latest_rvec = None
latest_tvec = None
running = True

# -------------------- Detector thread --------------------

def detector_thread_func(stop_event):
    global latest_rvec, latest_tvec

    # Load camera calibration
    camMatrix = np.loadtxt(cam_mtx_path, delimiter=',')
    distCoeffs = np.loadtxt(cam_dist_path, delimiter=',')

    # Load camera pose (checkerboard) if available
    cam_tvec = np.load(cam_tvec_path)
    cam_rvec = np.load(cam_rvec_path)
    T_checkerboard_in_camera = pose_to_matrix(cam_rvec, cam_tvec)
    T_camera_in_checkerboard = np.linalg.inv(T_checkerboard_in_camera)

    # Set up ArUco detector
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    detectorParams = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, detectorParams)

    # Prepare object points
    objPoints = np.array([
        [-markerLength / 2,  markerLength / 2, 0],
        [ markerLength / 2,  markerLength / 2, 0],
        [ markerLength / 2, -markerLength / 2, 0],
        [-markerLength / 2, -markerLength / 2, 0]
    ], dtype=np.float32).reshape((4, 1, 3))

    # Kalman filter for smoothing
    kf = KalmanFilter3D()
    kf_initialized = False

    cap = cv2.VideoCapture(video_source if video_source else cam_id)
    waitTime = 1

    print("Detector: starting capture")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        corners, ids, rejected = detector.detectMarkers(frame)
        rvecs_frame = []
        tvecs_frame = []

        if ids is not None and len(ids) > 0:
            for i in range(len(ids)):
                # Using solvePnP directly as in Overhead_track
                success, rvec, tvec = cv2.solvePnP(objPoints, corners[i], camMatrix, distCoeffs)
                if not success:
                    continue

                # Transform marker pose from camera frame to checkerboard (floor) frame
                T_marker_in_camera = pose_to_matrix(rvec, tvec)
                T_marker_in_checkerboard = T_camera_in_checkerboard @ T_marker_in_camera
                robot_pos_world = T_marker_in_checkerboard[:3, 3]

                # Smooth via Kalman
                if not kf_initialized:
                    kf.x[:3] = robot_pos_world.reshape((3, 1))
                    kf_initialized = True
                kf.predict()
                kf.update(robot_pos_world)
                filtered_pos = kf.get_position()

                with state_lock:
                    path_positions.append(tuple(filtered_pos))
                    latest_rvec = cv2.Rodrigues(T_marker_in_checkerboard[:3, :3])[0]
                    latest_tvec = filtered_pos

                # Optionally draw frame axes on the camera feed
                if show_camera:
                    cv2.drawFrameAxes(frame, camMatrix, distCoeffs, rvec, tvec, markerLength * 1.5)

        # show camera UI
        if show_camera:
            cv2.imshow("Camera", frame)
            if cv2.waitKey(waitTime) & 0xFF == 27:
                stop_event.set()
                break

    cap.release()
    if show_camera:
        cv2.destroyAllWindows()
    print("Detector: stopped")


# -------------------- Visualization (main thread) --------------------
def run_visualization():
    # 2D XY visualization: path, current point, and heading-only orientation lines.
    global latest_rvec, latest_tvec

    fig, ax = plt.subplots(figsize=(8, 6))

    # initial empty plot elements
    # path_line, = ax.plot([], [], color='k', linewidth=1)
    point, = ax.plot([], [], 'ro')

    # orientation lines: forward (red) and left (green)
    orientation_lines = []

    def init():
        # path_line.set_data([], [])
        point.set_data([], [])
        # return (path_line, point)
        return (point)

    def animate(i):
        # Read shared state snapshot
        with state_lock:
            pts = list(path_positions)
            lr = None if latest_rvec is None else latest_rvec.copy()
            lt = None if latest_tvec is None else np.array(latest_tvec)

        # Update path
        if pts:
            xs, ys, zs = zip(*pts)
            # path_line.set_data(xs, ys)
            # auto-scale with padding
            ax.set_xlim(min(xs) - 1, max(xs) + 1)
            ax.set_ylim(min(ys) - 1, max(ys) + 1)

        # Update current point
        if lt is not None:
            x, y, z = lt
            point.set_data([x], [y])
        else:
            x = y = 0

        # Remove previous orientation lines
        for ln in orientation_lines:
            try:
                ln.remove()
            except Exception:
                pass
        orientation_lines.clear()

        # If we have rotation, draw fixed-length heading-only markers in XY
        if lr is not None and lt is not None:
            R, _ = cv2.Rodrigues(lr)
            yaw = np.arctan2(R[1, 0], R[0, 0])

            fx = arrow_length * np.cos(yaw)
            fy = arrow_length * np.sin(yaw)

            lx = arrow_length * np.cos(yaw + np.pi/2)
            ly = arrow_length * np.sin(yaw + np.pi/2)

            # forward line (red)
            ln_f, = ax.plot([x, x + fx], [y, y + fy], color='r', linewidth=2)
            orientation_lines.append(ln_f)

            # left line (green)
            ln_l, = ax.plot([x, x + lx], [y, y + ly], color='g', linewidth=2)
            orientation_lines.append(ln_l)

        # artists = (path_line, point, *orientation_lines)
        artists = (point, *orientation_lines)
        return artists

    ani = FuncAnimation(fig, animate, init_func=init, interval=plot_interval_ms, blit=False)
    ax.set_title('2D Trajectory & Orientation (live)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    plt.show()

# def run_visualization():
#     global latest_rvec, latest_tvec

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # initial empty plot elements
#     # path_line, = ax.plot([], [], [], color='k', linewidth=1)
#     point, = ax.plot([], [], [], 'ro')

#     # orientation lines (forward, left, up)
#     orientation_lines = []

#     def init():
#         # path_line.set_data([], [])
#         # path_line.set_3d_properties([])
#         point.set_data([], [])
#         point.set_3d_properties([])
#         return (point)

#     def animate(i):
#         # Read shared state snapshot
#         with state_lock:
#             pts = list(path_positions)
#             lr = None if latest_rvec is None else latest_rvec.copy()
#             lt = None if latest_tvec is None else np.array(latest_tvec)

#         # Update path
#         if pts:
#             xs, ys, zs = zip(*pts)
#             # path_line.set_data(xs, ys)
#             # path_line.set_3d_properties(zs)
#             ax.set_xlim(min(xs) - 1, max(xs) + 1)
#             ax.set_ylim(min(ys) - 1, max(ys) + 1)
#             ax.set_zlim(min(zs) - 1, max(zs) + 1)

#         # Update current point
#         if lt is not None:
#             x, y, z = lt
#             point.set_data([x], [y])
#             point.set_3d_properties([z])
#         else:
#             x = y = z = 0

#         # Remove previous orientation lines
#         while orientation_lines:
#             ln = orientation_lines.pop()
#             try:
#                 ln.remove()
#             except Exception:
#                 pass

#         # If we have rotation, draw fixed-length heading-only markers
#         if lr is not None:
#             R, _ = cv2.Rodrigues(lr)
#             yaw = np.arctan2(R[1, 0], R[0, 0])

#             fx = arrow_length * np.cos(yaw)
#             fy = arrow_length * np.sin(yaw)
#             fz = 0

#             lx = arrow_length * np.cos(yaw + np.pi/2)
#             ly = arrow_length * np.sin(yaw + np.pi/2)
#             lz = 0

#             ux, uy, uz = 0, 0, arrow_length

#             # forward
#             ln_f, = ax.plot([x, x+fx], [y, y+fy], [z, z+fz], color='r', linewidth=2)
#             orientation_lines.append(ln_f)
#             # left
#             ln_l, = ax.plot([x, x+lx], [y, y+ly], [z, z+lz], color='g', linewidth=2)
#             orientation_lines.append(ln_l)
#             # up
#             ln_u, = ax.plot([x, x+ux], [y, y+uy], [z, z+uz], color='b', linewidth=2)
#             orientation_lines.append(ln_u)

#         artists = (point, *orientation_lines)
#         return artists

#     ani = FuncAnimation(fig, animate, init_func=init, interval=plot_interval_ms, blit=False)
#     plt.show()


# -------------------- Main --------------------
if __name__ == '__main__':
    stop_event = threading.Event()
    detector_thread = threading.Thread(target=detector_thread_func, args=(stop_event,), daemon=True)
    detector_thread.start()

    try:
        run_visualization()
    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down...")
        stop_event.set()
        detector_thread.join(timeout=2)
        print("Exit")
