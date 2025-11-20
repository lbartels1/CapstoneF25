import cv2
import numpy as np
import os
import math
import time
from heapq import heappush, heappop
import threading
import socket
import json
import atexit

camera_mtx_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\camera_mtx.txt"
camera_dist_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\camera_dist.txt"
image_points_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\image_points.npy"
world_points_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\world_points.npy"
cam_rvec_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\camera_rvec.npy"
cam_tvec_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\camera_tvec.npy"

map_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Perimeter\map.pgm"
map_scale_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Perimeter\map_scale.txt"

height = 720
width = 1280
np.set_printoptions(legacy='1.25')

# ---- helpers ----
def rad2deg_wrapped(rad):
    """Convert radians to degrees (wrapped to [-180, 180])."""
    deg = rad * 180.0 / math.pi
    return deg

def angle_diff_deg(a, b):
    """
    Smallest signed difference between two angles in degrees.
    Returns a value in [-180, 180].
    """
    diff = (a - b + 180) % 360 - 180
    return diff

def shutdown(HOST = "192.168.0.103"):
    PORT = 7002 # TCP port on the Phidget 

    byteData = json.dumps({ "x": 0,
                        "circle": 0,
                        "square": 0,
                        "triangle": 0,
                        "share": 0,
                        "PS": 0,
                        "options": 0,
                        "left_stick_click": 0,
                        "right_stick_click": 0,
                        "L1": 0,
                        "R1": 0,
                        "up_arrow": 0,
                        "down_arrow": 0,
                        "left_arrow": 0,
                        "right_arrow": 0,
                        "touchpad": 0,

                        "left_joystick_x": 0,
                        "left_joystick_y": 0,
                        "right_joystick_x": 0,
                        "right_joystick_y": 0,

                        "left_trigger": 0,
                        "right_trigger": 0,

                        "control": "idle",
                        "lights": "r"
                        })

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        try:
            client.settimeout(1)
            client.connect((HOST, PORT))
            client.send(byteData)
        except:
            print("SOCKET ERROR")
            return 0
    
    return 1

def goal_reached(robot, HOST = "192.168.0.103"):
    PORT = 7002 # TCP port on the Phidget 
    robot.left_wheel_speed = 0
    robot.right_wheel_speed = 0
    byteData = json.dumps({ "x": 0,
                        "circle": 0,
                        "square": 0,
                        "triangle": 0,
                        "share": 0,
                        "PS": 0,
                        "options": 0,
                        "left_stick_click": 0,
                        "right_stick_click": 0,
                        "L1": 0,
                        "R1": 0,
                        "up_arrow": 0,
                        "down_arrow": 0,
                        "left_arrow": 0,
                        "right_arrow": 0,
                        "touchpad": 0,

                        "left_joystick_x": 0,
                        "left_joystick_y": 0,
                        "right_joystick_x": 0,
                        "right_joystick_y": 0,

                        "left_trigger": 0,
                        "right_trigger": 0,

                        "control": "idle",
                        "lights": "g"
                        })

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        try:
            client.settimeout(1)
            client.connect((HOST, PORT))
            client.send(byteData)
        except:
            print("SOCKET ERROR")
            return 0
    
    return 1

def get_distance(p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def is_near(pt1, pt2, threshold):
    if(get_distance(pt1, pt2)) <= threshold:
        return True
    return False

def pose_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

def parse_scale(scale_path=r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Perimeter\map_scale.txt", item='pixels_per_meter'):
    """Read pixels_per_meter from scale file (expects 'pixels_per_meter: <value>')."""
    if not os.path.exists(scale_path):
        raise FileNotFoundError(f"Scale file not found: {scale_path}")
    with open(scale_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(item):
                try:
                    return float(line.split(':', 1)[1].strip())
                except ValueError:
                    break
    raise ValueError(f"pixels_per_meter not found or invalid in {scale_path}")

def sendToPhidget(HOST, data): # IP address of the Phidget on XOVER
    PORT = 7002 # TCP port on the Phidget 

    if(HOST == ''):
        # print("INVALID VEHICLE INDEX")
        return

    byteData = data.encode()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        try:
            client.settimeout(1)
            client.connect((HOST, PORT))
            client.send(byteData)
        except:
            print("ERROR")
            return 0
    
    return 1

def visualize_path(map_obj, path, start, use_gradient=False, color=(0, 0, 255), thickness=2):
    """
    Visualize a path over the map or gradient map.

    Args:
        map_obj (Map): instance of Map containing .map and optionally .gradient_map
        path (list[tuple[int, int]]): list of (x, y) pixel coordinates
        use_gradient (bool): if True, visualize over the gradient map instead of raw map
        color (tuple[int, int, int]): BGR color for the path
        thickness (int): path line thickness

    Returns:
        np.ndarray: BGR image with path drawn
    # """
    if not hasattr(map_obj, "map"):
        raise TypeError("Expected a Map object with .map attribute")

    # Choose base image
    if use_gradient and getattr(map_obj, "gradient_map", None) is not None:
        base = map_obj.gradient_map.copy()
    else:
        base = map_obj.map.copy()

    # Convert grayscale to BGR for drawing colored lines
    if len(base.shape) == 2:
        base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    else:
        base_bgr = base.copy()

    # Draw path
    for i in range(1, len(path)):
        cv2.line(base_bgr, path[i - 1], path[i], color, thickness)
    
    # Draw robot position
    cv2.circle(base_bgr, start, radius=10, color=(0, 255, 0), thickness=-1)  # Start
    cv2.circle(base_bgr, path[-1], radius=10, color=(0, 0, 255), thickness=-1) # End

    # cv2.circle(base_bgr, robot_position, radius=5, color=(255, 155, 0), thickness=-1)  # Start

    return base_bgr

""" Class for all things map"""
class Map:
    def __init__(self, map_path, map_scale_path):
        self.map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if self.map is None:
            raise FileNotFoundError(f"Map image not found: {map_path}")
        
        self.params = {}
        with open(map_scale_path, "r") as f:
            for line in f:
                if ":" in line and not line.strip().startswith("#"):
                    key, val = line.strip().split(":")
                    self.params[key.strip()] = float(val.strip())

        self.gradient_map = None  # to be created later
        self.scale = self.params["pixels_per_meter"]
        self.map_width = self.map.shape[1]
        self.map_height = self.map.shape[0]

        print(f"[Map] Loaded scale: {self.scale} px/m, width: {self.map_width}px, height: {self.map_height}px")

    def path_to_world(self, path):
        new_path = []
        for point in path:
            new_point = self.pgm_to_world(point[0], point[1])
            new_path.append(new_point)
        return new_path

    def pgm_to_world(self, px, py):
        """
        Convert pixel coordinates (px, py) from a PGM map into world coordinates (xw, yw)
        in meters, using the metadata from the _scale.txt file.

        Args:
            px, py : int
                Pixel coordinates in the PGM image (origin at top-left, y downward).

        Returns:
            (xw, yw) : tuple of floats
                World coordinates in meters.
        """

        # Extract required parameters
        scale = self.params["pixels_per_meter"]
        min_x = self.params["min_x_m"]
        max_y = self.params["max_y_m"]  # for Y-axis inversion

        # Compute world coordinates
        xw = px / scale + min_x
        yw = max_y - (py / scale)

        return xw, yw


    def world_to_pgm(self, xw, yw):
        """
        Convert a world coordinate (xw, yw) in meters to pixel coordinates (px, py)
        in the saved PGM map using the metadata from the _scale.txt file.

        Args:
            xw, yw : float
                World coordinates in meters.
            scale_file : str
                Path to the *_scale.txt file created alongside the PGM.

        Returns:
            (px, py) : tuple of ints
                Pixel coordinates in the PGM image (origin at top-left, y downward).
        """

        # Extract required values
        scale = self.params["pixels_per_meter"]
        min_x = self.params["min_x_m"]
        max_y = self.params["max_y_m"]  # note: used for y inversion

        # Compute pixel coordinates
        px = int(round((xw - min_x) * scale))
        py = int(round((max_y - yw) * scale))

        return px, py


    def create_gradient_around_obstacles(self, map_input=None, radius=2.5, gradient_path=None):
        """
        Create a gradient map around obstacles and store it in self.gradient_map.
        
        Args:
            map_input (str | np.ndarray | None): map to use; defaults to self.map
            radius (float): distance (in meters) to compute gradient
            gradient_path (str | None): optional path to save gradient image
        Returns:
            np.ndarray: the gradient map as uint8 image (0–255)
        """
        import numpy as np, cv2, os

        if map_input is None:
            img = self.map.copy()
        elif isinstance(map_input, str):
            img = cv2.imread(map_input, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load map from {map_input}")
        elif isinstance(map_input, np.ndarray):
            img = map_input.copy()
        else:
            raise ValueError("map_input must be None, filepath, or numpy array")

        # Convert radius (m) to pixels
        radius_px = max(1, int(radius * self.scale))

        
        # Distance transform (white = free, black = obstacle)
        fg = (img > 127).astype(np.uint8) * 255
        dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)

        # Normalize and invert so near obstacles = dark, far = bright
        clipped = np.minimum(dist, radius_px)
        grad = (clipped / radius_px * 255.0).astype(np.uint8)
        grad[fg == 0] = 0  # ensure obstacles remain black

        # Save to file if requested
        if gradient_path:
            if not cv2.imwrite(gradient_path, grad):
                raise IOError(f"Failed to save gradient file: {gradient_path}")
            print(f"Saved gradient visualization: {gradient_path}")

        # Store in class instance
        self.gradient_map = grad
        return grad
    
    def get_goal_gui(self):
        goal = (0,0)
        return goal
"""Extended Three dimensional Kalman Filter, not correctly tuned as math for a slipping wheel robot is hard"""
class EKF3:
    def __init__(self, dt=0.1):
        self.x = np.array([0.0, 0.0, 0.0])  # state: x, y, heading
        self.P = np.eye(3) * 0.1

        # Motion noise (tune as needed)
        self.Q = np.diag([0.002, 0.002, 0.004])

        # Measurement noise for x, y, heading
        self.R = np.diag([0.005, 0.005, 0.005])

        self.dt = dt

    # --------------------------------------------------
    # Nonlinear motion model
    # --------------------------------------------------
    def f(self, x, u):
        v, w = u
        dt = self.dt
        th = x[2]

        return np.array([
            x[0] + v * np.cos(th) * dt,
            x[1] + v * np.sin(th) * dt,
            th + w * dt
        ])

    def F_jac(self, x, u):
        v, w = u
        dt = self.dt
        th = x[2]
        return np.array([
            [1, 0, -v * np.sin(th) * dt],
            [0, 1,  v * np.cos(th) * dt],
            [0, 0, 1]
        ])

    # --------------------------------------------------
    # Measurement model: z = [x, y, heading]
    # --------------------------------------------------
    def h(self, x):
        return x.copy()

    def H_jac(self, x):
        return np.eye(3)

    # --------------------------------------------------
    # Prediction step
    # --------------------------------------------------
    def predict(self, v, w):
        u = np.array([v, w])
        self.x = self.f(self.x, u)
        F = self.F_jac(self.x, u)
        self.P = F @ self.P @ F.T + self.Q
        self.x[2] = self.normalize_angle(self.x[2])

    # --------------------------------------------------
    # Update step
    # --------------------------------------------------
    def update(self, z):
        z = np.array(z)

        z_pred = self.h(self.x)
        H = self.H_jac(self.x)

        # Innovation
        y = z - z_pred
        y[2] = self.normalize_angle(y[2])   # fix angle wrap

        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.x[2] = self.normalize_angle(self.x[2])

        self.P = (np.eye(3) - K @ H) @ self.P

    @staticmethod
    def normalize_angle(a):
        return (a + np.pi) % (2*np.pi) - np.pi
"""Class for PID controller"""
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.prev_error = 0.0

    def angle_wrap(self, angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def update(self, current_heading, previous_heading, dt, target_angular_velocity):
        # Compute measured angular velocity
        delta_heading = self.angle_wrap(current_heading - previous_heading)
        # print(delta_heading)
        if(dt != 0):
            measured_angular_velocity = delta_heading / dt
        else:
            print("DT: ERROR")
            measured_angular_velocity = target_angular_velocity

        # Compute error
        error = target_angular_velocity - measured_angular_velocity
        # print(error)
        # if (measured_angular_velocity <= 0.1+target_angular_velocity and measured_angular_velocity >= target_angular_velocity - 0.1 ):
        # PID terms
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0

        # PID output
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        # Store error for next iteration
        self.prev_error = error

        return output, measured_angular_velocity
"""Navigator class: Basic follow the carrot"""
class FollowTheCarrot:
    def __init__(self, lookahead_distance=1, Kp=1.0):
        """
        lookahead_distance: how far ahead the carrot point should be (in meters)
        Kp: proportional gain for angular velocity control
        """
        self.lookahead_distance = lookahead_distance
        self.Kp = Kp
        print("Follow the Carror kP = ", self.Kp)

    def find_carrot_point(self, robot_pos, path):
        """
        Find the carrot (lookahead) point along the path.
        """
        if not path:
            return None
        
        # Find the closest point on the path to the robot
        dists = [get_distance(robot_pos, p) for p in path]
        closest_idx = min(range(len(dists)), key=dists.__getitem__)

        # Move forward along the path until the lookahead distance is reached
        for i in range(closest_idx, len(path) - 1):
            segment_dist = get_distance(robot_pos, path[i])
            if segment_dist >= self.lookahead_distance:
                return path[i]
        
        # If end of path reached
        return path[-1]

    def compute_control(self, robot_pos, robot_heading, path, base_speed):

        carrot = self.find_carrot_point(robot_pos, path)
        if carrot is None:
            return 0.0, 0.0  # no movement if no path

        # Compute angle to carrot
        dx = carrot[0] - robot_pos[0]
        dy = carrot[1] - robot_pos[1]

        # print("Robot: ", robot_pos, "Carrot:", carrot)

        target_heading = rad2deg_wrapped(math.atan2(dy, dx))

        robot_heading = rad2deg_wrapped(robot_heading)

        heading_error = angle_diff_deg(robot_heading, target_heading)

        print(heading_error)

        # Angular velocity control (proportional)

        if (heading_error < 25 and heading_error > -25 ):
            linear_speed = (base_speed) #greater the heading error, the slower the base turns
            angular_velocity = self.Kp * heading_error * 1.5
            # heading_error = 0
        else:
            # print("Just turning")
            linear_speed = 0
            angular_velocity = self.Kp * heading_error

        # print(self.Kp)
        # print(angular_velocity, linear_speed)
        # Return control signals
        time.sleep(0.1)
        return linear_speed, angular_velocity, carrot
    

""" Class to update and store robot pose estimation"""
class Robot:
    def __init__(self, cam, speed, tolerance, Kp, Ki, Kd):
        self.position_world = (0,0)  # x, y in meters OF WORLD FRAME
        self.position_frame = (0,0)  # x, y in meters OF ROBOT FRAME
        self.heading = 0.0  # theta in radians
        self.prev_heading = 0.0
        self.angle_pid = PIDController(Kp, Ki, Kd)
        self.dt = time.time()
        self.ekf = EKF3()
        self.tolerance = tolerance
        self.speed = speed
        self.last_reading = time.time()
        self.lad = 0.5 # look ahead distance

        self.ip_address = "192.168.0.103" # IP of the robot
        self.size = (0.75, 1) # in meters
        self.wheel_size = 0.2 # in meters
        self.left_wheel_speed = 0.0 #between -10 and 10q
        self.right_wheel_speed = 0.0 #between -10 and 10

        self.cam_index = cam
        self.camera_matrix = np.loadtxt(camera_mtx_path, delimiter=',')
        self.dist_coeffs = np.loadtxt(camera_dist_path, delimiter=',')
        self.image_points = np.load(image_points_path)    # Nx2 image points
        self.world_points = np.load(world_points_path)    # Nx2 world points (meters)
        self.H, status = cv2.findHomography(self.image_points, self.world_points, method=cv2.RANSAC)
        if self.H is None:
            raise RuntimeError("Homography failed. Check image/world correspondences.")
        self.Hw = np.linalg.inv(self.H)

        # try load camera extriqnsics (optional)
        have_cam_extrinsics = os.path.exists(cam_rvec_path) and os.path.exists(cam_tvec_path)
        if have_cam_extrinsics:
            self.cam_rvec = np.load(cam_rvec_path)
            self.cam_tvec = np.load(cam_tvec_path)
            self.cam_T = pose_to_matrix(self.cam_rvec, self.cam_tvec)
        else:
            self.cam_rvec = self.cam_tvec = None
            self.cam_T = None

        #Aruco setup
        self.aruco = cv2.aruco
        self.dictionary = self.aruco.getPredefinedDictionary(self.aruco.DICT_4X4_50)
        self.params = self.aruco.DetectorParameters()
        self.detector = self.aruco.ArucoDetector(self.dictionary, self.params)
        self.marker_length = 0.47625  # meters

        self.objPoints = np.array([
            [-self.marker_length / 2,  self.marker_length / 2, 0],
            [ self.marker_length / 2,  self.marker_length / 2, 0],
            [ self.marker_length / 2, -self.marker_length / 2, 0],
            [-self.marker_length / 2, -self.marker_length / 2, 0]
        ], dtype=np.float32).reshape((4, 1, 3))

    def update_ekf_with_camera(self, x, y, yaw):
    # Full measurement: x, y, heading
        self.ekf.update((x, y, yaw))

        # Update robot pose from EKF output
        self.position_world = (self.ekf.x[0], self.ekf.x[1])
        self.heading = self.ekf.x[2]

    def update_ekf_prediction(self):
        v, w = self.compute_odometry()
        self.ekf.predict(v, w)
        self.position_world = (self.ekf.x[0], self.ekf.x[1])
        self.heading = self.ekf.x[2]

    def compute_odometry(self):
        # Differential drive kinematics
        L = self.size[0]          # wheel separation
        r = self.wheel_size / 2   # wheel radius

        v_l = self.left_wheel_speed  * r
        v_r = self.right_wheel_speed * r

        v = (v_r + v_l) / 2
        w = (v_r - v_l) / L

        return v, w

    
    def get_pose_continuous(self):
        i = 0
        """TODO: Import code from Track_homo that returns the robot's position_world and orientation 
            Consider making this its own thread and making the pose atomic. other option is not making it atomic and assuming it wont matter"""
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # print("here")
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.cam_index}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # frame_idx += 1

            self.update_ekf_prediction()

            # undistort frame
            frame_und = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

            # detect markers
            corners, ids, rejected = self.detector.detectMarkers(frame_und)

            world_markers = []
            if ids is not None and len(ids) > 0:
                for i, c in enumerate(corners):
                    id_val = int(ids[i])
                    img_corners = c.reshape(4, 2).astype(np.float32)  # tl,tr,br,bl

                    # map corners to world plane via homography
                    world_corners = cv2.perspectiveTransform(np.array([img_corners]), self.H)[0]  # (4,2)
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
                        retval, rvec, tvec = cv2.solvePnP(self.objPoints, img_corners.reshape(4,1,2), self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                        if retval:
                            marker_info["rvec_cam"] = rvec
                            marker_info["tvec_cam"] = tvec
                            if self.cam_T is not None:
                                marker_T_cam = pose_to_matrix(rvec, tvec)
                                marker_T_world = self.cam_T @ marker_T_cam
                                t_world = marker_T_world[:3, 3]
                                Rw = marker_T_world[:3, :3]
                                rvec_world, _ = cv2.Rodrigues(Rw)
                                yaw_world = math.atan2(Rw[1, 0], Rw[0, 0])
                                marker_info["tvec_world"] = t_world.flatten()
                                marker_info["rvec_world"] = rvec_world.flatten()
                                marker_info["yaw_world_from_pose"] = yaw_world

                                x, y = marker_info["center_world"]
                                # Full measurement: x, y, heading
                                self.update_ekf_with_camera(x, y, yaw)
                                # self.position_world = (x,y)
                                # self.heading = yaw
                                
                    except Exception:
                        pass

                    world_markers.append(marker_info)
                    i = 0
                    
            else:
                i += 1
                if (i >= 20):
                    print("NO ROBOT")
                    shutdown()
                    os._exit(1)
            """ ---- OpenCV overlay ----"""
            # vis = frame_und.copy()
            # for m in world_markers:
            #     pts = m["img_corners"].astype(int)
            #     cv2.polylines(vis, [pts.reshape(-1,1,2)], isClosed=True, color=(0,255,0), thickness=2)
            #     img_center = cv2.perspectiveTransform(np.array([[m["center_world"]]], dtype=np.float32), self.Hw)[0,0]
            #     cv2.circle(vis, (int(img_center[0]), int(img_center[1])), 4, (0,0,255), -1)
            #     cv2.putText(vis, f"id:{m['id']}", (int(img_center[0])+5, int(img_center[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            #     tip_world = m["center_world"] + np.array([math.cos(m["yaw"]), math.sin(m["yaw"])]) * (self.marker_length * 0.75)
            #     tip_img = cv2.perspectiveTransform(np.array([[tip_world]], dtype=np.float32), self.Hw)[0,0]
            #     cv2.arrowedLine(vis, (int(img_center[0]), int(img_center[1])), (int(tip_img[0]), int(tip_img[1])), (0,0,255), 2, tipLength=0.2)

            """show frame"""
            # cv2.imshow("ArUco realtime (undistorted)", vis)
            # key = cv2.waitKey(1)
            # if key == 27 or key == ord('q'):
                # break

        return None    
    def followpath(self, path, goal,_navigator):
        # TODO: get robot position, compute angular velocity and linear velocity to follow path
        # Set linear and angular velocities
        navigator = _navigator
        self.dt = time.time()
        while is_near(self.position_world, goal, self.tolerance) is False:
            # print(self.position_world, goal)
            lvelo, avelo, carrot = navigator.compute_control(self.position_world, self.heading, path, self.speed)
            # print(lvelo)
            self.set_wheel_speeds(lvelo, avelo)
        
        print("I MADE IT TO: ", self.position_world)
        # print(self.position_world)
        self.set_wheel_speeds(0,0)
        return True


    def set_linear_velo(self):

        return None
    
    def set_angular_velo(self, angular_velocity):
        # print(time.time()  - self.last_reading)
        output, curr_angluar_velo = self.angle_pid.update(self.heading, self.prev_heading, time.time() - self.last_reading, angular_velocity)
        return output
    
    def set_wheel_speeds(self,lvelo, avelo):
        angular_wheel_speed = self.set_angular_velo(avelo)
        linear_wheel_speed = lvelo  # some constant forward speed
        # print(lvelo)
        self.left_wheel_speed = linear_wheel_speed + (-angular_wheel_speed * self.wheel_size/ 2)
        self.right_wheel_speed = linear_wheel_speed + (angular_wheel_speed * self.wheel_size/ 2)
        return self.left_wheel_speed, self.right_wheel_speed
    
    def send_wheel_speeds(self, stop_event):
        # left_speed, right_speed = self.set_wheel_speeds(lvelo, avelo)
        while not stop_event.is_set():
            left = min(max(self.left_wheel_speed, -2), 2)
            right = min(max(self.right_wheel_speed, -2), 2)

            data = json.dumps({ "x": 0,
                            "circle": 0,
                            "square": 0,
                            "triangle": 0,
                            "share": 0,
                            "PS": 0,
                            "options": 0,
                            "left_stick_click": 0,
                            "right_stick_click": 0,
                            "L1": 0,
                            "R1": 0,
                            "up_arrow": 0,
                            "down_arrow": 0,
                            "left_arrow": 0,
                            "right_arrow": 0,
                            "touchpad": 0,

                            "left_joystick_x": 0,
                            "left_joystick_y": 0,
                            "right_joystick_y": left,
                            "right_joystick_x": right,

                            "left_trigger": 0,
                            "right_trigger": 0,

                            "control": "auton",
                            "lights": "y"
                            })

            #TODO: get old code to send to the phidget
            sendToPhidget(self.ip_address, data)
            # print("Sent\nLeft: ", left, "Right: ", right)
            time.sleep(0.15)
""" Helper Class for Astar"""
class Node:
    def __init__(self, position, g_cost=float('inf'), h_cost=0):
        self.position = position  # (x, y) tuple
        self.g_cost = g_cost    # cost from start to current
        self.h_cost = h_cost    # heuristic estimate to goal
        self.parent = None      # parent node for path reconstruction
        
    def f_cost(self):
        return self.g_cost + self.h_cost
    
    def __lt__(self, other):
        return self.f_cost() < other.f_cost()
"""Uses Map class to create a path using A star"""
class AStar:
    def __init__(self, map_obj, cost_weight=2.0):
        """
        A* pathfinding using an existing Map instance.

        Args:
            map_obj (Map): an instance of your Map class
            cost_weight (float): how strongly obstacle proximity affects cost
        """
        if not hasattr(map_obj, "map"):
            raise TypeError("AStar requires a valid Map instance with .map attribute")

        # Use map data directly
        self.map = map_obj.gradient_map if map_obj.gradient_map is not None else map_obj.map
        self.scale = map_obj.scale
        self.height, self.width = self.map.shape

        # Movement model (8-connected grid)
        self.directions = [
            (-1, -1), (0, -1), (1, -1),
            (-1,  0),          (1,  0),
            (-1,  1), (0,  1), (1,  1)
        ]

        self.cost_weight = float(cost_weight)

    def is_valid(self, x, y):
        """Check if position is within bounds and not an obstacle"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.map[y, x] > 0  # Consider anything darker than mid-gray as obstacle
        return False


    def heuristic(self, pos1, pos2):
        """Octile distance heuristic for 8-directional movement"""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)

    def get_path(self, node):
        """Reconstruct the final path from a goal node"""
        path = []
        while node:
            path.append(node.position)
            node = node.parent
        return path[::-1]

    def find_path(self, start, goal):
        """Run A* from start→goal in pixel space"""
        print(f"A* searching from {start} to {goal}...")
        if not (self.is_valid(*start) and self.is_valid(*goal)):
            print("Invalid start or goal point.")
            return None

        start_node = Node(start, g_cost=0)
        start_node.h_cost = self.heuristic(start, goal)
        open_set = [start_node]
        closed_set = set()
        best_g = {start: 0.0}

        while open_set:
            current = heappop(open_set)
            if current.position == goal:
                return self.get_path(current)

            if current.position in closed_set:
                continue
            closed_set.add(current.position)

            for dx, dy in self.directions:
                nx, ny = current.position[0] + dx, current.position[1] + dy
                if not self.is_valid(nx, ny) or (nx, ny) in closed_set:
                    continue

                base_cost = math.sqrt(2) if dx * dy != 0 else 1.0
                pv = int(self.map[ny, nx])
                multiplier = 1.0 if pv == 0 else 1.0 + self.cost_weight * (1.0 - pv / 255.0)
                movement_cost = base_cost * multiplier
                tentative_g = current.g_cost + movement_cost

                if tentative_g < best_g.get((nx, ny), float('inf')):
                    best_g[(nx, ny)] = tentative_g
                    next_node = Node((nx, ny), tentative_g, self.heuristic((nx, ny), goal))
                    next_node.parent = current
                    heappush(open_set, next_node)

        return None


    def smooth_path(self, path, window_size=3):
        """
        Smooth a path using a simple moving average filter.
        path: list of (x, y) waypoints
        window_size: number of neighboring points to average over
        """

        if window_size < 2:
            return path  # no smoothing needed

        smoothed = []
        half_window = window_size // 2
        n = len(path)
        
        for i in range(n):
            # compute window bounds
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)
            
            # average over the window
            window_points = np.array(path[start:end])
            smoothed_point = (np.mean(window_points, axis=0))
            smoothed_point_int = tuple(map(int, smoothed_point))
            smoothed.append(tuple(smoothed_point_int))
        
        return smoothed


"""Main control loop
    Step 1: Initialize and find location of robot (program)
    Step 2: Input goal location (either x,y) of world or from a gui (user)
    Step 3: Create a path (program)
    Step 4: Accept path (user)
    Step 5: Start Navigation (user)
    Step 6 repeat at Step 2 (program)
"""

def main():

    # path_buffer = 50
    stop_event = threading.Event()
    atexit.register(shutdown)
    robot = Robot(1, 1.5, .5, 0.7, 0.00015, 0.1)
    map_data = Map(map_path, map_scale_path)
    map_data.create_gradient_around_obstacles(map_path, radius = 2)

    astar = AStar(map_data)

    navigator = FollowTheCarrot(1, .75)
    
    print("Starting robot pose thread...")
    update_Position = threading.Thread(target=robot.get_pose_continuous, args=())
    update_Position.daemon = True
    update_Position.start()

    goals = [(6, 3.5), (3,3), (2,8)]

    print(goals)
    for goal in goals:
        print("Getting start position...")
        print("Waiting for robot pose...")
        while robot.position_world is None or np.linalg.norm(robot.position_world) < 1e-6:
            time.sleep(0.1)

        # print(robot.position_world)
        wx,wy = map_data.world_to_pgm(robot.position_world[0], robot.position_world[1])
        start = (wx,wy)
        # start = (800, 200)

        gx,gy = map_data.world_to_pgm(goal[0], goal[1])
        goal = (gx,gy)

        # start_world = map_data.pgm_to_world(start[0], start[1])
        goal_world = map_data.pgm_to_world(goal[0], goal[1])

        print("Robot Position (world): ", robot.position_world)
        # print("Start (world): ", start_world)
        print("Goal (world): ", goal_world)

        print("Starting A* pathfinding...")
        path = astar.find_path(start, goal)
        smooth_path = astar.smooth_path(path, 80)
        if(path is None):
            print("No path found.")
            exit(1)

        result = visualize_path(map_data, smooth_path, start, use_gradient=False, color=(0, 0, 255), thickness=2)
        mirror = cv2.flip(result, 1)
        cv2.namedWindow('My Resizable Window', cv2.WINDOW_NORMAL)
        cv2.imshow('My Resizable Window', mirror)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
        cv2.destroyAllWindows()

        """converting path points to world coordinates for robot navigation"""
        world_path = map_data.path_to_world(smooth_path)
        print("First point: ", world_path[0])    

        print("Starting robot data thread")
        send_data = threading.Thread(target=robot.send_wheel_speeds, args=(stop_event,))
        send_data.daemon = True
        send_data.start()

        robot.followpath(world_path, goal_world, navigator)

        # stop_event.set()
        # send_data.join()
        goal_reached(robot)
        
        # update_Position.join()
        # print(world_path)

    os._exit(0)
    update_Position.join()

if __name__ == "__main__":
    main()