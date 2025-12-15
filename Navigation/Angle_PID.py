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

def rad2deg(rad):
    """Convert radians to degrees (wrapped to [-180, 180])."""
    deg = rad * 180.0 / math.pi
    return deg

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
            print("SOCKET ERROR")
            return 0
    
    return 1

def pose_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

class EKF3:
    def __init__(self, dt=0.1):
        self.x = np.array([0.0, 0.0, 0.0])  # state: x, y, heading
        self.P = np.eye(3) * 0.1

        # Motion noise (tune as needed)
        self.Q = np.diag([0.02, 0.02, 0.01])

        # Measurement noise for x, y, heading
        self.R = np.diag([0.05, 0.05, 0.02])

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


class Robot:
    def __init__(self, cam):
        self.position_world = (0,0)  # x, y in meters OF WORLD FRAME
        self.position_frame = (0,0)  # x, y in meters OF ROBOT FRAME
        self.heading = 0.0  # theta in radians
        self.prev_heading = 0.0
        self.angle_pid = PIDController(Kp=1, Ki=0.0, Kd=0.0)
        self.ekf = EKF3()

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

# marker forward axis in world frame:
                                robot_fwd = Rw[:, 0]                 # first column

                                # heading:
                                yaw_world = math.atan2(robot_fwd[1], robot_fwd[0])

                                marker_info["yaw_world_from_pose"] = yaw_world

                                marker_info["tvec_world"] = t_world.flatten()
                                # marker_info["rvec_world"] = rvec_world.flatten()
                                marker_info["yaw_world_from_pose"] = yaw_world

                                x, y = marker_info["center_world"]
                                # Full measurement: x, y, heading
                                self.update_ekf_with_camera(x, y, yaw)
                                print(rad2deg_wrapped(yaw))
                                
                    except Exception:
                        pass

                    world_markers.append(marker_info)

            """ ---- OpenCV overlay ----"""
            vis = frame_und.copy()
            for m in world_markers:
                pts = m["img_corners"].astype(int)
                cv2.polylines(vis, [pts.reshape(-1,1,2)], isClosed=True, color=(0,255,0), thickness=2)
                img_center = cv2.perspectiveTransform(np.array([[m["center_world"]]], dtype=np.float32), self.Hw)[0,0]
                cv2.circle(vis, (int(img_center[0]), int(img_center[1])), 4, (0,0,255), -1)
                cv2.putText(vis, f"id:{m['id']}", (int(img_center[0])+5, int(img_center[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                tip_world = m["center_world"] + np.array([math.cos(m["yaw"]), math.sin(m["yaw"])]) * (self.marker_length * 0.75)
                tip_img = cv2.perspectiveTransform(np.array([[tip_world]], dtype=np.float32), self.Hw)[0,0]
                cv2.arrowedLine(vis, (int(img_center[0]), int(img_center[1])), (int(tip_img[0]), int(tip_img[1])), (0,0,255), 2, tipLength=0.2)

            """show frame"""
            cv2.imshow("ArUco realtime (undistorted)", vis)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break


        return None
    
    def followpath(self, path, goal,_navigator):
        # TODO: get robot position, compute angular velocity and linear velocity to follow path
        # Set linear and angular velocities
        navigator = _navigator
        while is_near(self.position_world, goal, 1) is False:
            print(self.position_world, goal)
            lvelo, avelo, carrot = navigator.compute_control(self.position_world, self.heading, path)
            # self.send_wheel_speeds(lvelo, avelo)
            
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
        self.left_wheel_speed = linear_wheel_speed + (-angular_wheel_speed * self.wheel_size/ 2)
        self.right_wheel_speed = linear_wheel_speed + (angular_wheel_speed * self.wheel_size/ 2)
        return self.left_wheel_speed, self.right_wheel_speed
    
    def send_wheel_speeds(self):
        # left_speed, right_speed = self.set_wheel_speeds(lvelo, avelo)
        while True:
            left = min(max(self.left_wheel_speed, -10), 10)
            right = min(max(self.right_wheel_speed, -10), 10)

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
                            "lights": "g"
                            })

            #TODO: get old code to send to the phidget
            sendToPhidget(self.ip_address, data)
            # print("Sent\nLeft: ", left, "Right: ", right)
            time.sleep(0.06)
            

# pid = PIDController(1, 0.25, 0.1)
robot = Robot(1)

atexit.register(shutdown)

print("Starting robot pose thread...")
update_Position = threading.Thread(target=robot.get_pose_continuous, args=())
update_Position.daemon = True
update_Position.start()

print("Starting robot data thread")
send_data = threading.Thread(target=robot.send_wheel_speeds, args=())
send_data.daemon = True
send_data.start()

lvelo = 0
for i in range(1000):
    key = cv2.waitKey(1)
    if key == 27 or key == ord('q'):
        break
    robot.set_wheel_speeds(lvelo, -5)
    # robot.send_wheel_speeds()
    time.sleep(0.06)