import cv2
import numpy as np
import os
import math
import time
from heapq import heappush, heappop
import threading

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
# ---- helpers ----
def pose_to_matrix(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

""" Class which contains map data and scale information"""
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

import cv2
import numpy as np

def visualize_path(map_obj, path, use_gradient=False, color=(0, 0, 255), thickness=2, robot_position = (0,0)):
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
    cv2.circle(base_bgr, path[0], radius=5, color=(255, 0, 0), thickness=-1)  # Start
    cv2.circle(base_bgr, path[-1], radius=5, color=(0, 0, 255), thickness=-1) # End

    cv2.circle(base_bgr, robot_position, radius=5, color=(255, 155, 0), thickness=-1)  # Start

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

    def pixel_to_world(self, pixel_coords):
        """Convert pixel coordinates to world coordinates (meters)."""
        return pixel_coords * self.scale

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

""" Class to update and store robot pose estimation"""
class Robot:
    def __init__(self, cam):
        self.position_world = (0,0)  # x, y in meters OF WORLD FRAME
        self.position_frame = (0,0)  # x, y in meters OF ROBOT FRAME
        self.heading = 0.0  # theta in radians
        self.ip_address = "192.168.0.101" # IP of the robot
        self.size = (0.75, 1) # in meters
        self.cam_index = cam
        self.camera_matrix = np.loadtxt(camera_mtx_path, delimiter=',')
        self.dist_coeffs = np.loadtxt(camera_dist_path, delimiter=',')
        self.image_points = np.load(image_points_path)    # Nx2 image points
        self.world_points = np.load(world_points_path)    # Nx2 world points (meters)
        self.H, status = cv2.findHomography(self.image_points, self.world_points, method=cv2.RANSAC)
        if self.H is None:
            raise RuntimeError("Homography failed. Check image/world correspondences.")
        self.Hw = np.linalg.inv(self.H)

        # try load camera extrinsics (optional)
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
                                self.position_world = (x, y)
                    except Exception:
                        pass

                    world_markers.append(marker_info)

            # ---- OpenCV overlay ----
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

            # show frame
            # cv2.imshow("ArUco realtime (undistorted)", vis)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break


        return None
    
    def followpath(self, path):
        # TODO: get robot position, compute angular velocity and linear velocity to follow path
        # Set linear and angular velocities
        return l_vel, a_vel

    def l_vel_pid(self):

        return None
    
    def a_vel_pid(self):
        return None
        
""" Uses map and AStar to create a path for the robot """
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

    

def main():

    robot = Robot(1)
    map_data = Map(map_path, map_scale_path)
    map_data.create_gradient_around_obstacles(map_path, radius = 2)

    astar = AStar(map_data)
    
    print("Starting robot pose thread...")
    update_Position = threading.Thread(target=robot.get_pose_continuous, args=())
    update_Position.daemon = True
    update_Position.start()

    print("Getting start position...")
    while robot.position_world is None or np.linalg.norm(robot.position_world) < 1e-6:
        print("Waiting for robot pose...")
        time.sleep(0.1)

    print(robot.position_world)
    wx,wy = map_data.world_to_pgm(robot.position_world[0], robot.position_world[1])
    start = (wx,wy)
    # print(start)
    goal = (300, 200)


    print("Starting A* pathfinding...")
    path = astar.find_path(start, goal)
    if(path is None):
        print("No path found.")
        exit(1)

    print("Path found!")



    # visualize_path(map_path, path, output_path=None)
    result = visualize_path(map_data, path, use_gradient=True, color=(0, 0, 255), thickness=2)

    # converting path points to world coordinates for robot navigation
    world_path = []

    robot.followpath(world_path)

    cv2.namedWindow('My Resizable Window', cv2.WINDOW_NORMAL)
    cv2.imshow('My Resizable Window', result)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        # print(robot.position)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    update_Position.join()


if __name__ == "__main__":
    main()