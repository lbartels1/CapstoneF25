import cv2
import numpy as np
import os
import math
import time
from heapq import heappush, heappop
# import socket
# import json
import threading

camera_mtx_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\camera_mtx.txt"
camera_dist_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\camera_dist.txt"
image_points_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\image_points.npy"
world_points_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\world_points.npy"
cam_rvec_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\camera_rvec.npy"
cam_tvec_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\camera_tvec.npy"

map_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Perimeter\map.pgm"
map_scale_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Perimeter\map_scale.txt"
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

def visualize_path(map_path, path, output_path=None):
    """Visualize the path on the map"""
    map_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    map_rgb = cv2.cvtColor(map_img, cv2.COLOR_GRAY2RGB)
    
    # Draw path
    if path:
        for i in range(len(path) - 1):
            pt1 = path[i]
            pt2 = path[i + 1]
            cv2.line(map_rgb, pt1, pt2, (0, 0, 255), 2)  # Red line
        
        # Mark start and goal
        cv2.circle(map_rgb, path[0], 5, (0, 255, 0), -1)     # Green start
        cv2.circle(map_rgb, path[-1], 5, (255, 0, 0), -1)    # Blue goal
    
    if output_path:
        cv2.imwrite(output_path, map_rgb)
    
    # Resize for display
    display_width = 400  
    scale = display_width / map_rgb.shape[1]
    display_height = int(map_rgb.shape[0] * scale)
    display_size = (display_width, display_height)
    display_img = cv2.resize(map_rgb, display_size, interpolation=cv2.INTER_AREA)
    
    return display_img

class Map:
    def __init__(self, map_path, map_scale_path):
        self.map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if self.map is None:
            raise FileNotFoundError(f"Map image not found: {map_path}")
        
        self.scale = parse_scale(map_scale_path, "pixels_per_meter")  # pixels per meter
        self.map_height = parse_scale(map_scale_path, "height_px")  # in meters
        self.map_width = parse_scale(map_scale_path, "width_px")    # in meters

    
    def pixel_to_world(self, pixel_coords):
        """ Convert pixel coordinates to world coordinates (meters) """
        return pixel_coords * self.scale
    
    def world_to_pixel(self, world_coords):
        """ Convert world coordinates (meters) to pixel coordinates """
        return (world_coords / self.scale).astype(float)
    
    def create_gradient_around_obstacles(self,map_input, radius=2.5, gradient_path="gradient.pgm"):
        import os
        import numpy as np
        import cv2

        radius_px = self.pixel_to_world(radius)

        # load image if a path was provided
        if isinstance(map_input, str):
            img = cv2.imread(map_input, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not load map from {map_input}")
            base_dir = os.path.dirname(map_input)
            base_name = os.path.splitext(os.path.basename(map_input))[0]
        elif isinstance(map_input, np.ndarray):
            if map_input.ndim == 3:
                img = cv2.cvtColor(map_input, cv2.COLOR_BGR2GRAY)
            else:
                img = map_input.copy()
            base_dir = os.getcwd()
            base_name = "map"
        else:
            raise ValueError("map_input must be a filepath or numpy ndarray")

        # ensure dtype
        img = img.astype(np.uint8)

        # foreground for distance transform: free=255, obstacle=0
        fg = (img > 127).astype(np.uint8) * 255

        # compute distance transform (float32)
        dist = cv2.distanceTransform(fg, distanceType=cv2.DIST_L2, maskSize=5)

        # clamp to radius and map to 0..255
        radius = float(max(1, radius_px))
        clipped = np.minimum(dist, radius)
        grad = (clipped / radius * 255.0).astype(np.uint8)

        # Obstacles remain 0; pixels farther than radius -> 255
        grad[fg == 0] = 0
        grad[dist >= radius] = 255

        # prepare nav-safe binary image (keep original navigation semantics)
        # nav = np.where(img > 127, 255, 0).astype(np.uint8)

        # default output paths
        if gradient_path is None:
            gradient_path = os.path.join(base_dir, f"{base_name}_gradient.pgm")
        # if nav_path is None:
        #     nav_path = os.path.join(base_dir, f"{base_name}_nav.pgm")

        # ensure directories exist
        gdir = os.path.dirname(gradient_path)
        # ndir = os.path.dirname(nav_path)
        if gdir and not os.path.exists(gdir):
            os.makedirs(gdir, exist_ok=True)
        # if ndir and not os.path.exists(ndir):
        #     os.makedirs(ndir, exist_ok=True)

        # save files
        if not cv2.imwrite(gradient_path, grad):
            raise IOError(f"Failed to write gradient file {gradient_path}")
        # if not cv2.imwrite(nav_path, nav):
        #     raise IOError(f"Failed to write nav-safe file {nav_path}")

        # return nav-safe path for navigation; gradient saved for visualization
        print(f"Saved gradient visualization: {gradient_path}")
        # print(f"Saved nav-safe PGM for navigation: {nav_path}")
        return gradient_path



""" Class to update and store robot pose estimation"""
class Robot:
    def __init__(self, cam):
        self.position = np.array([0.0, 0.0])  # x, y in meters OF WORLD FRAME
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
        """TODO: Import code from Track_homo that returns the robot's position and orientation 
            Consider making this its own thread and making the pose atomic. other option is not making it atomic and assuming it wont matter"""
        cap = cv2.VideoCapture(self.cam_index)
        # print("here")
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.cam_index}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # # undistort frame
            frame_und = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

            # # detect markers
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
                    except Exception:
                        pass

                    self.position = marker_info["tvec_world"]
                    self.heading = marker_info["yaw_world_from_pose"]

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
            cv2.imshow("ArUco realtime (undistorted)", vis)


        return None
    
    def path_to_velocities(self, point):
        l_vel = 0
        a_vel = 0
        return l_vel, a_vel

    def l_vel_pid(self):
        return None
    
    def a_vel_pid(self):
        return None
        


""" Uses map and AStar to navigate the robot """
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
    def __init__(self, map_path, scale_path=None, cost_weight=2.0):
        # Load PGM map
        self.map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if self.map is None:
            raise ValueError(f"Could not load map from {map_path}")
        
        # In PGM: 0 = obstacle (black), 255 = free space (white)
        self.height, self.width = self.map.shape
        
        # Load scale information if provided
        self.scale = 1.0  # pixels per meter
        if scale_path:
            self.load_scale(scale_path)
            
        # Define possible movements (8-directional)
        self.directions = [
            (-1,-1), (0,-1), (1,-1),
            (-1, 0),         (1, 0),
            (-1, 1), (0, 1), (1, 1)
        ]

        # cost scaling parameter:
        # cost_weight = 0 => no extra cost; larger -> more penalty near low-valued pixels
        self.cost_weight = float(cost_weight)
        
    def load_scale(self, scale_path):
        """Load scale information from metadata file"""
        with open(scale_path, 'r') as f:
            for line in f:
                if line.startswith('pixels_per_meter'):
                    self.scale = float(line.split(':')[1].strip())
                    break
    
    def is_valid(self, x, y):
        """Check if position is within bounds and not an obstacle"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.map[y, x] > 127  # Consider anything darker than mid-gray as obstacle
        return False
    
    def heuristic(self, pos1, pos2):
        """Calculate octile distance heuristic"""
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        return (max(dx, dy) * 1.0 + (math.sqrt(2) - 1.0) * min(dx, dy))
    
    def get_path(self, current):
        """Reconstruct path from goal to start"""
        path = []
        while current:
            path.append(current.position)
            current = current.parent
        return path[::-1]  # Reverse to get start-to-goal order
    
    def find_path(self, start, goal):
        """Find path from start to goal using A* algorithm; movement cost is scaled by pixel value."""
        if not (self.is_valid(*start) and self.is_valid(*goal)):
            return None
        
        start_node = Node(start, g_cost=0)
        start_node.h_cost = self.heuristic(start, goal)
        
        open_set = []
        heappush(open_set, start_node)
        closed_set = set()
        
        # track best g for positions
        best_g = {start: 0.0}
        
        while open_set:
            current = heappop(open_set)
            
            if current.position == goal:
                return self.get_path(current)
            
            if current.position in closed_set:
                continue
            closed_set.add(current.position)
            
            for dx, dy in self.directions:
                next_x = current.position[0] + dx
                next_y = current.position[1] + dy
                next_pos = (next_x, next_y)
                
                if not self.is_valid(next_x, next_y) or next_pos in closed_set:
                    continue
                
                # base movement cost: diagonal = sqrt(2), straight = 1
                base_cost = math.sqrt(2) if dx*dy != 0 else 1.0

                # pixel value at the neighbor (0..255)
                pv = int(self.map[next_y, next_x])

                # Per your request: pixel value influences movement cost.
                # - If pixel == 0 keep the usual movement cost (no extra penalty).
                # - Otherwise scale cost by multiplier in (1 .. 1+cost_weight], stronger penalty closer to obstacles.
                #   We map pixel value to distance-like factor: higher pixel -> farther from obstacle -> smaller penalty.
                #   multiplier = 1.0 + cost_weight * (1 - (pv / 255.0))
                if pv == 0:
                    multiplier = 1.0
                else:
                    multiplier = 1.0 + self.cost_weight * (1.0 - (pv / 255.0))

                movement_cost = base_cost * multiplier
                tentative_g = current.g_cost + movement_cost
                
                # Only push/update if we improved g-score for this neighbour
                if tentative_g < best_g.get(next_pos, float('inf')):
                    best_g[next_pos] = tentative_g
                    next_node = Node(next_pos)
                    next_node.g_cost = tentative_g
                    next_node.h_cost = self.heuristic(next_pos, goal)
                    next_node.parent = current
                    heappush(open_set, next_node)
        
        return None  # No path found
    

def main():

    robot = Robot(0)
    map = Map(map_path, map_scale_path)
    astar = AStar(map_path, map_scale_path)
    start = (800, 1500)  # (x, y) in pixels
    goal = (730, 80)   # (x, y) in pixels
        # robot.get_pose_continuous()
    
    update_Position = threading.Thread(target=robot.get_pose_continuous(), args=())
    update_Position.daemon = True
    update_Position.start()

    gradient_map = map.create_gradient_around_obstacles(map_path, radius = 2.5)

    print("Starting A* pathfinding...")
    path = astar.find_path(start, goal)
    print("Path found!")

    visualize_path(map_path, path, output_path=None)
    result = visualize_path(map_path, path, output_path = None)
    cv2.imshow("Path", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()