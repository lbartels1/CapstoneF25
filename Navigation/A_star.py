import numpy as np
import cv2
from heapq import heappush, heappop
import math

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

def create_gradient_around_obstacles(map_input, radius_px=50, gradient_path=None):
    """
    Create a visual gradient PGM around obstacles while preserving a nav-safe binary PGM.
    - map_input: filepath to PGM or a numpy grayscale image
    - radius_px: gradient radius in pixels (gradient goes 0->255 over radius)
    - gradient_path: optional filepath to save gradient visualization (default: <map>_gradient.pgm)
    - nav_path: optional filepath to save nav-safe binary PGM (default: <map>_nav.pgm)
    Returns: path to the nav-safe PGM (string). Gradient file is saved alongside (path returned via print).
    NOTE: This function does not change the binary occupancy semantics used by navigation.
    """
    import os
    import numpy as np
    import cv2

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

def main():
    # Example usage
    map_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Perimeter\map.pgm"
    scale_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Perimeter\map_scale.txt"
    output_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Navigation\path_result.png"
    # Create A* pathfinder
    gradient_map = create_gradient_around_obstacles(map_path, radius_px=50)  # Optional: create gradient visualization
    astar = AStar(gradient_map, scale_path)
    
    print("Starting A* pathfinding...")
    # Example coordinates (replace with actual start/goal)
    start = (170, 1700)  # (x, y) in pixels
    goal = (190, 1500)   # (x, y) in pixels
    
    # Find path
    path = astar.find_path(start, goal)
    print("FINISHED A* pathfinding...")
    
    if path:
        print(f"Path found with {len(path)} points")
        # Visualize and save result
        result = visualize_path(gradient_map, path, output_path)
        cv2.imshow("Path", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No path found")

if __name__ == "__main__":
    main()