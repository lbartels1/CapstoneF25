import cv2
import math
import heapq
import numpy as np
import os

# ---- Cell class (same as original reference) ----
class Cell:
    def __init__(self):
        self.parent_i = 0
        self.parent_j = 0
        self.f = float('inf')
        self.g = float('inf')
        self.h = 0

# ---- helper checks will be set after loading map ----
ROW = 0
COL = 0
_grid = None  # internal grid (0=blocked,1=free)

def is_valid(row, col):
    return (row >= 0) and (row < ROW) and (col >= 0) and (col < COL)

def is_unblocked(grid, row, col):
    # grid is 2D array of 0/1
    return grid[row][col] == 1

def is_destination(row, col, dest):
    return row == dest[0] and col == dest[1]

def calculate_h_value(row, col, dest):
    # Euclidean heuristic (consistent for 8-neighborhood with unit/diag costs)
    return math.hypot(row - dest[0], col - dest[1])

def trace_path(cell_details, dest):
    # return path as list of (row,col) from source to dest
    path = []
    row = dest[0]
    col = dest[1]

    while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
        path.append((row, col))
        temp_row = cell_details[row][col].parent_i
        temp_col = cell_details[row][col].parent_j
        row = temp_row
        col = temp_col

    path.append((row, col))
    path.reverse()
    return path

def a_star_search(grid, src, dest):
    global ROW, COL
    ROW, COL = grid.shape
    if not is_valid(src[0], src[1]) or not is_valid(dest[0], dest[1]):
        print("Source or destination is invalid")
        return None

    if not is_unblocked(grid, src[0], src[1]) or not is_unblocked(grid, dest[0], dest[1]):
        print("Source or the destination is blocked")
        return None

    if is_destination(src[0], src[1], dest):
        print("We are already at the destination")
        return [src]

    closed_list = [[False for _ in range(COL)] for _ in range(ROW)]
    cell_details = [[Cell() for _ in range(COL)] for _ in range(ROW)]

    i = src[0]
    j = src[1]
    cell_details[i][j].f = 0
    cell_details[i][j].g = 0
    cell_details[i][j].h = 0
    cell_details[i][j].parent_i = i
    cell_details[i][j].parent_j = j

    open_list = []
    heapq.heappush(open_list, (0.0, i, j))

    found_dest = False
    # 8 directions
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                  (1, 1), (1, -1), (-1, 1), (-1, -1)]

    while len(open_list) > 0:
        p = heapq.heappop(open_list)
        i = p[1]
        j = p[2]
        if closed_list[i][j]:
            continue
        closed_list[i][j] = True

        for dir in directions:
            new_i = i + dir[0]
            new_j = j + dir[1]

            if is_valid(new_i, new_j) and is_unblocked(grid, new_i, new_j) and not closed_list[new_i][new_j]:
                if is_destination(new_i, new_j, dest):
                    cell_details[new_i][new_j].parent_i = i
                    cell_details[new_i][new_j].parent_j = j
                    found_dest = True
                    return trace_path(cell_details, dest)
                else:
                    # diagonal cost = sqrt(2), straight = 1.0
                    g_new = cell_details[i][j].g + (math.hypot(dir[0], dir[1]))
                    h_new = calculate_h_value(new_i, new_j, dest)
                    f_new = g_new + h_new

                    if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
                        heapq.heappush(open_list, (f_new, new_i, new_j))
                        cell_details[new_i][new_j].f = f_new
                        cell_details[new_i][new_j].g = g_new
                        cell_details[new_i][new_j].h = h_new
                        cell_details[new_i][new_j].parent_i = i
                        cell_details[new_i][new_j].parent_j = j

    if not found_dest:
        print("Failed to find the destination cell")
        return None

# ---- visualization helper (similar to A_star.py) ----
def visualize_path_on_map(map_path, path, output_path=None, display_width=400):
    img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load map image {map_path}")
    map_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if path:
        # path is list of (row,col); convert to (col,row) for drawing
        pts = [(int(c), int(r)) for (r, c) in path]
        for i in range(len(pts) - 1):
            cv2.line(map_rgb, pts[i], pts[i + 1], (0, 0, 255), 2)
        cv2.circle(map_rgb, pts[0], 4, (0, 255, 0), -1)   # start
        cv2.circle(map_rgb, pts[-1], 4, (255, 0, 0), -1)  # goal

    if output_path:
        cv2.imwrite(output_path, map_rgb)

    # resize for display
    scale = display_width / map_rgb.shape[1]
    display_height = int(map_rgb.shape[0] * scale)
    display_img = cv2.resize(map_rgb, (display_width, display_height), interpolation=cv2.INTER_AREA)
    return display_img

# ---- main: load PGM and run A* ----
def main():
    # paths (adjust as needed)
    map_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Perimeter\map.pgm"
    output_path = r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Navigation\path_result.png"

    # load pgm as grayscale
    img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Failed to load map:", map_path)
        return

    # convert to binary grid: free=1 if pixel > 127 (white), blocked=0 if dark/black
    grid = (img > 127).astype(np.uint8)

    # Example start/goal in pixel coordinates (x,y). Modify or set via user input.
    # These are (x, y) -> convert to (row, col) for algorithm
    start_px = (10, 10)   # (x, y)
    goal_px = (img.shape[1] - 10, img.shape[0] - 10)  # (x, y)

    src = (start_px[1], start_px[0])   # (row, col)
    dest = (goal_px[1], goal_px[0])    # (row, col)

    print(f"Map size: {img.shape[1]} x {img.shape[0]} (w x h)")
    print(f"Start (px): {start_px} -> src (row,col): {src}")
    print(f"Goal  (px): {goal_px} -> dest(row,col): {dest}")

    path = a_star_search(grid, src, dest)
    if path:
        print(f"Path found length: {len(path)}")
        disp = visualize_path_on_map(map_path, path, output_path=output_path, display_width=400)
        cv2.imshow("A* Path", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No path found")

if __name__ == "__main__":
    main()