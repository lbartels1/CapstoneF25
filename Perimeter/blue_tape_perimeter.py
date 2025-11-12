import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load camera calibration data
camera_matrix = np.loadtxt(r'C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\camera_mtx.txt', delimiter=',')
dist_coeffs = np.loadtxt(r'C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\camera_dist.txt', delimiter=',')

# Define blue color range in HSV
LOWER_BLUE = np.array([80, 45, 65])
UPPER_BLUE = np.array([119, 188, 255])

height = 720
width = 1280

# ArUco dictionary and marker size (in meters)
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
MARKER_LENGTH = 0.05  # 5 cm

def detect_blue_tape(frame):
    # brighten = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
    blur = cv2.blur(frame, (5,5))
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, mask

def compute_real_world_length(cnt_world):
    # cnt_world: Nx2 array of points already in world coordinates (meters)
    max_length = 0.0
    for i in range(len(cnt_world)):
        for j in range(i + 1, len(cnt_world)):
            dist = np.linalg.norm(cnt_world[i] - cnt_world[j])
            if dist > max_length:
                max_length = dist
    return max_length

def contour_to_world_segments(cnt, H, min_area=500, approx_eps=0.01):
    """
    Convert an image contour to a list of line segments in world coords using homography H.
    Returns list of ((x1,y1),(x2,y2),length).
    """
    if cv2.contourArea(cnt) < min_area:
        return []

    pts = cnt.reshape(-1, 2).astype(np.float32)
    pts_world = cv2.perspectiveTransform(np.array([pts]), H)[0]  # shape (N,2)

    # Approximate contour in world coords to reduce points and form straight segments
    pts_world_for_approx = pts_world.reshape(-1, 1, 2).astype(np.float32)
    # epsilon chosen relative to contour perimeter in world units
    peri = cv2.arcLength(pts_world_for_approx, True)
    eps = max(approx_eps, 0.001 * peri)
    approx = cv2.approxPolyDP(pts_world_for_approx, eps, True).reshape(-1, 2)

    # Build segments between consecutive approx points
    segments = []
    if len(approx) >= 2:
        for i in range(len(approx)):
            p1 = approx[i]
            p2 = approx[(i + 1) % len(approx)]
            length = np.linalg.norm(p2 - p1)
            # discard degenerate tiny segments
            if length > 1e-4:
                segments.append(((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])), float(length)))
    return segments

def save_segments_to_pgm(segments, filename, max_size_px=2000, line_thickness_px=2, pad_m=0.1, scale_filename=None):
    """
    Save world-space line segments to a grayscale PGM image and write scale/meta to a text file.
    Now draws white background with black lines (inverted from original).
    """
    if not segments:
        raise ValueError("No segments to save.")

    # compute bounds
    xs = [p for seg in segments for p in (seg[0][0], seg[1][0])]
    ys = [p for seg in segments for p in (seg[0][1], seg[1][1])]
    min_x, max_x = min(xs) - pad_m, max(xs) + pad_m
    min_y, max_y = min(ys) - pad_m, max(ys) + pad_m

    width_m = max_x - min_x
    height_m = max_y - min_y
    if width_m <= 0 or height_m <= 0:
        raise ValueError("Invalid world extents.")

    # choose scale so largest side maps to max_size_px
    scale = max_size_px / max(width_m, height_m)
    W = max(1, int(np.ceil(width_m * scale)))
    H = max(1, int(np.ceil(height_m * scale)))

    # create grayscale canvas (255=white background now)
    canvas = np.full((H, W), 255, dtype=np.uint8)  # Changed from zeros to full(255)

    # helper to map world -> pixel (origin at top-left)
    def world_to_px(xw, yw):
        px = int(round((xw - min_x) * scale))
        py = int(round((max_y - yw) * scale))  # invert y so world+Y is up
        return px, py

    for a, b, _ in segments:
        x1, y1 = world_to_px(a[0], a[1])
        x2, y2 = world_to_px(b[0], b[1])
        cv2.line(canvas, (x1, y1), (x2, y2), color=0, thickness=line_thickness_px, lineType=cv2.LINE_AA)  # Changed color to 0 (black)

    # ensure output directory exists
    out_dir = os.path.dirname(filename)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # write as PGM
    ok = cv2.imwrite(filename, canvas)
    if not ok:
        raise IOError(f"Failed to write {filename}")

    # write scale and metadata to text file
    if scale_filename is None:
        scale_filename = os.path.splitext(filename)[0] + "_scale.txt"
    scale_dir = os.path.dirname(scale_filename)
    if scale_dir and not os.path.exists(scale_dir):
        os.makedirs(scale_dir, exist_ok=True)

    with open(scale_filename, "w") as f:
        f.write(f"# Scale and canvas metadata\n")
        f.write(f"pixels_per_meter: {scale}\n")
        f.write(f"width_px: {W}\n")
        f.write(f"height_px: {H}\n")
        f.write(f"min_x_m: {min_x}\n")
        f.write(f"max_x_m: {max_x}\n")
        f.write(f"min_y_m: {min_y}\n")
        f.write(f"max_y_m: {max_y}\n")
        f.write(f"pad_m: {pad_m}\n")

    return filename

def main_single_image(image_source=r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Perimeter\taped_floor.jpg", preview=True, save_plot=False, plot_filename="tape_world.png", save_pgm=True, pgm_filename="map.pgm"):
    """
    Capture a single image (or load from file) and run the blue-tape -> world-segments pipeline once.
    image_source: int camera index (default 0) or path to image file.
    preview: show OpenCV preview windows (mask + overlay)
    save_plot: save the matplotlib world plot to plot_filename
    """

    print("Getting image from camera ", image_source)
    # Acquire single frame
    if isinstance(image_source, int):
        cap = cv2.VideoCapture(image_source, cv2.CAP_MSMF)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print("Failed to capture image from camera.")
            return
    else:
        frame = cv2.imread(image_source)
        if frame is None:
            print(f"Failed to read image from {image_source}")
            return
    
    print("Image acquired, processing...")
    # frame = cv2.imread(r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Perimeter\taped_floor.jpg")
    # Load calibration / homography points (same paths used previously)
    image_points = np.load(r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\image_points.npy")
    world_points = np.load(r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\world_points.npy")

    H, status = cv2.findHomography(image_points, world_points, method=cv2.RANSAC)
    if H is None:
        print("Homography computation failed. Check calibration points.")
        return

    # Undistort once
    frame_und = cv2.undistort(frame, camera_matrix, dist_coeffs)

    contours, mask = detect_blue_tape(frame_und)

    all_segments = []
    vis = frame_und.copy()
    Hw = np.linalg.inv(H)

    print("Processing contours...")
    for cnt in contours:
        if cv2.contourArea(cnt) < 500:
            continue
        cv2.drawContours(vis, [cnt], -1, (0, 255, 0), 2)
        segments = contour_to_world_segments(cnt, H, min_area=500, approx_eps=0.01)
        for seg in segments:
            all_segments.append(seg)
            # draw projected line on preview
            p1_world = np.array([[seg[0]]], dtype=np.float32)
            p2_world = np.array([[seg[1]]], dtype=np.float32)
            p1_img = cv2.perspectiveTransform(p1_world, Hw)[0][0].astype(int)
            p2_img = cv2.perspectiveTransform(p2_world, Hw)[0][0].astype(int)
            cv2.line(vis, tuple(p1_img), tuple(p2_img), (255, 0, 0), 3)
            # annotate length in image midpoint
            mid_w = np.array([[(seg[0][0] + seg[1][0]) / 2.0, (seg[0][1] + seg[1][1]) / 2.0]], dtype=np.float32)
            mid_img = cv2.perspectiveTransform(np.array([mid_w]), Hw)[0][0]
            cv2.putText(vis, f"{seg[2]:.2f}m", (int(mid_img[0]), int(mid_img[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    if preview:
        cv2.imshow("Frame (single)", vis)
        cv2.imshow("Mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Post-process segments: remove near-duplicates
    uniq = {}
    tol = 1e-3
    for a, b, L in all_segments:
        if (b[0], b[1]) < (a[0], a[1]):
            a, b = b, a
        key = (round(a[0]/tol)*tol, round(a[1]/tol)*tol, round(b[0]/tol)*tol, round(b[1]/tol)*tol)
        if key not in uniq or uniq[key][2] < L:
            uniq[key] = (a, b, L)
    segments_final = list(uniq.values())

    if not segments_final:
        print("No tape segments found in the single image.")
        return

    # Save segments as PGM if requested
    if save_pgm:
        try:
            out = save_segments_to_pgm(segments_final, pgm_filename, max_size_px=2000, line_thickness_px=2, pad_m=0.1)
            print(f"Saved PGM: {out}")
        except Exception as e:
            print("Failed to save PGM:", e)

    # Plot on matplotlib
    fig, ax = plt.subplots(figsize=(8, 8))
    for a, b, L in segments_final:
        ax.plot([a[0], b[0]], [a[1], b[1]], color='blue', linewidth=3)
    ax.set_title("Detected Blue Tape (World Coordinates) - Single Image")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis('equal')
    ax.grid(True)

    xs = [p for seg in segments_final for p in (seg[0][0], seg[1][0])]
    ys = [p for seg in segments_final for p in (seg[0][1], seg[1][1])]
    pad = 0.1
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    if save_plot:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {plot_filename}")
    plt.show()

# Replace the previous main entry with a single-image call
if __name__ == "__main__":
    # set image_source to 0 to grab from camera, or to a filepath string to load an image.
    # enable save_pgm to write the detected lines to a .pgm file
    print("Running blue tape detection on camera 1")
    main_single_image(image_source=1, preview=True, save_plot=False, save_pgm=True, pgm_filename=r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Perimeter\map.pgm")
