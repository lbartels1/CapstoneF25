import cv2
import numpy as np

# Globals
clicked_points = []
real_world_points = []
num_points = 0
img = None  # Placeholder, will be set in main()

def click_event(event, x, y, flags, param):
    global clicked_points, real_world_points

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) >= num_points:
            print("✅ You've already selected the required number of points.")
            return

        clicked_points.append((x, y))
        print(f"\n📍 Point {len(clicked_points)}: Image coords = ({x}, {y})")
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select Points", img)

        # Ask for corresponding real-world coordinates
        try:
            X = float(input(f"Enter real-world X for point {len(clicked_points)} (in meters or cm): "))
            Y = float(input(f"Enter real-world Y for point {len(clicked_points)}: "))
            real_world_points.append((X, Y))
        except ValueError:
            print("❌ Invalid input. Please enter numeric coordinates.")
            clicked_points.pop()
            return

def main(image_source=1):
    global img, num_points

    # Load an image (e.g. from camera snapshot)
    height = 720
    width = 1280
    
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
        
    img = frame.copy()

    # Ask user how many points they want to use
    try:
        num_points = int(input("🔢 How many point correspondences do you want to collect (minimum 4)? "))
        if num_points < 4:
            print("❌ At least 4 points are required for homography.")
            return
    except ValueError:
        print("❌ Please enter a valid integer.")
        return

    print("\n🖱️ Click points on the image.")
    print("You'll be prompted for the real-world coordinates after each click.")
    print("Press ESC when done or after all points are collected.\n")

    cv2.imshow("Select Points", img)
    cv2.setMouseCallback("Select Points", click_event)

    while True:
        key = cv2.waitKey(1)
        if key == 27 or len(clicked_points) == num_points:
            break

    cv2.destroyAllWindows()

    if len(clicked_points) == num_points:
        image_points = np.array(clicked_points, dtype=np.float32)
        world_points = np.array(real_world_points, dtype=np.float32)

        np.save(r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\image_points.npy", image_points)
        np.save(r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Calibration_Data\world_points.npy", world_points)

        print("\n✅ Saved:")
        print(" - image_points.npy")
        print(" - world_points.npy")
    else:
        print("⚠️ Not enough points selected. Files not saved.")

if __name__ == "__main__":
    main()
