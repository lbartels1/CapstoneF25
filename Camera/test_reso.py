import cv2
import os

def test_webcam_resolutions(cam_index=0, save_dir="cam_test_output"):
    # Common resolutions to test (you can add more)
    resolutions = [
        (640, 480),
        (800, 600),
        (1280, 720),
        (1280, 960),
        (1600, 900),
        (1920, 1080),
        (2560, 1440),
        (3840, 2160)
    ]

    os.makedirs(save_dir, exist_ok=True)

    print("=== Webcam Resolution Test ===")
    print(f"Testing camera index: {cam_index}\n")

    for w, h in resolutions:
        print(f"Trying {w}x{h} ...")
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)  # or CAP_MSMF / CAP_V4L2
        if not cap.isOpened():
            print("❌ Could not open camera.")
            return
        
        # Set desired resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        # Allow the camera to adjust for a few frames
        for _ in range(5):
            cap.read()

        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            cap.release()
            continue

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"✅ Captured {actual_w}x{actual_h}")

        # Save and display frame
        filename = os.path.join(save_dir, f"frame_{actual_w}x{actual_h}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}\n")

        cv2.imshow("Test Frame", frame)
        key = cv2.waitKey(500)  # show for 0.5 sec
        if key == 27:  # ESC to quit early
            cap.release()
            break

        cap.release()

    cv2.destroyAllWindows()
    print("=== Done ===")

if __name__ == "__main__":
    test_webcam_resolutions(cam_index=1)
