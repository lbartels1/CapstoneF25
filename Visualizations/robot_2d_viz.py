import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import cv2  # For Rodrigues

# Load position (tvecs) and rotation (rvecs)
coordinates = np.load(r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Visualizations\translated_tvec.npy")
rvecs = np.load(r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Visualizations\translated_rvec.npy")

coordinates = np.squeeze(coordinates)  # (N, 3)
rvecs = np.squeeze(rvecs)              # (N, 3)

tvecs = [tuple(point) for point in coordinates]
xs, ys = zip(*[(x, y) for x, y, z in tvecs])

fig, ax = plt.subplots(figsize=(8, 6))
point, = ax.plot([], [], 'ro')
# Removed traj_line
ax.set_xlim(min(xs), max(xs))
ax.set_ylim(min(ys), max(ys))
ax.set_title('2D Trajectory & Orientation Animation')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.grid(True)

orientation_quivers = []
# For the no-quiver (line) visualization
orientation_lines = []

arrow_length = 1  # Adjust as needed

def init():
    point.set_data([], [])
    return (point,)


def animate_quivers(i):
    """Quiver-based animation using fixed-length arrows rotating only about Z."""
    global orientation_quivers
    # Remove previous quivers
    for q in orientation_quivers:
        try:
            q.remove()
        except Exception:
            pass
    orientation_quivers = []

    x, y, z = tvecs[i]
    point.set_data([x], [y])

    rvec = rvecs[i]
    R, _ = cv2.Rodrigues(rvec)

    # Compute yaw (heading) from rotation matrix
    yaw = np.arctan2(R[1, 0], R[0, 0])

    # Forward (red) vector in XY plane
    fx = arrow_length * np.cos(yaw)
    fy = arrow_length * np.sin(yaw)

    # Left (green) vector in XY plane (90 deg rotated)
    lx = arrow_length * np.cos(yaw + np.pi/2)
    ly = arrow_length * np.sin(yaw + np.pi/2)

    # Plot the two fixed-length quivers
    qf = ax.quiver(x, y, fx, fy, color='r', angles='xy', scale_units='xy', scale=1, width=0.01)
    ql = ax.quiver(x, y, lx, ly, color='g', angles='xy', scale_units='xy', scale=1, width=0.01)
    orientation_quivers.extend([qf, ql])

    return (point, *orientation_quivers)


def animate_no_quivers(i):
    """Line-based animation: fixed-length colored segments rotating about Z only."""
    global orientation_lines
    # Remove previous lines
    for ln in orientation_lines:
        try:
            ln.remove()
        except Exception:
            pass
    orientation_lines = []

    x, y, z = tvecs[i]
    point.set_data([x], [y])

    rvec = rvecs[i]
    R, _ = cv2.Rodrigues(rvec)
    yaw = np.arctan2(R[1, 0], R[0, 0])

    fx = arrow_length * np.cos(yaw)
    fy = arrow_length * np.sin(yaw)

    lx = arrow_length * np.cos(yaw + np.pi/2)
    ly = arrow_length * np.sin(yaw + np.pi/2)

    # Forward line (red)
    lf, = ax.plot([x, x + fx], [y, y + fy], color='r', linewidth=2)
    orientation_lines.append(lf)

    # Left line (green)
    ll, = ax.plot([x, x + lx], [y, y + ly], color='g', linewidth=2)
    orientation_lines.append(ll)

    return (point, *orientation_lines)


# Choose animation style: set to False to use the no-quiver (line) version
use_quivers = False
animate_func = animate_quivers if use_quivers else animate_no_quivers

ani = FuncAnimation(fig, animate_func, init_func=init, frames=len(tvecs), interval=30, blit=False)
plt.show()