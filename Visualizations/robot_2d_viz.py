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

arrow_length = 1  # Adjust as needed

def init():
    point.set_data([], [])
    # Removed traj_line.set_data([], [])
    return (point,)

def animate(i):
    global orientation_quivers
    for q in orientation_quivers:
        q.remove()
    orientation_quivers = []

    x, y, z = tvecs[i]
    point.set_data([x], [y])
    # Removed traj_line.set_data(xs[:i+1], ys[:i+1])

    rvec = rvecs[i]
    R, _ = cv2.Rodrigues(rvec)

    # Local X and Y axes in object space
    local_axes = np.eye(3) * arrow_length
    rotated_axes = R @ local_axes.T

    # Only plot X and Y axes as quivers in 2D
    colors = ['r', 'g']
    for j in range(2):
        dir = rotated_axes[:, j]
        q = ax.quiver(x, y, dir[0], dir[1], color=colors[j], angles='xy', scale_units='xy', scale=1, width=0.01)
        orientation_quivers.append(q)

    return (point, *orientation_quivers)

ani = FuncAnimation(fig, animate, init_func=init, frames=len(tvecs), interval=30, blit=False)
plt.show()