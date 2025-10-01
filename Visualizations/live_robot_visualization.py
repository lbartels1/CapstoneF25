import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import cv2  # For Rodrigues

# Load position (tvecs) and rotation (rvecs)
coordinates = np.load(r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Visualizations\translated_tvec.npy")
rvecs = np.load(r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Visualizations\translated_rvec.npy")

print("Loaded tvecs shape:", coordinates.shape)
print("Loaded rvecs shape:", rvecs.shape)

# Ensure proper shapes
coordinates = np.squeeze(coordinates)  # (N, 3)
rvecs = np.squeeze(rvecs)              # (N, 3)

# Convert to list of tuples for tvecs
tvecs = [tuple(point) for point in coordinates]

# Extract xs, ys, zs for plotting limits
xs, ys, zs = zip(*tvecs)

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Main point
point, = ax.plot([], [], [], 'ro')

# Set axes limits
ax.set_xlim(min(xs), max(xs))
ax.set_ylim(min(ys), max(ys))
ax.set_zlim(min(zs), max(zs))

# Add fixed global orientation arrows
arrow_length = 1

ax.quiver(0, 0, 0, arrow_length, 0, 0, color='r', arrow_length_ratio=1)
ax.quiver(0, 0, 0, 0, arrow_length, 0, color='g', arrow_length_ratio=1)
ax.quiver(0, 0, 0, 0, 0, arrow_length, color='b', arrow_length_ratio=1)

ax.text(arrow_length, 0, 0, 'X', color='r')
ax.text(0, arrow_length, 0, 'Y', color='g')
ax.text(0, 0, arrow_length, 'Z', color='b')

# Prepare to remove arrows per frame
orientation_quivers = []

def init():
    point.set_data([], [])
    point.set_3d_properties([])
    return (point,)

def animate_quivers(i):
    global orientation_quivers

    # Clear previous orientation arrows
    for q in orientation_quivers:
        q.remove()
    orientation_quivers = []

    # Update point position
    x, y, z = tvecs[i]
    point.set_data([x], [y])
    point.set_3d_properties([z])

    # Get and convert rvec to rotation matrix
    rvec = rvecs[i]
    R, _ = cv2.Rodrigues(rvec)  # (3, 3)

    # Define local axes in object space, scale as needed
    local_axes = np.eye(3) * arrow_length  # Arrow length = 5cm equivalent
    rotated_axes = R @ local_axes.T  # Shape: (3, 3)

    # Plot X, Y, Z orientation arrows
    colors = ['r', 'g', 'b']
    for j in range(3):
        dir = rotated_axes[:, j]
        q = ax.quiver(x, y, z, dir[0], dir[1], dir[2], color=colors[j], arrow_length_ratio=0.3, linewidth=2)
        orientation_quivers.append(q)

    return (point , *orientation_quivers)
    

def animate(i):

    # Update point position
    x, y, z = tvecs[i]
    point.set_data([x], [y])
    point.set_3d_properties([z])

    return (point)
    

# Animate (default to quiver-based animation)
ani = FuncAnimation(fig, animate, init_func=init, frames=len(tvecs), interval=30, blit=False)
plt.show()
