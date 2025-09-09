import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Generate coordinates
# coordinates = [(i, i, i) for i in range(100)]
coordinates = np.load(r"C:\Users\larsc\Documents\CAPSTONE\repos\CapstoneF25\Visualizations\recorded_tvecs.npy")
tvecs = [tuple(point) for point in coordinates.squeeze(axis=2)]

xs, ys, zs = zip(*tvecs)

# Find the maximum in each dimension
max_x = max(xs)
max_y = max(ys)
max_z = max(zs)

min_x = min(xs)
min_y = min(ys)
min_z = min(zs)
# Set up figure and 3D axes

print(tvecs)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initial scatter point
point, = ax.plot([], [], [], 'ro')  # Make sure to include a comma to unpack as a tuple

# Set axes limits
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)
ax.set_zlim(min_z, max_z)

# Initialization function
def init():
    point.set_data([], [])
    point.set_3d_properties([])
    return (point,)  # return as a tuple

# Animation function
def animate(i):
    x, y, z = tvecs[i]
    # print(x,y,z)
    point.set_data([x], [y])
    point.set_3d_properties([z])
    return (point,)  # return as a tuple

# Create animation
# animate(2)
ani = FuncAnimation(fig, animate, init_func=init, frames=len(tvecs), interval=30, blit = True)

plt.show()
