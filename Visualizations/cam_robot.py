from mpl_toolkits import mplot3d

import numpy as np
# import matplotlib as plt
import matplotlib.pyplot as plt
import time

fig = plt.figure()
ax = plt.axes(projection='3d')

# Data for a three-dimensional line


# Data for three-dimensional scattered points
zdata = [0.6170, 47.5] 
xdata = [-1.1621, -89.1]
ydata = [-0.2567, -23.8]
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');

plt.show()