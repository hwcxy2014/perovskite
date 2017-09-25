from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

v = 0.1
fig = plt.figure()
ax = fig.gca(projection='3d')
X = np.arange(-2, 2, 0.05)
Y = np.arange(-2, 2, 0.05)
X, Y = np.meshgrid(X, Y)
#R = np.sqrt(X**2 + Y**2)
# Z = np.sin(10.*X + 5.*Y)
# surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False)

W = (1 - X)*(1 - X) + 100 * (Y - X*X) * (Y - X*X)
# surf = ax.plot_surface(X, Y, W, color='0.8', linewidth=0.01, antialiased=False)
Z = np.sin(10.*X + 5.*Y)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
# surf = ax.plot_surface(X, Y, Z, color=(1,0,0), linewidth=0.01, antialiased=False)

# ax.set_zlim(-5.01, 5.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
