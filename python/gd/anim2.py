import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

X_MIN = -10
X_MAX = 10

Y_MIN = -10
Y_MAX = 10

Z_MAX = 200

SAMPLES = 50

def rastrigin_3D(X, Y, A=10):
    return A * 2 + (X**2 - A * np.cos(2 * np.pi * X)) + (Y**2 - A * np.cos(2 * np.pi * Y))

def ackley_3D(X, Y, A=20, B=0.4, C=2 * np.pi):
    return -A * np.exp(-B * np.sqrt(0.5 * (X**2 + Y**2))) - np.exp(0.5 * (np.cos(C * X) + np.cos(C * Y))) + A + np.exp(1)

def min_fn(x, y):
    # A = 10
    # return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))
    return ackley_3D(x, y, 200, 0.1)

def plt_func_3D():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    x = np.linspace(X_MIN, X_MAX, SAMPLES)  # Coarser grid for x and y
    y = np.linspace(X_MIN, Y_MAX, SAMPLES)
    X, Y = np.meshgrid(x, y)
    Z = min_fn(X, Y)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=1, edgecolor='black', lw=0.2, cmap='coolwarm', zorder=0)

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_zlim(0, Z_MAX)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title(f'Rastrigin Function', fontsize=18)

    return ax

ax = plt_func_3D()
scatter = ax.scatter([], [], [], color='black', s=100)

def draw_point(x, y):
    scatter._offsets3d = ([x], [y], [min_fn(x, y)])


def update(t):
    x = t
    y = t
    print(x, y, min_fn(x, y))
    draw_point(x, y)
    pass


frames = 1000
interval = 500 

for frame in range(frames):
    update(frame)
    plt.pause(interval / 1000.0)

plt.show()