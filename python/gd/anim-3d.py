import os
import numpy as np
from vispy import app, scene
from vispy.color import Color
from vispy.visuals.transforms import STTransform


X_MIN = -20
X_MAX = 20

Y_MIN = -20
Y_MAX = 20

Z_MAX = 200

SAMPLES = 100

os.environ['VISPY_APP_BACKEND'] = 'pyqt5'

def rastrigin_3D(X, Y, A=10):
    return A * 2 + (X**2 - A * np.cos(2 * np.pi * X)) + (Y**2 - A * np.cos(2 * np.pi * Y))

def ackley_3D(X, Y, A=20, B=0.4, C=2 * np.pi):
    return -A * np.exp(-B * np.sqrt(0.5 * (X**2 + Y**2))) - np.exp(0.5 * (np.cos(C * X) + np.cos(C * Y))) + A + np.exp(1)

def sphere_3D(x, y):
    return x**2 + y**2

noise_int_x = 0
noise_int_y = 0
noise_int_z = 0

def min_fn(x, y):
    # A = 10
    # return A * 2 + (x**2 - A * np.cos(2 * np.pi * x)) + (y**2 - A * np.cos(2 * np.pi * y))
    return ackley_3D(x - noise_int_x, y - noise_int_y, 100, 0.1) + noise_int_z

def init():
    # Generate data
    x = np.linspace(X_MIN, X_MAX, SAMPLES)
    y = np.linspace(Y_MIN, Y_MAX, SAMPLES)
    X, Y = np.meshgrid(x, y)
    Z = min_fn(X, Y)

    # Create a canvas
    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='white')  # Set background to white

    # Create a view
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.fov = 45  # Field of view
    view.camera.distance = 25  # Set the initial camera distance farther away

    scale=(0.5, 0.5, 0.5/10)
    surface = scene.visuals.SurfacePlot(x=x, y=y, z=Z, color=(0.5, 0.5, 1, 0.3))
    surface.transform = STTransform(translate=(0, 0, 0), scale=scale)  # Adjust Z scaling for better visualization
    view.add(surface)

    # Add scatter point
    scatter_data = np.array([[x_0, y_0, min_fn(x_0, y_0)]])  # Initial scatter point at (0, 0)
    scatter = scene.visuals.Markers()
    scatter.transform = STTransform(translate=(0, 0, 0), scale=scale) 
    scatter.set_data(scatter_data, face_color=Color('red'), size=20)
    view.add(scatter)

    # Add axes
    axis = scene.visuals.XYZAxis(parent=view.scene)
    # Scale and position the axes
    axis.transform = STTransform(scale=(5, 5, 5))

    return scatter, surface



t = 0

x_0 = 10
y_0 = 10
x = x_0
y = y_0

scatter, surface = init()

def draw_point(x, y, z):
    scatter_data = np.array([[x, y, z ]])
    scatter.set_data(scatter_data, face_color=Color('red'), size=20)

def update_surface():
    x = np.linspace(X_MIN, X_MAX, SAMPLES)
    y = np.linspace(Y_MIN, Y_MAX, SAMPLES)
    X, Y = np.meshgrid(x, y)
    z = min_fn(X, Y)

    surface.set_data(x=x, y=y, z=z)

def df(f, a):
    h = 0.000001
    return ( f(a + h) - f(a) ) / h


def df_x(f, x, y):
    return df(
        lambda x_upd: f(x_upd, y),
        x
    )

def df_y(f, x, y):
    return df(
        lambda y_upd: f(x, y_upd),
        y
    )


def gradient_descent(lp=0.09):
    global x, y

    vx = df_x(min_fn, x, y)
    vy = df_y(min_fn, x, y)

    x = x - vx * lp
    y = y - vy * lp


def frame(_):
    global t, x, y, noise_int_x, noise_int_y, noise_int_z

    gradient_descent(0.1)
    # update_surface()
    # noise_min = -2
    # noise_max = 2
    # noise_int_x = np.random.uniform(noise_min, noise_max)
    # noise_int_y = np.random.uniform(noise_min, noise_max)
    # noise_int_z = np.random.uniform(noise_min, noise_max)

    draw_point(x, y, min_fn(x, y))

    t += 1


timer = app.Timer()
timer.connect(frame)
timer.start(1/16) 


if __name__ == '__main__':
    app.run()
