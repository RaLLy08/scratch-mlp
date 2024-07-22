import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math

x_min = -10
x_max = 10
x_0 = 10


def sphere_with_holes(x):
    return x**2 + x_max*3*np.exp(-0.2*(x-2)**2) + x_max*np.exp(-0.5*(x+5)**2)

def sphere_2D(x):
    return x**2

def rasstrigin_2D(x):
    return x**2 - 2*np.cos(2*np.pi*x) + 20

def min_fn(x):
    return sphere_2D(x)


found_min_x = 999999

def sp_many(x):
    y = min_fn(x)

    scatter.set_offsets(np.c_[x, y])

    if (found_min_x != 999999):
        scatter_cross.set_offsets(np.c_[found_min_x, min_fn(found_min_x)])


fig, ax = plt.subplots()

x_ln = np.linspace(x_min, x_max, 100)
y_ln = min_fn(x_ln)

ln, = plt.plot(x_ln, y_ln)

scatter = plt.scatter([], [], s=200, c='red')

scatter_cross = plt.scatter([], [], s=200, marker='x')




def df(f, x):
    h = 0.0000001
    return ( f(x + h) - f(x) ) / h

x_curr = x_0

der_line = plt.plot([], [], 'k-')

text = ax.text(0.3, 0.9, '', transform=ax.transAxes)

plt.legend(['Function', 'Current Point', 'Found Minimum', 'Derivative Line'], loc='upper left')


def update(t):
    global x_curr, found_min_x, vx_prev
    lp = 0.03

    vx = df(min_fn, x_curr)
    
    x_curr = x_curr - lp * vx

    if (min_fn(x_curr) < min_fn(found_min_x)):
        found_min_x = x_curr

    sp_many(x_curr)

    der_line[0].set_data([x_curr, x_curr + vx/2], [min_fn(x_curr), min_fn(x_curr + vx/2)])
    text.set_text(f'vx: {vx:.2f}, x: {x_curr:.2f}, f(x): {min_fn(x_curr):.2f}')




frames = 1000
interval = 500

for frame in range(frames):
    update(frame)
    plt.pause(interval / 1000.0)

plt.show()