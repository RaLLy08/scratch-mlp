import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

x_min = -10
x_max = 10
x_0 = 10


def sphere_with_holes(x):
    return x**2 + x_max*3*np.exp(-0.2*(x-2)**2) + x_max*np.exp(-0.5*(x+5)**2)

def sphere_2D(x):
    return x**2

def rasstrigin_2D(x):
    return x**2 - 22*np.cos(2*np.pi*x) + 20

def min_fn(x):
    return rasstrigin_2D(x)


found_min_x = 9999

def sp_many(x):
    # x_p.append(x)
    # y_p.append(min_fn(x))
    y = min_fn(x)

    scatter.set_offsets(np.c_[x, y])

    if (found_min_x != 9999):
        scatter_cross.set_offsets(np.c_[found_min_x, min_fn(found_min_x)])

    

    # plt.scatter(x,[min_fn(x) for x in x], s=100)
    # for x in x:
    #     plt.text(x - 1, min_fn(x), x)


fig, ax = plt.subplots()

x_ln = np.linspace(x_min, x_max, 100)
y_ln = min_fn(x_ln)

ln, = plt.plot(x_ln, y_ln)

x_p, y_p = [], []
scatter = plt.scatter(x_p, y_p, s=200, c='red')


scatter_cross = plt.scatter([], [], s=200, marker='x')


def df(f, x):
    h = 0.000001
    return ( f(x + h) - f(x) ) / h

x_curr = x_0

def update(t):
    global x_curr, found_min_x
    feature_step = 1
    lp = 0.003

    der = df(min_fn, x_curr)
    print(found_min_x)

    x_curr = x_curr - der*lp

    if (min_fn(x_curr) < min_fn(found_min_x)):
        found_min_x = x_curr

    sp_many(x_curr)





frames = 1000
interval = 500 

for frame in range(frames):
    update(frame)
    plt.pause(interval / 1000.0)

plt.show()