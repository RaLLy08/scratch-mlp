import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Initialize a blank 28x28 grayscale image
image = np.zeros((28, 28), dtype=np.uint8)

# Variables to keep track of drawing state
is_drawing = False

def on_press(event):
    global is_drawing
    is_drawing = True

def on_release(event):
    global is_drawing
    is_drawing = False

def on_motion(event):
    if is_drawing and event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        if 0 <= x < 28 and 0 <= y < 28:
            image[y, x] = 255  # Set pixel to white
            update_canvas()

def update_canvas():
    plt.clf()
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')  # Hide axes for better visualization
    plt.draw()

def save_image(event=None):
    plt.imsave('digit_image.png', image, cmap='gray', format='png')
    print("Image saved as 'digit_image.png'")

fig, ax = plt.subplots()
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

# Connect the close event to save the image
fig.canvas.mpl_connect('close_event', save_image)

update_canvas()
plt.show()
