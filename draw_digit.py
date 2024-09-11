import tkinter as tk
from PIL import Image, ImageDraw
import time
import numpy as np
import math
from nn import NeuralNetwork, LayerDense
from nn_extractor import NeuralNetworkExtractor
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
from tkinter import StringVar
import os

MODELS_PATH = './models/'
MODELS = os.listdir(MODELS_PATH)
DEFAULT_MODEL_NAME = MODELS[0]


MODEL_PATH = f'{MODELS_PATH}{DEFAULT_MODEL_NAME}'


mlp_mnist = NeuralNetwork.load(MODEL_PATH)


time_now = time.time()

WIDTH, HEIGHT = 400, 400
PEN_RADIUS = 16

class DrawingApp:
    def init_nn_figures(self):
        figure = Figure(figsize=(5, 4), dpi=100)
        self.ax =  figure.add_axes([-0.02, 0.03, 0.3, 0.94])
        self.ax2 = figure.add_axes([0.14, 0.01, 0.2, 0.98])
        self.ax3 = figure.add_axes([0.18, 0.01, 0.2, 0.98])

        self.ax4 = figure.add_axes([0.33, 0.25, 0.1, 0.5])
        self.ax5 = figure.add_axes([0.4, 0.25, 0.2, 0.5])
        self.ax6 = figure.add_axes([0.4, 0.25, 0.3, 0.5])

        self.ax7 = figure.add_axes([0.58, 0.375, 0.2, 0.25])
        self.ax8 = figure.add_axes([0.62, 0.375, 0.3, 0.25])
        self.ax9 = figure.add_axes([0.67, 0.375, 0.3, 0.25])
        
        self.ax10 = figure.add_axes([0.87, 0.12, 0.1, 0.8])

        figure.subplots_adjust(hspace=0.5)

        figure.tight_layout()

        return figure

    def init_nn_axes(self):
        mlp_mnist_nne = NeuralNetworkExtractor(mlp_mnist)
        weigths = mlp_mnist_nne.get_weights()
        biases = mlp_mnist_nne.get_biases()
        neurons = mlp_mnist_nne.get_neurons()

        self.ax.set_ylabel(f'First layer weights {weigths[0].T.shape}', fontsize=14)
        self.ax.yaxis.tick_right()
        
        self.ax2.set_ylabel(f'First layer biases ({neurons[1]})', fontsize=14)
        self.ax2.set_frame_on(False)
        self.ax2.set_yticks([])
        self.ax2.set_xticks([])

        self.ax3.set_ylabel(f'First outputs activated by ReLU ({neurons[1]})', fontsize=14)
        self.ax3.yaxis.tick_right()
        self.ax3.set_xticks([])


        self.ax4.set_ylabel(f'Second layer weights {weigths[1].T.shape}', fontsize=14)
        self.ax4.yaxis.tick_right()

        self.ax5.set_ylabel(f'Second layer biases ({neurons[2]})', fontsize=14)
        self.ax5.yaxis.tick_right()
        self.ax5.set_frame_on(False)
        self.ax5.set_yticks([])
        self.ax5.set_xticks([])


        self.ax6.set_ylabel(f'Second outputs activated by ReLU ({neurons[2]})', fontsize=14)
        self.ax6.yaxis.tick_right()
        self.ax6.set_xticks([])

        self.ax7.set_ylabel(f'Output layer weights {weigths[2].T.shape}', fontsize=14)
        self.ax7.yaxis.tick_right()

        self.ax8.set_ylabel(f'Output layer biases ({neurons[3]})', fontsize=14)
        self.ax8.yaxis.tick_right()
        self.ax8.set_xticks([])
        self.ax8.set_frame_on(False)
        self.ax8.set_yticks([])
        

        self.ax9.set_ylabel(f'Output layer outputs ({neurons[3]})', fontsize=14)
        self.ax9.yaxis.tick_right()
        self.ax9.set_xticks([])


        self.ax10.set_title('Bar chart', fontsize=14)
        self.ax10.set_xlabel('Probability')
        self.ax10.set_yticks(np.arange(10))
        self.ax10.set_xticks([])
        self.ax10.set_frame_on(False)
        self.ax10.invert_yaxis()


    def init_input_image_figure(self):
        figure = Figure(figsize=(5, 4), dpi=100)
        self.img_plot_ax = figure.add_subplot(111)

        return figure

    def init_input_image_axes(self):
        self.img_plot_ax.set_title('Input image (28x28 pixels)')
        self.img_plot_ax.axis('off')
        # fill with black color

    def get_options(self) -> list:
        return MODELS

    def on_option_change(self, *args):
        global mlp_mnist
        path = f'{MODELS_PATH}{self.selected_option.get()}'
        mlp_mnist = NeuralNetwork.load(path)

        hyperparams = f"learning rate: {mlp_mnist.learning_rate}, epochs: {mlp_mnist.epochs}, batch size: {mlp_mnist.batch_size}, train size: {mlp_mnist.train_size}"
        
        self.hyperparameters_label.config(text=hyperparams)

        self.clear_static()
        self.draw_static(mlp_mnist)

        self.clear_plots()
        self.update_thread = threading.Thread(target=self.update_probabilities, daemon=True)
        self.update_thread.start()


    def __init__(self, root):
        self.root = root
        self.root.title("Drawing App")

        # Create a frame for the plot
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)


        self.plot_frame_img = tk.Frame(self.root)
        self.plot_frame_img.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Create a Matplotlib figure and axes

        nn_figure = self.init_nn_figures()
        self.init_nn_axes()

        self.canvas = FigureCanvasTkAgg(nn_figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.draw()


        input_image_figure = self.init_input_image_figure()

        self.canvas_img = FigureCanvasTkAgg(input_image_figure, master=self.plot_frame_img)
        self.canvas_img.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.init_input_image_axes()


        self.canvas_img.draw()


        # Create the drawing canvas
        self.canvas_draw = tk.Canvas(self.root, bg='black', width=WIDTH, height=HEIGHT)
        self.canvas_draw.pack(side=tk.BOTTOM)

        self.image = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)

        self.canvas_draw.bind("<B1-Motion>", self.paint)
        self.canvas_draw.bind("<ButtonRelease-1>", self.reset)

        # self.coord_label = tk.Label(self.root, text="Coordinates: ")
        # self.coord_label.pack()

        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack()

        # OptionMenu for the dropdown select input
        self.selected_option = StringVar()
        self.selected_option.set(DEFAULT_MODEL_NAME)

        self.option_menu = tk.OptionMenu(self.control_frame, self.selected_option, *self.get_options())
        self.selected_option.trace_add('write', self.on_option_change)

        self.option_menu.grid(row=10, column=1)

        dropdown_label = tk.Label(self.control_frame, text="Selected Model:", font=("Helvetica", 12))
        dropdown_label.grid(row=10, column=0, padx=10, pady=10)

        hyperparams = f"learning rate: {mlp_mnist.learning_rate}, epochs: {mlp_mnist.epochs}, batch size: {mlp_mnist.batch_size}, train size: {mlp_mnist.train_size}"
        self.hyperparameters_label = tk.Label(self.root, text=hyperparams, font=("Helvetica", 12))
        self.hyperparameters_label.pack(side=tk.TOP)

        # self.prob_frame = tk.Frame(self.root)
        # self.prob_frame.pack()

        self.clear_button = tk.Button(self.root, text="Clear", command=lambda: self.clear_canvas() or self.clear_plots(),
         font=("Helvetica", 12), bg="red", fg="white")
        self.clear_button.pack(side=tk.BOTTOM)

        # self.prob_labels = []
        # for i in range(10):
        #     bg = 'light blue'
        #     label = tk.Label(self.prob_frame, text=f"Probability {i}: 0.00", font=("Helvetica", 12), bg=bg, width=20, anchor='w')
        #     label.grid(row=i, column=0, padx=5, pady=2)
        #     self.prob_labels.append(label)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.last_x, self.last_y = None, None

        self.draw_static(mlp_mnist)

    def paint(self, event):
        if self.last_x and self.last_y:
            x, y = event.x, event.y

            radius = PEN_RADIUS
            self.canvas_draw.create_oval(x - radius, y - radius, x + radius, y + radius, fill='white', outline='white')
            self.draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill='white', outline='white')

            # self.coord_label.config(text=f"Coordinates: {x}, {y}")

            # Run update_probabilities in a separate thread to avoid freezing

        global time_now
        current_time = time.time()
        delay = 0.05
        if time_now + delay < current_time:
            if hasattr(self, 'update_thread') and self.update_thread.is_alive():
                return

            # Run update_probabilities in a separate thread
            self.update_thread = threading.Thread(target=self.update_probabilities, daemon=True)
            self.update_thread.start()
            time_now = current_time


        self.last_x, self.last_y = event.x, event.y

    def draw_static(self, mlp):
        mlp_mnist_nne = NeuralNetworkExtractor(mlp)

        mlp_weights = mlp_mnist_nne.get_weights()
        mlp_biases = mlp_mnist_nne.get_biases()

        self.ax.imshow(mlp_weights[0], cmap='gray')
        self.ax2.imshow(mlp_biases[0].T, cmap='gray')

        self.ax4.imshow(mlp_weights[1], cmap='gray')
        self.ax5.imshow(mlp_biases[1].T, cmap='gray')

        self.ax7.imshow(mlp_weights[2], cmap='gray')
        self.ax8.imshow(mlp_biases[2].T, cmap='gray')

    def clear_static(self):
        self.ax.clear()
        self.ax2.clear()

        self.ax4.clear()
        self.ax5.clear()

        self.ax7.clear()
        self.ax8.clear()


    def reset(self, event):
        self.last_x, self.last_y = None, None

    def update_probabilities(self):
        image = self.image.resize((28, 28))
        image_array = np.array(image)[:, :, 0].flatten() / 255.0

        mlp_mnist_nne = NeuralNetworkExtractor(mlp_mnist)
        predictions = mlp_mnist_nne.forward(image_array)[0]
        predictions_rounded = np.round(predictions, 2)
        max_index = np.argmax(predictions_rounded)

        mlp_outputs = mlp_mnist_nne.get_layers_outputs()
        neurons = mlp_mnist_nne.get_neurons()

        # Update existing plots instead of clearing and redrawing
        self.ax3.axis('on')
        self.ax3.imshow(mlp_outputs[0].T, cmap='gray')
        self.ax3.set_ylabel(f'First outputs activated by ReLU ({neurons[1]})', fontsize=14)

        self.ax6.axis('on')
        self.ax6.imshow(mlp_outputs[1].T, cmap='gray')
        self.ax6.set_ylabel(f'Second outputs activated by ReLU ({neurons[2]})', fontsize=14)

        self.ax9.axis('on')
        self.ax9.imshow(mlp_outputs[2].T, cmap='gray')
        self.ax9.set_ylabel('Output layer outputs activated by softmax (10)', fontsize=14)

        self.img_plot_ax.imshow(image, cmap='gray')

        self.ax10.barh(np.arange(9, -1, -1), predictions[::-1], color=['lightgreen' if (9 -i) == max_index else 'blue' for i in range(10)])

        # for i in range(10, 0):
        #     self.ax10.text(i, predictions[i], f"{predictions_rounded[::-1][i]:.2f}", ha='center', va='center', color='black')

        self.canvas.draw()
        self.canvas_img.draw()

        self.ax10.clear()
        self.ax10.set_title('Bar chart', fontsize=14)
        self.ax10.set_xlabel('Probability')
        self.ax10.set_yticks(np.arange(10))
        self.ax10.set_xticks([])
        self.ax10.set_frame_on(False)
        self.ax10.invert_yaxis()


    def on_closing(self):
        self.image = self.image.resize((28, 28))
        self.image.save("drawing.png")
        self.root.destroy()

    def clear_canvas(self):
        self.img_plot_ax.clear()

        self.canvas_img.draw()

        self.canvas_draw.delete("all")
        self.image = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.image)

        # for i, label in enumerate(self.prob_labels):
        #     bg = 'light blue'
        #     label.config(text=f"Probability {i}: 0.00", bg=bg)

    def clear_plots(self):
        self.ax3.clear()
        self.ax3.axis('off')

        self.ax6.clear()
        self.ax6.axis('off')

        self.ax9.clear()
        self.ax9.axis('off')

        self.ax10.clear()


        self.img_plot_ax.clear()

        self.init_input_image_axes()
        self.init_nn_axes()
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
