import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    Z = model.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = (Z > 0.5).astype(int).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def plot_decision_boundary_categorical(model, X, y, resolution=0.02):
    # Define the color maps
    markers = ('s', 'x', 'o', '^', 'v', '+', 'p', 'd', 'h', '8', '<', '>')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan', 'orange', 'pink', 'purple', 'brown', 'black', 'yellow', 'green')
    cmap = ListedColormap(colors[:len(np.unique(np.argmax(y, axis=1)))])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))

    grid = np.c_[xx.ravel(), yy.ravel()]
    
    Z = model.forward(grid)
    
    Z = np.argmax(Z, axis=1)
    
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # Plot the data points
    for idx, cl in enumerate(np.unique(np.argmax(y, axis=1))):
        plt.scatter(x=X[np.argmax(y, axis=1) == cl, 0], 
                    y=X[np.argmax(y, axis=1) == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=f'Class {cl}', 
                    edgecolor='black')

    # plt.legend(loc='upper left')
    plt.legend()

    plt.show()


def plot_nn_history(nn, skip_epochs=0):
    fig, axs = plt.subplots(2)
    fig.suptitle('Training history')
    # axs[0].set_ylim(0, 1)
    axs[0].plot(range(skip_epochs, len(nn.history_accuracy)), nn.history_accuracy[skip_epochs:], label='Train')
    axs[0].plot(range(skip_epochs, len(nn.history_val_accuracy)), nn.history_val_accuracy[skip_epochs:], label='Validation')
    axs[0].legend()
    axs[0].set_title('Accuracy')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')

    max_loss = np.max([np.max(nn.history_loss), np.max(nn.history_val_loss)])
    # axs[1].set_ylim(0, max_loss + max_loss * 0.1)
    axs[1].plot(range(skip_epochs, len(nn.history_loss)), nn.history_loss[skip_epochs:], label='Train')
    axs[1].plot(range(skip_epochs, len(nn.history_val_loss)), nn.history_val_loss[skip_epochs:], label='Validation')
    axs[1].legend()
    axs[1].set_title('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')

    plt.tight_layout()

    plt.show()


# adapted from: https://gist.github.com/craffel/2d727968c3aaebd10359
def draw_nn(
        ax,
        left, 
        right, 
        bottom, 
        top, 
        layer_sizes, 
        weights, 
        biases
    ):
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)

    node_r = v_spacing/2.7
    inner_circle_max_r = (node_r - node_r/4)

    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            color = (1, 1, 1, 1)
            x = n*h_spacing + left
            y = layer_top - m*v_spacing

            circle = plt.Circle(
                (x, y), 
                node_r,            
                color=color, 
                ec='k', 
                zorder=4,
            )

            if (n != 0):
                max_bias = np.max(np.abs(biases[n - 1]))

                inner_circle_r = inner_circle_max_r * (np.abs(biases[n - 1][0][m]) / max_bias)

                inner_circle = plt.Circle(
                    (x, y), 
                    inner_circle_r,            
                    color='black', 
                    zorder=5,
                )

                ax.add_artist(inner_circle)


            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                x1x2 = [n*h_spacing + left, (n + 1)*h_spacing + left]
                y1y2 = [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing]

                weight = weights[n][m, o]
                max_lw = 3
                min_lw = 0.1
                lw = min_lw + (max_lw - min_lw) * np.abs(weight)
                color = 'black'

                line = plt.Line2D(x1x2, y1y2, c=color, lw=lw)
            
                ax.add_artist(line)
