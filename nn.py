import numpy as np
import math
import pickle
import os


class LayerDense:
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)
                
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def heaviside(x):
        return np.heaviside(x, 0)
    
    @staticmethod
    def softmax(x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probs
    
    
    def df_activation(self, activated_output):
        """
            activated_output -> aL = σ(zL)
        """

        if self.activation == LayerDense.sigmoid:
            return activated_output * (1 - activated_output)
        elif self.activation == LayerDense.relu:
            return np.where(activated_output <= 0, 0, 1)
        elif self.activation == LayerDense.softmax:
            batch_size = activated_output.shape[0]
            # Initialize the tensor for storing the derivatives
            d_softmax = np.zeros((batch_size, activated_output.shape[1], activated_output.shape[1]))
            for i in range(batch_size):
                s = activated_output[i].reshape(-1, 1)
                d_softmax[i] = np.diagflat(s) - np.dot(s, s.T)

            return d_softmax


    def init_weights(self):
        self.weights = np.random.randn(self.inputs_len, self.neurons_len) * np.sqrt(2. / self.inputs_len)

    def init_biases(self):
        self.biases = np.zeros((1, self.neurons_len))

    def __init__(self, inputs_len, neurons_len, activation=relu):
        self.inputs_len = inputs_len
        self.neurons_len = neurons_len
        self.activation = activation
        self.init_weights()
        self.init_biases()

    def forward(self, inputs):
        self.output = self.activation(np.dot(inputs, self.weights) + self.biases)
        return self.output


class NeuralNetwork:
    @staticmethod
    def load(path) -> 'NeuralNetwork':
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save(self, name, folder=''):
        hyperparameters = f'{self.learning_rate}_{self.epochs}_{self.batch_size}_{self.train_size}'
        filename = name + '_' + hyperparameters + '.pkl'
        path = os.path.join(folder, filename)

        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def loss_mse(y_true, y_pred):
        return np.mean((y_true - y_pred)**2)
    
    @staticmethod
    def df_loss_mse(y_true, y_pred):
        return y_pred - y_true
    
    @staticmethod
    def loss_binary_crossentropy(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def loss_categorical_crossentropy(y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss

    def __init__(self, loss=loss_mse, log=True):
        self.log = log
        self.loss = loss
        self.history_accuracy = []
        self.history_loss = []
        self.history_val_accuracy = []
        self.history_val_loss = []
        self.layers = []

        # hyperparameters
        self.learning_rate = 0.01
        self.epochs = 10
        self.batch_size = 10
        self.validation_split = 0.1
        self.train_size = 0


    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            layer.forward(inputs)
            inputs = layer.output

        return inputs
    
    def calculate_output_delta(self, layer, y_true, y_pred):
        if layer.activation == LayerDense.softmax:
            return y_pred - y_true
        if layer.activation == LayerDense.sigmoid:
            grad_loss_input = self.df_loss_mse(y_true, y_pred) # derivative of loss function = ∇aL | samples -> neurons
            activation_derivative = layer.df_activation(layer.output) # derivative of sigmoid(zL) = sigmoid(zL) * (1 - sigmoid(zL)) | samples -> neurons

            return grad_loss_input * activation_derivative # δL = ∇aL * σ'(zL) | samples -> neurons

    def backward_auto(self, inputs, y_true, y_pred, lr, batch_size):
        layers_count = len(self.layers)

        for layer_index in reversed(range(0, layers_count)):
            layer = self.layers[layer_index]

            is_output_layer = layer_index == layers_count - 1
            
            if is_output_layer:
                # output layer
                layer.delta = self.calculate_output_delta(layer, y_true, y_pred)
                # print(layer.delta)
            else:
                prev_backward_layer = self.layers[layer_index + 1]        

                layer.propagated_error = np.dot(prev_backward_layer.delta, prev_backward_layer.weights.T) # δL * wL.T

                layer.activation_derivative = layer.df_activation(layer.output) 
                layer.delta = layer.propagated_error * layer.activation_derivative

            if layer_index != 0:
                next_backward_layer = self.layers[layer_index - 1]

                next_backward_layer_output = next_backward_layer.output.T
            else:
                next_backward_layer_output = inputs.T

            layer.grad_loss_weights = np.dot(next_backward_layer_output, layer.delta) # ∇wL = aL-1.T * δL | neurons -> neurons
            layer.grad_loss_biases = np.sum(layer.delta, axis=0, keepdims=True)  # ∇bL = δL | neurons -> 1

            # clip the gradients to prevent exploding gradients
            np.clip(layer.grad_loss_weights, -1, 1, out=layer.grad_loss_weights)  
            np.clip(layer.grad_loss_biases, -1, 1, out=layer.grad_loss_biases)

            layer.weights = layer.weights - lr * layer.grad_loss_weights / batch_size
            layer.biases = layer.biases - lr * layer.grad_loss_biases / batch_size

    def validation_train_split(self, inputs, y_true, validation_split, log=True):
        """
            Split the data into training and validation sets based on the validation_split percentage
            (Holdout Method)
        """

        x_val = inputs[-int(len(inputs) * validation_split):]
        y_val = y_true[-int(len(inputs) * validation_split):]
        x_true = inputs[:-int(len(inputs) * validation_split)]
        y_true = y_true[:-int(len(inputs) * validation_split)]

        if log:
            print(f'Validation set size: {len(x_val)}')
            print(f'Training set size: {len(x_true)}')


        return x_val, y_val, x_true, y_true

    def output_layer(self):
        return self.layers[-1]
    
    def accuracy(self, y_true, y_pred):
        if self.output_layer().activation == LayerDense.softmax:
            return np.mean(y_pred == y_true.argmax(axis=1))
        if self.output_layer().activation == LayerDense.sigmoid:
            return np.mean(y_true == np.round(y_pred))
        
    def evaluate(self, x, y):
        total_loss = self.loss(y, self.forward(x))
        accuracy = self.accuracy(y, self.predict(x))

        return total_loss, accuracy

    def record_history(self, x_train, y_train, x_val, y_val, epoch, epochs):
        validation_loss, validation_accuracy = self.evaluate(x_val, y_val)
        loss, accuracy = self.evaluate(x_train, y_train)

        self.history_accuracy.append(accuracy)
        self.history_loss.append(loss)
        self.history_val_accuracy.append(validation_accuracy)
        self.history_val_loss.append(validation_loss)

        if self.log:
            print(f'Evaluation set size: {len(x_train)}')
            print(f'Validation set size: {len(x_val)}')
            print(f'Epoch: {epoch+1}/{epochs} | loss {loss} | accuracy {accuracy} | validation loss {validation_loss} | validation accuracy {validation_accuracy}')
        

    def fit(self, inputs, y_true, learning_rate=0.01, epochs=10, batch_size=10, validation_split=0.1):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split

        print(f'Total iterations {epochs * (math.ceil(len(inputs) / batch_size))}')
        print(f'Iterations of each epoch {math.ceil(len(inputs) / batch_size)}')

        x_val, y_val, x_true, y_true = self.validation_train_split(inputs, y_true, validation_split)
        self.train_size = len(x_true)

        for epoch in range(epochs):
            self.record_history(x_true, y_true, x_val, y_val, epoch, epochs)

            shuffle_indices = np.random.permutation(len(x_true))
            x_true = x_true[shuffle_indices]
            y_true = y_true[shuffle_indices]

            for i in range(0, len(x_true), batch_size):
                inputs_batch = x_true[i:i+batch_size]
                y_true_batch = y_true[i:i+batch_size]

                y_hats = self.forward(inputs_batch) # predict the output
                 
                self.backward_auto(
                    inputs_batch, 
                    y_true_batch, 
                    y_hats, 
                    learning_rate, 
                    batch_size
                )

    def reset(self):
        self.history_accuracy = []
        self.history_loss = []
        self.history_val_accuracy = []
        self.history_val_loss = []

        for layer in self.layers:
            layer.init_weights()
            layer.init_biases()

    def predict(self, inputs):
        y_hat = self.forward(inputs)

        if self.output_layer().activation == LayerDense.softmax:
            return y_hat.argmax(axis=1)
        if self.output_layer().activation == LayerDense.sigmoid:
            return np.round(y_hat)
