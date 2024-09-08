import numpy as np
from nn import NeuralNetwork

class NeuralNetworkExtractor:
    def __init__(self, nn: NeuralNetwork):
        self.layers = nn.layers
        self.layers_outputs = []

    def get_layer_output_neurons(self, layer):
        return np.array(layer.weights.shape[1])
    
    def get_layer_input_neurons(self, layer):
        return np.array(layer.weights.shape[0])
        
    def get_neurons(self):
        outputs = [self.get_layer_output_neurons(layer) for layer in self.layers]
        outputs.insert(0, self.get_layer_input_neurons(self.layers[0]))

        return np.array(outputs)
    
    def get_weights(self):
        return [layer.weights for layer in self.layers]
    
    def get_biases(self):
        return [layer.biases for layer in self.layers]

    def get_layer_biases(self, index):
        return np.array(self.layers[index].biases)

    def get_layer_weights(self, index):
        return np.array(self.layers[index].weights)

    def forward(self, inputs):
        self.layers_outputs = []

        for layer in self.layers:
            layer.forward(inputs)
            self.layers_outputs.append(layer.output)
            inputs = layer.output

        return inputs

    def get_layers_outputs(self):
        return self.layers_outputs