import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.zeros(output_size)
        self.input = None

    def forward(self, input_data):
        output = np.dot(input_data, self.weights) + self.biases
        self.input = input_data
        return output

        
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(self.input.T, output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0)
        input_gradient = np.dot(output_gradient, self.weights.T)
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * bias_gradient

        return input_gradient

            
    

            