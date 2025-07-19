import numpy as np

class ReLU:
    def __init__(self, leak=0.01):
        self.leak = leak
        self.input = None

    def forward(self, input_data):
        self.input = input_data
        return np.maximum(self.leak * self.input, self.input)

    def backward(self, output_gradient):
        derivative = np.ones_like(self.input)
        derivative[self.input < 0] = self.leak
        return output_gradient * derivative

class Linear:
    def __init__(self, slope=1):
        self.slope = slope
    def forward(self, input):
        return self.slope*input
    def backward(self, output_gradient):
        return self.slope*output_gradient