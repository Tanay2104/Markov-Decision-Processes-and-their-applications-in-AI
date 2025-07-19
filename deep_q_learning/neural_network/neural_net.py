import numpy as np


class NeuralNetwork:
    def __init__(self, layers=None):
        if layers is None:
            self.layers = []
        else:
            self.layers = layers

    def add(self, layer, activation_function):
        self.layers.append((layer, activation_function))

    def forward(self, input_data):
        data = input_data
        for layer, activation_function in self.layers:
            data = activation_function.forward(layer.forward(data))

        return data

    def backward(self, output_gradient, learning_rate):
        gradient = output_gradient
        for layer, activation_function in self.layers[::-1]:
            gradient = layer.backward(activation_function.backward(gradient), learning_rate)
        
        return gradient
    
if __name__ == "__main__":  


    
    from layer import Layer
    from loss_functions import mse, mse_prime
    from activation_functions import ReLU, Linear  

    net1 = NeuralNetwork() # Gets the default list
    net1.add(Layer(2, 2), ReLU()) # Modifies the default list

    net2 = NeuralNetwork() # Also gets the SAME default list
    print(len(net2.layers))

    X_train = np.array([ [0,0], [0,1], [1,0], [1,1] ])
    y_train = np.array([ [0], [1], [1], [0] ])

    net = NeuralNetwork()

    net.add(Layer(input_size=2, output_size=16), ReLU())
    net.add(Layer(input_size=16, output_size=1), Linear())


    epochs = 1000
    learning_rate = 0.01

    for i in range(epochs):
        predictions = net.forward(X_train)
        loss = mse(y_train, predictions)
        gradient = mse_prime(y_train, predictions)
        net.backward(gradient, learning_rate)
        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss}")

    final_predictions = net.forward(X_train)
    print("Final Predictions:")
    print(final_predictions)

