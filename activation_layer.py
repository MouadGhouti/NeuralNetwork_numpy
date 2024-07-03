from base_layer import Layer
import numpy as np

class activation(Layer):
    def __init__(self, activation, activation_derivative):
        self.activation = activation
        self.activation_derivative = activation_derivative
    
    def forward(self, input):
        self.input = input
        return self.activation(input)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_derivative(self.input))
    

class Sigmoid(activation):
    def __init__(self):
        sigmoid = lambda x: 1/(1+np.exp(-x))
        sigmoid_derivative= lambda x: sigmoid(x) * (1-sigmoid(x))
        super().__init__(sigmoid, sigmoid_derivative)


class Tanh(activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_derivative = lambda x: 1 - np.tanh(x)**2
        super().__init__(tanh,tanh_derivative)
        
class Threshold(activation):
    def __init__(self, theta = 0.5):
        threshold = lambda x: 1 if x > theta else 0
        threshold_derivative = lambda x: 0 
        super().__init__(threshold,threshold_derivative)