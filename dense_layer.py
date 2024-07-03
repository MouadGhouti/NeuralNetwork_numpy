from base_layer import Layer
import numpy as np 

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) 
        self.bias = np.random.randn(output_size,1)


    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias #W X + b
    
    def backward(self, output_gradient, learning_rate):
        dW = np.dot(output_gradient, self.input.T)
        dx = np.dot(self.weights.T, output_gradient)
        db = output_gradient
        self.weights -= learning_rate * dW
        self.bias-=    learning_rate * db
        return dx #dE/dX
         