from base_layer import Layer
import numpy as np 

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) 
        self.bias = np.random.randn(output_size,1)
        self.gradient = 0


    def forward(self, input):
        self.input = input
        stream = np.dot(self.weights, self.input) + self.bias #W X + b
        #print("dense forward: ",stream.shape)
        return  stream
    
    def backward(self, output_gradient, learning_rate):
        dW = np.dot(output_gradient, self.input.T)
        dx = np.dot(self.weights.T, output_gradient)
        db = output_gradient
        self.weights -= learning_rate * dW
        self.bias-=    learning_rate * db
        self.gradient = dx
        #print("Gradient of dense layer:", dx.shape)
        return dx #dE/dX
         