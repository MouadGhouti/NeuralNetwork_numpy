from base_layer import Layer
import numpy as np 

class Dropout(Layer):
    def __init__(self, input_size, output_size, training=True, dropout_t=0.5):
        self.weights = np.random.randn(output_size, input_size) 
        self.bias = np.random.randn(output_size,1)
        #self.training = training
        self.dropout_t = dropout_t
        self.mask = self.create_dropout_mask(self.weights, dropout_t)



    def forward(self, input):
        self.input = input
        self.mask = self.create_dropout_mask(self.weights, self.dropout_t)
        return np.dot(self.weights*self.mask, self.input) + self.bias #W X + b
    
    def backward(self, output_gradient, learning_rate):
        dW = np.dot(output_gradient, self.input.T)*self.mask
        dx = np.dot(self.weights.T, output_gradient)
        db = output_gradient
        self.weights -= learning_rate * dW
        self.bias-=    learning_rate * db
        return dx #dE/dX
         
    def create_dropout_mask(self,input, dropout_rate):
        mask = np.ones(input.shape)
        num_zero_rows = int(input.shape[0] * dropout_rate)
        zero_indices = np.random.choice(input.shape[0], size=num_zero_rows, replace=False)
        mask[zero_indices, :] = 0
        return mask 


# x = np.random.randn(5, 14) 


# print(Dropout.create_dropout_mask(x,0))