import tqdm
from losses import mse, mse_derivative
from display import visNetwork
import matplotlib.pyplot as plt

def train(network, epochs, learning_rate, input, outputs, verbose=False):
    for e in tqdm.tqdm(range(epochs)):
        error = 0 
        for x,y in zip(input,outputs):
            output=x
            for layer in network:
                output=layer.forward(output)
            
            error += mse(y,output)

            grad=mse_derivative(y,output)
            for layer in reversed(network):
                grad=layer.backward(grad, learning_rate)
                
        if(verbose):
            visNetwork(network, error)
    plt.ioff()
    plt.show()


def predict(network, input):
    output=input
    for layer in network:
        if layer.__module__ == 'dropout':
           layer.dropout_t=0
        if layer.__module__ == 'Threshold':
            continue
        # else:
        #     output=layer.forward(output)
    return output