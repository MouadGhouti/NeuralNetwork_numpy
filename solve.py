from activation_layer import Tanh, Sigmoid
from dense_layer import Dense
from dropout_layer import Dropout
from network import train, predict
from losses import mse
import numpy as np
from sklearn.datasets import make_moons

number_samples = 1000
X, y = make_moons(n_samples=number_samples, noise=0.1)
x_data = [a for a in enumerate(X)]
x_data_train = x_data[:int(len(x_data) * .5)]
x_data_train = [i[1] for i in x_data_train]


y_data = [y[i[0]] for i in x_data]
y_data_train = y_data[:int(len(y_data) * .5)]

x_test = [a[1] for a in x_data[::-1][:int(len(x_data) * .5)]]
y_test = [a for a in y_data[::-1][:int(len(y_data) * .5)]]

x = np.array(x_data_train).astype(float).reshape((500,2,1)) # <2>
y = np.array(y_data_train).astype(float).reshape((500,1,1))
x_test = np.array(x_test).astype(float)
y_test = np.array(y_test).astype(float)

# X = np.reshape([[0,0],[0,1],[1,0],[1,1]],(4,2,1))
# Y = np.reshape([[0],[1],[1],[0]], (4,1,1))
network = [
    Dense(2,8),
    Sigmoid(),
    Dropout(8,8),
    Dense(8,5),
    Tanh(),
    Dense(5,1),
    Sigmoid(),
]

epocs = 1000
learning_rate = 0.1

train(network, epocs, learning_rate, x, y, verbose=True)

print(mse(y_test,predict(network,x_test.T)))