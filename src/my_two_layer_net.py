import os
import sys
import numpy as np
import pickle
import matplotlib.pylab as plt
sys.path.append(os.pardir)

from dataset.mnist import load_mnist

# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)

# network = pickle.load(open("../ch03/sample_weight.pkl", "rb"))

# W1, W2, W3 = network["W1"], network["W2"], network["W3"]
# b1, b2, b3 = network["b1"], network["b2"], network["b3"]


# train_size = x_train.shape[0]
# batch_size = 10
# batch_mask = np.random.choice(train_size, batch_size)
# x_batch = x_train[batch_mask]
# t_batch = t_train[batch_mask]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    c = np.max(x)
    return np.exp(x - c) / np.sum(np.exp(x - c))

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)** 2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t*np.log(y+delta)) / batch_size

def predict(x):
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        # f(x+h)
        x[idx] = orig + h
        fxh1 = f(x)

        # f(x-h)
        x[idx] = orig - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = orig

        it.iternext()
    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2

        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)
# class SimpleNet:
#     def __init__(self):
#         self.W = np.random.randn(2, 3)

#     def predict(self, x):
#         return np.dot(x, self.W)

#     def loss(self, x, t):
#         z = self.predict(x)
#         y = softmax(z)
#         return cross_entropy_error(y, t)

# correct = 0
# # for i in range(len(x_test)):
# #     y = predict(x_test[i])
# #     p = np.argmax(y)
# #     if p == t_test[i]:
# #         correct += 1

# batch_size = 100
# for i in range(0, len(x_test), batch_size):
#     x_batch = x_test[i:i + batch_size]
#     y_batch = predict(x_batch)
#     p = np.argmax(y_batch, axis=1)
#     correct += np.sum(p == t_test[i:i + batch_size])

# print("Accuracy:" + str(float(correct) / len(x_test)))