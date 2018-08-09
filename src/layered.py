import os
import sys
import numpy as np
import pickle
import matplotlib.pylab as plt
from collections import OrderedDict
sys.path.append(os.pardir)

from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Copy-and-paste
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

# ブロードキャストは列になされる
def softmax(x):
    if x.ndim == 2:
        # 列にテストケース
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    y = x
    y = y - np.max(y)
    return np.exp(y) / np.sum(np.exp(y))

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)** 2)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t*np.log(y+delta)) / batch_size

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


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy

class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        return dout * (self.out) * (1 - self.out)

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.dB = np.sum(dout, axis=0)

        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # output
        self.t = None  # Test

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dx = dout.copy()
        dx[self.mask] = 0
        return dx

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = np.sum(y == t) / float(x.shape[0])
        return acc

    def numerical_gradient(self, x, t):
        lw = lambda w: self.loss(x, t)

        grads = {}
        params = ["W1", "b1", "W2", "b2"]
        for p in params:
            grads[p] = numerical_gradient(lw, self.params[p])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].dB
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].dB

        return grads



net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

train_loss_list = []
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

epoch_size = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # grad = net.numerical_gradient(x_batch, t_batch)
    grad = net.gradient(x_batch, t_batch)

    for key in ("W1", "b1", "W2", "b2"):
        net.params[key] -= learning_rate * grad[key]

    loss = net.loss(x_batch, t_batch)
    if i % epoch_size == 0:
        print("Train Accuracy = " + str(net.accuracy(x_train, t_train)))
        print("Test Accuracy = " + str(net.accuracy(x_test, t_test)))

    train_loss_list.append(loss)
# x = np.random.rand(100, 784)
# t = np.random.rand(100, 10)
# y = net.predict(x)
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