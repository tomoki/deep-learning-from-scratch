import os
import sys
import numpy as np
import pickle
import matplotlib.pylab as plt
sys.path.append(os.pardir)

from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=True)

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

# Copy-and-paste
def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

# def softmax(x):
#     c = np.max(x)
#     return np.exp(x - c) / np.sum(np.exp(x - c))


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

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
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

    # Copy-and-paste
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

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

print(train_loss_list)
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