import os
import sys
import numpy as np
import pickle
import matplotlib.pylab as plt
sys.path.append(os.pardir)

from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)

network = pickle.load(open("../ch03/sample_weight.pkl", "rb"))

W1, W2, W3 = network["W1"], network["W2"], network["W3"]
b1, b2, b3 = network["b1"], network["b2"], network["b3"]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    c = np.max(x)
    return np.exp(x - c) / np.sum(np.exp(x - c))

def predict(x):
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

correct = 0
# for i in range(len(x_test)):
#     y = predict(x_test[i])
#     p = np.argmax(y)
#     if p == t_test[i]:
#         correct += 1

batch_size = 100
for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i:i + batch_size]
    y_batch = predict(x_batch)
    p = np.argmax(y_batch, axis=1)
    correct += np.sum(p == t_test[i:i + batch_size])

print("Accuracy:" + str(float(correct) / len(x_test)))