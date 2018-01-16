"""
Module that includes makers of different prediction functions of different classification models.

* They take inputs. 
* Create predict() function that takes classifier parameters as input
* and return classifier of predictions.
* make_ functions encapsulate `inputs` so that predict() is a function of single argument.
"""
from autograd import numpy as np


def make_logreg_predict(inputs):
    """Return probability prediction function of Logistic Regression."""
    def predict(weights, inputs=inputs):
        probs_label_1 = sigmoid_exp(np.dot(inputs, weights))
        probs_label_1 = probs_label_1.reshape(-1, 1)
        probs_label_0 = 1.0 - probs_label_1
        probs = np.hstack([probs_label_0, probs_label_1])
        return probs

    return predict


def make_fully_connected_predict(inputs):
    """Return probability prediction function of Fully-Connected Neural Network."""
    def predict(params, inputs=inputs, activation=relu):
        """        
        :param params: a list of (weights, bias) tuples
        :param activation: the activation function
        :return: normalized class probabilities
        """
        inp = inputs
        for W, b in params:
            out = np.dot(inp, W) + b
            inp = activation(out)  # activation tanh
        return softmax(out)

    return predict


def softmax(mat):
    return np.exp(mat) / np.sum(np.exp(mat), axis=1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def sigmoid_exp(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_tanh(x):
    return 0.5 * (np.tanh(x) + 1.0)
