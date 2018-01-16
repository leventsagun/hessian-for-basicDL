"""
Module for abstract Model class.

Different models differ only by their prediction function predict().
"""
import autograd
from autograd import numpy as np
from autograd.util import flatten, quick_grad_check

import model


class Model(object):
    """Generic model that have methods to calculate loss, gradient etc."""

    def __init__(self, params, predict, inputs, targets):
        """Construct a Model object given a prediction function."""
        self.__params = params
        self.__params_flat, self.unflatten_params = flatten(self.params)
        self.predict = predict
        self.inputs = inputs
        self.targets = targets

        self.gradient = autograd.grad(self.loss)
        self.hessian = autograd.hessian(self.loss)
        self.hess_dot_vec = autograd.hessian_vector_product(self.loss)
        self.grad_rayleigh = autograd.grad(self.rayleigh_quotient)

    def loss(self, params_flat, inputs=None, targets=None):
        """Take unflatten parameters, flatten them and calculate log-loss."""
        params = self.unflatten_params(params_flat)

        if inputs is None:
            predictions = self.predict(params)
            return log_loss(self.targets, predictions)

        predictions = self.predict(params, inputs)
        return log_loss(targets, predictions)

    @property
    def params(self):
        return self.__params

    @params.setter
    def params(self, params):
        self.__params = params
        self.__params_flat, self.unflatten_params = flatten(self.__params)

    @property
    def params_flat(self):
        return self.__params_flat

    @params_flat.setter
    def params_flat(self, params_flat):
        self.__params_flat = params_flat
        self.__params = self.unflatten_params(self.__params_flat)

    def rayleigh_quotient(self, vec):
        hv_val = self.hess_dot_vec(self.params_flat, vec)
        rq = np.dot(hv_val, vec) / np.dot(vec, vec)
        return rq


def log_loss(targets, preds, eps=1e-15):
    preds = np.clip(preds, eps, 1 - eps)
    return - np.mean(np.sum(targets * np.log(preds), axis=1))


def create_model(args, inputs, targets):
    if args.classifier == 'logreg':
        mdl = model.create_logreg_model(args, inputs, targets)
    elif args.classifier == 'fullconn':
        mdl = model.create_fully_connected_model(args, inputs, targets)
    else:
        raise Exception('Unknown classifier type {}'.format(args.classifier))

    quick_grad_check(mdl.loss, mdl.params_flat, verbose=False)
    return mdl
