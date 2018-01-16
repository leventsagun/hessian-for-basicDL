from autograd import numpy as np

from . import Model
from .forward import make_logreg_predict


def create_logreg_model(args, inputs, targets):
    """Create a Logistic Regression model. (Factory function)"""
    initial_params = np.ones(args.input_dim) * 0.01
    predict_logreg = make_logreg_predict(inputs)
    mdl = Model(initial_params, predict_logreg, inputs, targets)

    return mdl
