from autograd import numpy as np

from .forward import make_fully_connected_predict
from . import Model


def create_fully_connected_model(args, inputs, targets):
    """Create a Fully-Connected NN model. (Factory function)"""
    hidden_layer_sizes = args.layer_sizes
    param_scale = 0.1

    layer_sizes = [args.input_dim] + hidden_layer_sizes + [args.num_classes]
    init_params = initalize_parameters(layer_sizes, param_scale)
    predict_full_conn = make_fully_connected_predict(inputs)

    mdl = Model(init_params, predict_full_conn, inputs, targets)

    return mdl


def initalize_parameters(layer_sizes, scale=0.1):
    """Build a list of (weights, biases) tuples, one for each layer in the net."""
    params = []
    for n_units_in_curr_layer, n_units_in_next_layer in zip(layer_sizes[:-1], layer_sizes[1:]):
        weight_matrix = scale * np.random.randn(n_units_in_curr_layer, n_units_in_next_layer)
        bias_vector = scale * np.random.randn(n_units_in_next_layer)
        params.append((weight_matrix, bias_vector))
    return params
