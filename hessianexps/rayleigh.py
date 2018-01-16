from autograd import numpy as np
from scipy.optimize import fmin_l_bfgs_b


def get_top_eigensystem(model, input_dim, k=1, tolerance=1e-5):
    """Compute the largest eigenvalue and vectors by the Rayleigh quotient minimization."""
    return get_extremal_eigensystem(model, input_dim, direction='top', k=k, tolerance=tolerance)


def get_bottom_eigensystem(model, input_dim, k=1, tolerance=1e-5):
    """Compute the smallest eigenvalue and vectors by the Rayleigh quotient minimization."""
    return get_extremal_eigensystem(model, input_dim, direction='bottom', k=k, tolerance=tolerance)


def get_extremal_eigensystem(model, input_dim, direction='top', k=1, tolerance=1e-5):
    """Compute the smallest/largest eigenvalue and vectors by the Rayleigh quotient minimization."""
    def maximizer(v):
        # Note: i, eigenvectors and model are defined in the outer function get_extremal_eigensystem().
        quot, grad_quot = _largest_rayleigh_quotient_in_subspace(v, i, eigenvectors, model)
        if direction == 'top':
            return -quot, -grad_quot
        else:
            return quot, grad_quot

    eigenvalues = []
    eigenvectors = []
    for i in range(k):
        v0 = np.random.normal(0, 1, len(model.params_flat))
        v_max, lambda_max, _ = fmin_l_bfgs_b(maximizer, v0, pgtol=tolerance)
        if direction == 'top':
            lambda_max *= -1
        eigenvalues.append(lambda_max)
        eigenvectors.append(v_max)
    return eigenvalues, eigenvectors


def _largest_rayleigh_quotient_in_subspace(v, i, eigvecs, model):
    """Compute largest Rayleigh quotient and its gradient of Hessian in subspace.
    
    Subspace is the vector space of Hessian minus the eigenspace spanned by first i eigenvectors.
    
    :param eigvecs: the list of first i eigenvectors
    """
    if i != 0:
        matrix = np.concatenate((eigvecs, [v]), axis=0)
        q, r = np.linalg.qr(matrix.transpose())
        v = q.transpose()[i] * r[i][i]
    return model.rayleigh_quotient(v), model.grad_rayleigh(v)
