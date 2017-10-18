"""
functions to compute various (epistemic) uncertainty measures for classification and regression tasks
input shape: N samples, C classes (=1 for regression), T Monte-Carlo simulations
"""

from scipy.stats import mode
import numpy as np


def variation_ratio(y_pred):
    """
    variation ratio from Y. Gal's "Uncertainty in deep learning"
    :param y_pred: (N, T, C)  
    :return: (N) percent of most likely class predicted in T simulations
    """
    prob_pred = y_pred.argmax(axis=-1)
    m = mode(prob_pred, 0)
    return (1 - (m.count / float(prob_pred.shape[0]))).ravel()


def bayesian_std(y_pred, N, p_dropout, weight_decay, l=1):
    """
    epistemic uncertainty in regression, adjusted to satisfy KL condition 
        (equivalence of dropout with weight decay with Variational Inference
    Y. Gal "Uncertainty in deep learning"
    :param y_pred: (N, T)
    :param l: length scale, refer to the paper, 1 is ok
    :param p_dropout: probability of dropping (!) of a unit
    :param weight_decay: l2 weight decay used
    :param N: number of training points
    :return: (N) bayesian standard deviation for every sample
    """
    prob_std = np.std(y_pred, 0)
    tau = l ** 2 * (1 - p_dropout) / (2 * N * weight_decay)
    prob_std += tau ** -1
    return prob_std


def _predictive_entropy_1(y_pred):
    """
    entropy of a single prediction
    :param y_pred: (C)
    :return: entropy
    """
    return -1 * np.sum(np.log(y_pred) * y_pred)


def predictive_entropy(y_pred):
    """
    entropy as a measure of epistemic uncertainty
    :param y_pred: (N, C, T)
    :return: (N) predictive entropy for averaged predictions for every sample
    """
    prediction_probabilities = np.mean(y_pred, axis=0)
    prediction_variances = np.apply_along_axis(_predictive_entropy_1, axis=1, arr=prediction_probabilities)
    return prediction_variances


def montecarlo_uncertainty(model, X, metric, T, batch_size=1, verbose=False):
    """
    helper function to calculate uncertainty for given metric
    :param model: 
    :param X: data
    :param metric: a function accepting an array of predictions of shape (N, C, T)
    :param T: number of Monte-Carlo simulations
    :param batch_size: batch size for the model
    :param verbose: verbosity of model's predict function
    :return: 
    """
    predictions = np.array([model.predict_stochastic(X, batch_size=batch_size, verbose=verbose) for _ in range(T)])
    return metric(predictions)
