"""
bayesian objectives for neural network training
from 
https://arxiv.org/pdf/1703.04977.pdf 
and 
https://medium.com/towards-data-science/building-a-bayesian-deep-learning-classifier-ece1845bc09

N - number of examples
C - number of classes
"""

import numpy as np
import keras.backend as K


def noise_var(pred, std, num_classes):
    if K.backend() == 'tensorflow':
        from tensorflow.contrib import distributions
        dist_var = distributions.Normal(loc=K.zeros_like(std), scale=std)
        return dist_var
    elif K.backend() == 'theano':
        from theano.sandbox.rng_mrg import MRG_RandomStreams
        theano_rng = MRG_RandomStreams(42)
        dist_var = theano_rng.normal(avg=0, std=K.tile(std, (num_classes, 1)).T, size=pred.shape)
        return dist_var
    raise NotImplementedError('noise_var: unknown backend %s' % K.backend())


def noise_sample(dist_var, num_classes):
    if K.backend() == 'tensorflow':
        return K.transpose(dist_var.sample(num_classes))
    elif K.backend() == 'theano':
        pass  # already a random stream
    raise NotImplementedError('noise_sample: unknown backend %s' % K.backend())


def bayesian_categorical_crossentropy_original(T, num_classes):
    """
    function to construct
    original loss function from https://arxiv.org/pdf/1703.04977.pdf
    the model is expected to return num_classes+1 outputs (the last one is predicted variance).
    :param T: number of monte-carlo integration iterations
    :param num_classes: number of classes
    :return: a function accepting 2 parameters:
        y_true (N, C) - true labels, one-hot encoded
        pred_var (N, C+1) - concatenation of logits and predicted variance
    can be used as keras optimization objective
    the general idea:
    1. sample gaussian noise located at 0 with predicted variance
    2. distort logits with the noise
    3. calculate cross-categorical entropy
    4. repeat T times
    5. average loss is returned
    """

    def bayesian_categorical_crossentropy_internal(true, pred_var):
        # shape: (N,)
        std = K.sqrt(pred_var[:, num_classes])
        # shape: (N,)
        variance = pred_var[:, num_classes:]
        variance_depressor = K.exp(variance) - K.ones_like(variance)
        # shape: (N, C)
        pred = pred_var[:, 0:num_classes]
        # shape: (N,)
        undistorted_loss = K.categorical_crossentropy(pred, true, from_logits=True)
        # shape: (T,)
        iterable = K.variable(np.ones(T))
        # dist = K.random_normal(mean=0, stddev=K.tile(std, (num_classes, )), shape=pred.shape)
        dist = noise_var(pred, std, num_classes)
        # distributions.Normal(loc=K.zeros_like(std), scale=std)
        monte_carlo_results = K.map_fn(gaussian_categorical_crossentropy_original(
            true, pred, dist, num_classes),
            iterable,
            name='monte_carlo_results')

        variance_loss = K.mean(monte_carlo_results, axis=0)  # * undistorted_loss

        # return undistorted_loss
        return variance_loss  # + undistorted_loss + variance_depressor

    return bayesian_categorical_crossentropy_internal


def gaussian_categorical_crossentropy_original(true, pred, dist, num_classes):
    """
    for a single monte carlo simulation,
    calculate categorical cross-entropy of
    predicted logit values plus gaussian
    noise vs true values.

    :param true: true values. Shape: (N, C)
    :param pred: predicted logit values. Shape: (N, C)
    :param dist: normally distributed noise. Shape: (N, C)
    :return: a function to compute distorted cross-categorical entropy (I'm not completely sure its cross-categorical
    entropy, but the difference between distorted_loss and distorted_loss2 is around 1e-8 on MNIST, so I'm assuming
    it is
    """

    def map_fn(i):
        std_samples = noise_sample(dist, num_classes)
        distorted_logits = pred + std_samples
        distorted_loss = -K.sum(distorted_logits * true, axis=1) + K.log(
            K.sum(K.exp(distorted_logits), axis=1))
        # distorted_loss2 = K.categorical_crossentropy(distorted_logits, true, from_logits=True)
        return distorted_loss

    return map_fn


def bayesian_categorical_crossentropy_elu(T, num_classes):
    """
    function to construct
    Bayesian categorical cross entropy - a loss function from
    https://medium.com/towards-data-science/building-a-bayesian-deep-learning-classifier-ece1845bc09
    :param T: number of monte-carlo integration iterations
    :param num_classes: number of classes
    :return: a function accepting 2 parameters:
        y_true (N, C) - true labels, one-hot encoded
        pred_var (N, C+1) - concatenation of logits and predicted variance
    can be used as keras optimization objective
    General idea:
    1. sample gaussian noise located at 0 with predicted variance
    2. distort logits with the noise
    3. calculate cross-categorical entropy - i.e. distorted loss
    4. calculate the difference between undistorted loss (normal categorical cross-entropy) and distorted
    5. pass the difference through elu function (read the blog post for explanation)
    5. this is the weight for every example (i.e. undistorted loss is multiplied by elu(diff))
    6. undistorted loss is added to ensure positivity
    7. variance depressor (exp(variance)) is added to prevent the model from predicting huge variance for every example
    """

    def bayesian_categorical_crossentropy_internal_elu(true, pred_var):
        # shape: (N,)
        std = K.sqrt(pred_var[:, num_classes])
        # shape: (N,)
        variance = pred_var[:, num_classes]
        variance_depressor = K.exp(variance) - K.ones_like(variance)
        # shape: (N, C)
        pred = pred_var[:, 0:num_classes]
        # shape: (N,)
        undistorted_loss = K.categorical_crossentropy(pred, true, from_logits=True)
        # shape: (T,)
        iterable = K.variable(np.ones(T))
        dist = noise_var(pred, std, num_classes)
        monte_carlo_results = K.map_fn(
            gaussian_categorical_crossentropy_elu(true, pred, dist, undistorted_loss, num_classes),
            iterable,
            name='monte_carlo_results')

        variance_loss = K.mean(monte_carlo_results, axis=0) * undistorted_loss
        return variance_loss + undistorted_loss + variance_depressor

    return bayesian_categorical_crossentropy_internal_elu


def gaussian_categorical_crossentropy_elu(true, pred, dist, undistorted_loss, num_classes):
    '''
    for a single monte carlo simulation, 
    calculate the elu of the difference between
        categorical cross-entropy of 
        predicted logit values plus gaussian 
        noise vs true values
    and
        undistorted categorical cross-entropy
    :param true: true values. Shape: (N, C)
    :param pred: predicted logit values. Shape: (N, C)
    :param dist: normally distributed noise. Shape: (N, C)
    :return: a function to compute the loss
    '''

    def map_fn(i):
        std_samples = noise_sample(dist, num_classes)
        distorted_loss = K.categorical_crossentropy(pred + std_samples, true, from_logits=True)
        diff = undistorted_loss - distorted_loss
        return -K.elu(diff)

    return map_fn


def bayesian_mean_squared_error(y_true, pred_var):
    """
    regression loss function from https://arxiv.org/pdf/1703.04977.pdf
    the model is expected to return 2 (regression mean and log-variance.
    :return: a function accepting 2 parameters:
        y_true (N, 1) - true labels, one-hot encoded
        pred_var (N, 2) - concatenation of logits and predicted variance
    :param y_true: (N, 1) true values
    :param pred_var: (N, 2) concatenation of predicted mean and log-variance
    :return: each square residual is weighted by 1 / exp(log-variance), plus log-variance itself (to prevent
    the model from predicting huge variance for every sample).
    General idea:
    For every sample, the model can reduce cost in 2 ways:
    1. predict better mean, so that the squared residual is lower
    2. predict higher variance, so that the weight of the squared residual is lower.
    """
    log_var = pred_var[:, 1]
    pred = pred_var[:, 0]
    se = K.pow(K.transpose(y_true) - pred, 2) * K.exp(-log_var) + log_var
    return K.mean(se)
