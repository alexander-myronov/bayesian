import argparse
import time
from functools import partial

import numpy as np
from keras import Input
from keras.callbacks import Callback
from keras.layers import Dropout, Dense
from keras.regularizers import l2

from src.bayesian.bayesian_dropout_model import BayesianDropoutModel
from src.bayesian.metrics import bayesian_std, predictive_entropy


def standardize(X):
    if type(X) == list:
        return X
    else:
        return [X]


def monkeypatch_model_predict_stochastic(model, num_input=0, num_output=0):
    if model is None:
        return
    if not hasattr(model, 'predict_stochastic'):
        model.predict_stochastic = partial(BayesianDropoutModel.predict_stochastic, model)
        model._predict_stochastic = BayesianDropoutModel.create_predict_stochastic(model,
                                                                                   num_input=num_input,
                                                                                   num_output=num_output)


class DrawUnivariateEpistemicUncertaintyCallback(Callback):
    def __init__(self, x, y, T, p_dropout, weight_decay, x_train, y_train, fig, ax):
        super().__init__()
        self.x = x
        self.y = y
        self.T = T
        self.ax = ax
        self.fig = fig
        self.p_dropout = p_dropout
        self.weight_decay = weight_decay
        self.x_train = x_train
        self.y_train = y_train

    def on_epoch_begin(self, epoch, logs={}):
        monkeypatch_model_predict_stochastic(self.model)
        prob = np.array([self.model.predict_stochastic(self.x, batch_size=500, verbose=0)
                         for _ in range(self.T)])
        prob_mean = np.mean(prob, 0)
        prob_std = bayesian_std(prob,
                                l=1,
                                p_dropout=self.p_dropout,
                                weight_decay=self.weight_decay,
                                N=len(self.x_train))

        # fig.clf()
        self.ax.cla()

        self.ax.plot(self.x, self.y, 'bo')
        self.ax.plot(self.x_train, self.y_train, 'ro')
        self.ax.plot(self.x, prob_mean, 'r--')
        for n_std, alpha in zip([0.5, 1, 1.5], [0.3, 0.2, 0.1]):
            self.ax.fill_between(self.x.ravel(),
                                 (prob_mean - n_std * prob_std).ravel(),
                                 (prob_mean + n_std * prob_std).ravel(), alpha=alpha, color='red')
        self.ax.set_xlim(self.x.min() - 0.5, self.x.max() + 0.5)
        self.ax.set_ylim(self.y.min() - 0.5, self.y.max() + 0.5)
        self.fig.canvas.draw()
        time.sleep(0.01)


class DrawUnivariateEpistemicAleatoricUncertaintyCallback(Callback):
    def __init__(self, x, y, T, p_dropout, weight_decay, x_train, y_train, fig, ax):
        super().__init__()
        self.x = x
        self.y = y
        self.T = T
        self.ax = ax
        self.fig = fig
        self.p_dropout = p_dropout
        self.weight_decay = weight_decay
        self.x_train = x_train
        self.y_train = y_train

    def on_epoch_begin(self, epoch, logs={}):
        monkeypatch_model_predict_stochastic(self.model)
        pred = np.array([self.model.predict_stochastic(self.x, batch_size=500, verbose=0)
                         for _ in range(self.T)])
        y_pred = pred[:, :, 0]

        prob_mean = np.mean(y_pred, 0)
        epistemic_prob_std = bayesian_std(y_pred, l=1,
                                          p_dropout=self.p_dropout,
                                          weight_decay=self.weight_decay,
                                          N=len(self.x_train))

        aleatoric_log_var = np.mean(pred[:, :, 1], 0)
        aleatoric_var = np.exp(aleatoric_log_var)
        aleatoric_std = np.sqrt(aleatoric_var)

        self.ax.cla()

        self.ax.plot(self.x, self.y, 'bo', label='f(x)')
        self.ax.plot(self.x_train, self.y_train, 'ro', label='training set')
        self.ax.plot(self.x, prob_mean, 'r--', label='MC prediction mean')

        for n_std, alpha in zip([0.5, 1, 1.5], [0.3, 0.2, 0.1]):
            self.ax.fill_between(self.x.ravel(),
                                 (prob_mean - n_std * epistemic_prob_std).ravel(),
                                 (prob_mean + n_std * epistemic_prob_std).ravel(), alpha=alpha, color='red',
                                 label='epistemic uncertainty % std' % n_std)

            self.ax.fill_between(self.x.ravel(),
                                 (prob_mean - n_std * aleatoric_std).ravel(),
                                 (prob_mean + n_std * aleatoric_std).ravel(), alpha=alpha, color='green',
                                 label='aleatoric uncertainty % std' % n_std)

        self.ax.set_xlim(self.x.min() - 0.5, self.x.max() + 0.5)
        # self.ax.set_ylim(y_pred.min(), y_pred.max())
        self.ax.set_ylim(self.y.min() - 0.5, self.y.max() + 0.5)
        self.ax.legend()
        self.fig.canvas.draw()
        time.sleep(0.001)


def univariate_function_example(x):
    return (1 - 2.5 * np.power(0.6 * x, 2)) * np.exp(-np.power(0.6 * x, 2))


def univariate_variance_example(x):
    v = np.copy(x)
    v[v < 0] = 1
    v = np.log(v) * 0.8
    return v


def univariate_regression_dataset(x=None,
                                  f=univariate_function_example,
                                  var_f=univariate_variance_example):
    if x is None:
        x = np.random.uniform(-10, 10, 30).reshape(-1, 1)
    y = f(x) + np.random.normal(0, 0.1, size=(len(x), 1)) * var_f(x)
    x = x.reshape(-1, 1)
    return x, y


def build_net_architecture(n_features, hidden_neurons, activation, drouput, weight_decay):
    inp = Input(shape=(n_features,))
    hidden_neurons = standardize(hidden_neurons)
    activation = standardize(activation)
    drouput = standardize(drouput)
    weight_decay = standardize(weight_decay)

    layer = inp
    for i, l_neurons in enumerate(hidden_neurons):
        l_dropout = drouput[min(len(drouput) - 1, i)]
        l_activation = activation[min(len(activation) - 1, i)]
        l_weight_decay = weight_decay[min(len(weight_decay) - 1, i)]

        layer = Dense(l_neurons,
                      activation=l_activation,
                      kernel_regularizer=l2(l_weight_decay) if l_weight_decay and l_weight_decay > 0 else None,
                      bias_regularizer=l2(l_weight_decay) if l_weight_decay and l_weight_decay > 0 else None)(layer)
        if l_dropout and l_dropout > 0:
            layer = Dropout(l_dropout)(layer)
    return [inp], [layer]


def create_arg_parser():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='sgd learning rate')
    parser.add_argument('--learning_rate_decay', type=float, default=1e-5,
                        help='sgd learning rate')
    parser.add_argument('--dropout', type=float, default=0.05,
                        help='probability of dropping')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='weight decay')
    parser.add_argument('-v', '--verbose', help='increase output verbosity',
                        action='store_true')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('-T', type=int, default=50,
                        help='number of MC simulations for dropout')
    return parser


def remove_options(parser, options):
    for option in options:
        for action in parser._actions:
            if vars(action)['option_strings'][0] == option:
                parser._handle_conflict_resolve(None, [(option, action)])
                break


# model - the trained classifier(C classes)
# where the last layer applies softmax
# X_data - a list of input data(size N)
# T - the number of monte carlo simulations to run
def classification_MC_prediction_with_epistemic_uncertainty(model, X_data, T, num_output=None,
                                                            batch_size=500,
                                                            verbose=False):
    # shape: (T, N, C)
    predictions = []
    for _ in range(T):
        out = model.predict_stochastic(X_data, batch_size=batch_size, verbose=verbose)
        if num_output is not None:
            out = out[num_output]
        predictions += [out]
    predictions = np.array(predictions)
    # print(predictions.min())

    # shape: (N, C)
    prediction_probabilities = np.mean(predictions, axis=0)

    # print(prediction_probabilities.shape)
    # shape: (N)
    prediction_uncertainty = predictive_entropy(predictions)
    return (prediction_probabilities, prediction_uncertainty)


def classification_prediction_with_aleatoric_uncertainty(model, X_data, batch_size=500, verbose=False):
    out = model.predict(X_data, batch_size=batch_size)
    assert isinstance(out, list) and len(out) >= 2

    probs = out[0]
    pred_var = out[1]

    # pred = pred_var[:, :-1]
    var = pred_var[:, -1]
    return probs, np.sqrt(var)
