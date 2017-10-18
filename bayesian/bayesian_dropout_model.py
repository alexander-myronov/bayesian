from warnings import warn

import numpy as np
from keras import backend as K
from keras.layers import Dropout, Dense
from keras.models import Model


class BayesianDropoutModel(Model):
    def __init__(self, *args, **kwargs):
        super(BayesianDropoutModel, self).__init__(*args, **kwargs)
        self.num_output = kwargs.pop('num_output', 0)
        self.num_input = kwargs.pop('num_output', 0)
        self._predict_stochastic = None

    def validate(self):
        had_dropout = False
        for l in self.layers:
            if isinstance(l, Dropout):
                had_dropout = True
            elif isinstance(l, Dense):
                pass
            else:
                raise Exception('Only dense and dropout layers are allowed, got ' + str(type(l)))

        if not had_dropout:
            warn('No dropout layer. The behaviour of predict_stochastic is the same as predict')

    def predict_stochastic(self, X, batch_size=1, verbose=False):
        """
        Generate output predictions for the input samples
        batch by batch, using stochastic forward passes. If
        dropout is used at training, during prediction network
        units will be dropped at random as well. This procedure
        can be used for MC dropout (see [ModelTest callbacks](callbacks.md)).

        # Arguments
            X: the input data, as a numpy array.
            batch_size: integer.
            verbose: verbosity mode, 0 or 1.

        # Returns
            A numpy array of predictions.

        # References
            - [Dropout: A simple way to prevent neural networks from overfitting](http://jmlr.org/papers/v15/srivastava14a.html)
            - [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](http://arxiv.org/abs/1506.02142)
        """
        if self._predict_stochastic is None:
            # self.validate()
            self._predict_stochastic = K.function([self.inputs[self.num_input]],
                                                  [self.outputs[self.num_output]],
                                                  givens={K.learning_phase(): np.uint8(1)})
        if not isinstance(X, list):
            X = [X]
        return self._predict_loop(self._predict_stochastic, X, batch_size=batch_size,
                                  verbose=verbose)
