from keras.layers import Dense, Concatenate
from keras.optimizers import SGD
from keras.regularizers import l2

from bayesian.bayesian_dropout_model import BayesianDropoutModel
from matplotlib import pyplot as plt
import numpy as np

from bayesian.objectives import bayesian_mean_squared_error
from bayesian.utils import DrawUnivariateEpistemicUncertaintyCallback, \
    DrawUnivariateEpistemicAleatoricUncertaintyCallback
from utils import univariate_function_example, univariate_regression_dataset, build_net_architecture, create_arg_parser, \
    remove_options


def create_model(X, dropout=0.1, weight_decay=0.01, **kwargs):
    inputs, outputs = build_net_architecture(X.shape[1],
                                             hidden_neurons=[25, 25],
                                             activation=['relu', 'relu'],
                                             weight_decay=weight_decay,
                                             drouput=dropout)
    last_layer = outputs[0]
    output = Dense(1,
                   activation='linear',
                   kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(last_layer)

    var = Dense(1, activation='linear',
                kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(last_layer)

    final_output = Concatenate()([output, var])
    model = BayesianDropoutModel(inputs=inputs, outputs=[final_output])
    optimiser = 'adam'
    model.compile(loss=bayesian_mean_squared_error, optimizer=optimiser)
    return model


def main():
    parser = create_arg_parser()
    remove_options(parser, ['--learning_rate', '--learning_rate_decay'])
    args = parser.parse_args()
    X, y = univariate_regression_dataset()
    if args.verbose:
        print(args.__dict__)
        print('Building model...')

    model = create_model(X, **args.__dict__)
    if args.verbose:
        print('model architecture: (layer name, input shape, output shape)')
        for l in model.layers:
            print(type(l), l.input_shape, l.output_shape)
        plt.close('all')
        plt.ion()
        fig, ax = plt.subplots()
        fig.set_size_inches(12, 8)
        x_draw = np.linspace(X.min() - 0.5, X.max() + 0.5, 100).reshape(-1, 1)
        _, y_draw = univariate_regression_dataset(x_draw)
        # plt.plot(x_draw, y_draw)
        # plt.show(block=True)
        draw_callback = DrawUnivariateEpistemicAleatoricUncertaintyCallback(
            x=x_draw,
            y=y_draw,
            T=args.T,
            ax=ax,
            fig=fig,
            x_train=X,
            y_train=y,
            p_dropout=args.dropout,
            weight_decay=args.weight_decay)
        callbacks = [draw_callback]
    else:
        callbacks = []

    model.fit(X, y, batch_size=args.batch_size, epochs=args.epochs,
              callbacks=callbacks, verbose=1 if args.verbose else 0)


if __name__ == '__main__':
    main()
