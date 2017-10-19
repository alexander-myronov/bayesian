import numpy as np
from keras.engine import Model
from keras.layers import Dense
from keras.optimizers import SGD
from keras.regularizers import l2
from matplotlib import pyplot as plt

from src.bayesian.utils import DrawUnivariateEpistemicUncertaintyCallback
from src.bayesian.callbacks import ModelTest
from src.bayesian.utils import univariate_function_example, univariate_regression_dataset, build_net_architecture, \
    create_arg_parser


def create_model(X, dropout=0.1, weight_decay=0.01, learning_rate=0.01, learning_rate_decay=1e-5, **kwargs):
    inputs, outputs = build_net_architecture(X.shape[1],
                                             hidden_neurons=[50, 25],
                                             activation=['relu', 'sigmoid'],
                                             weight_decay=weight_decay,
                                             drouput=dropout)
    last_layer = outputs[0]
    output = Dense(1,
                   activation='linear',
                   kernel_regularizer=l2(weight_decay) if weight_decay and weight_decay > 0 else None,
                   bias_regularizer=l2(weight_decay) if weight_decay and weight_decay > 0 else None)(last_layer)
    model = Model(inputs=inputs, outputs=[output])
    optimiser = SGD(lr=learning_rate, decay=learning_rate_decay)
    model.compile(loss='mean_squared_error', optimizer=optimiser)
    return model


def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    X, y = univariate_regression_dataset()
    if args.verbose:
        print(args.__dict__)
        print('Building model...')

    model = create_model(X, **args.__dict__)
    mc_callback = ModelTest(X, y, T=10, test_every_X_epochs=1, loss='euclidean', verbose=0)
    if args.verbose:
        print('model architecture: (layer name, input shape, output shape)')
        for l in model.layers:
            print(type(l), l.input_shape, l.output_shape)
        plt.close('all')
        plt.ion()
        fig, ax = plt.subplots()

        x_draw = np.linspace(X.min() - 0.5, X.max() + 0.5, 100).reshape(-1, 1)
        y_draw = univariate_function_example(x_draw)
        draw_callback = DrawUnivariateEpistemicUncertaintyCallback(
            x=x_draw,
            y=y_draw,
            T=args.T,
            ax=ax,
            fig=fig,
            x_train=X,
            y_train=y,
            p_dropout=args.dropout,
            weight_decay=args.weight_decay)
        callbacks = [draw_callback, mc_callback]
    else:
        callbacks = [mc_callback]

    model.fit(X, y, batch_size=args.batch_size, epochs=args.epochs,
              callbacks=callbacks, verbose=1 if args.verbose else 0)


if __name__ == '__main__':
    main()
