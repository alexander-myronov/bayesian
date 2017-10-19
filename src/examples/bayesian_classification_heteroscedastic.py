import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Concatenate, Activation
from keras.metrics import categorical_accuracy
from keras.regularizers import l2
from keras.utils import to_categorical
from matplotlib import pyplot as plt

from src.bayesian.bayesian_dropout_model import BayesianDropoutModel
from src.bayesian.callbacks import ModelTest
from src.bayesian.objectives import bayesian_categorical_crossentropy_original
from src.bayesian.utils import build_net_architecture, \
    create_arg_parser, remove_options
from src.bayesian.utils import classification_prediction_with_aleatoric_uncertainty, \
    classification_MC_prediction_with_epistemic_uncertainty


def create_model(n_features, n_classes, dropout=0.5, weight_decay=0.01, **kwargs):
    inputs, outputs = build_net_architecture(n_features,
                                             hidden_neurons=[200, 100],
                                             activation=['relu', 'relu'],
                                             weight_decay=weight_decay,
                                             drouput=dropout)
    last_layer = outputs[0]
    logits = Dense(n_classes,
                   kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(last_layer)
    # output = Activation('softmax')(logits)
    variance_pre = Dense(1)(last_layer)
    variance = Activation('softplus', name='variance')(variance_pre)
    logits_variance = Concatenate(name='logits_variance')([logits, variance])
    softmax_output = Activation('softmax', name='softmax_output')(logits)

    model = BayesianDropoutModel(inputs, [softmax_output, logits_variance])

    model.compile(
        optimizer='adam',
        loss={
            'logits_variance': bayesian_categorical_crossentropy_original(100, 10),
            'softmax_output': 'categorical_crossentropy'
        },
        metrics={'softmax_output': categorical_accuracy},
        loss_weights={'logits_variance': 1,
                      'softmax_output': 0
                      }
    )

    return model


def main():
    parser = create_arg_parser()
    remove_options(parser, ['--learning_rate', '--learning_rate_decay'])
    parser.add_argument('--train_max', type=int, default=10000,
                        help='number of training examples')
    parser.add_argument('--test_max', type=int, default=10000,
                        help='number of training examples')
    args = parser.parse_args()
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train[:min(args.train_max, len(X_train))]
    y_train = y_train[:min(args.train_max, len(X_train))]
    X_test = X_train[:min(args.test_max, len(X_test))]
    y_test = y_train[:min(args.test_max, len(X_test))]

    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    if args.verbose:
        print(args.__dict__)
        print('Building model...')
        mc_callback = ModelTest(X_test, y_test, T=50, test_every_X_epochs=1, loss='categorical', verbose=0,
                                num_output=0)
        callbacks = [mc_callback]
    else:
        callbacks = []
    model = create_model(X_train.shape[1], y_train.shape[1], **args.__dict__)


    if args.verbose:
        print('model architecture: (layer name, input shape, output shape)')
        for l in model.layers:
            print(type(l), l.input_shape, l.output_shape)

    model.fit(X_train, [y_train] * 2, batch_size=args.batch_size, epochs=args.epochs,
              validation_data=(X_test, [y_test] * 2),
              callbacks=callbacks, verbose=1 if args.verbose else 0)

    if args.verbose:
        mc_pred, epi = classification_MC_prediction_with_epistemic_uncertainty(model, X_test, args.T)
        pred, ale = classification_prediction_with_aleatoric_uncertainty(model, X_test)
        k = 25
        p = epi / epi.mean() + ale / ale.mean()
        p = p / p.sum()

        fig, axs = plt.subplots(ncols=int(k / 5), nrows=5)
        axs = axs.ravel()
        for i, i_test in enumerate(np.random.choice(len(y_test), k, p=p)):
            img = X_test[i_test].reshape(28, 28) * 255
            axs[i].imshow(img, cmap=plt.cm.get_cmap('Greys'))
            axs[i].set_title('%d - %d, ale=%.3f, epi=%.3f' % (y_test[i_test].argmax(),
                                                              mc_pred[i_test].argmax(),
                                                              ale[i_test],
                                                              epi[i_test]))
        fig.set_size_inches(15, 15)
        plt.show()


if __name__ == '__main__':
    main()
