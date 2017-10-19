import sys, os

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from src.examples.bayesian_classification_heteroscedastic import main as bayesian_classification_heteroscedastic_main
from src.examples.bayesian_regression_heteroscedastic import main as bayesian_regression_heteroscedastic_main

from src.examples.bayesian_regression_epistemic import main as bayesian_regression_epistemic_main


def test_bayesian_univariate_epistemic():
    sys.argv = ['', '--epochs', '1', '-T', '10']
    bayesian_regression_epistemic_main()
    assert True


def test_bayesian_univariate_aleatoric():
    sys.argv = ['', '--epochs', '1', '-T', '10']
    bayesian_regression_heteroscedastic_main()
    assert True


def test_bayesian_mnist_epistemic_aleatoric():
    sys.argv = ['', '--epochs', '1', '-T', '10', '--train_max', '1000', '--test_max', '100']
    bayesian_classification_heteroscedastic_main()
    assert True
