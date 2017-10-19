This is a repository for educational resources, examples, and utils related to Bayesian machine learning. 


# Structure
-  `docs/bayesian-resources.ipynb`:  a Jupyter notebook with publications, tutorials and blog posts on Bayesian learning
-  `src/bayesian`: a collection of utils
  -  `src/bayesian/callbacks.py`: Keras callback for testing MC dropout during training
  -  `src/bayesian/metrics.py`: 'some uncertainty metrics for bayesian models'
  -  `src/bayesian/objectives.py`: 'optimization objectives for heteroscedatic uncertainty'
  -  `src/src/bayesian/droupout_bayesian_model.py`: a class derived from Keras's Model with predict_stochastic method (MC droupout)
  -  `src/bayesian/utils.py`: various helper functions
- `src/examples`: training a bayesian model for univariate regression problem (`bayesian-regr.ipynb`) and MNIST classification (`bayesian-classification.ipynb`). Some simple usage examples, details in Examples section
- `tests/` pytest integral tests (runs examples with minimal parameters)


# Usage

## callbacks.py

ModelTest is is a validation callback, can be used as a callback to Keras fit function to compare MC dropout predictions accuracy with regular predictions.
Can be used on a regular Keras model, but monkey-patches it to have `predict_stochastic method` (from BayesianDropoutModel)
Usage shown in `python src/examples/callback.py`

## metrics.py

Selected prediction quality metrics for classification and regression

## objectives.py

-  `bayesian_mean_squared_error` for regression tasks (from https://arxiv.org/pdf/1703.04977.pdf)
-  `bayesian_categorical_crossentropy_original` for classification tasks (from https://arxiv.org/pdf/1703.04977.pdf)
-  `bayesian_categorical_crossentropy_elu` for classification tasks (weighted metric from https://medium.com/towards-data-science/building-a-bayesian-deep-learning-classifier-ece1845bc09)

## droupout_bayesian_model.py

Keras Model subclass that automatically creates the predict_stochastic method if needed (for MC dropout).
Usage shown in scripts and notebooks in `python src/examples/`

# Examples

*Hint: every example has a built it help, type `python examples/*.py -h`.*
*Note that examples should be run as submodules, because the imported code is in `bayesian` directory*

-  `python3 -m src.examples.callback --epochs=100 -T=50 -v` univariate regression with epistemic uncertainty using MC dropout
-  `python3 -m src.examples.bayesian_regression_epistemic --epochs=100 -T=50 -v` same thing, but using BayesianDropoutModel instead of callback.
-  `python3 -m src.examples.bayesian_regression_heteroscedastic --epochs=100 -T=50 -v` aleatoric uncertainty using extra output for predicted variance (https://arxiv.org/pdf/1703.04977.pdf)
-  `python3 -m src.examples.bayesian_classification_heteroscedastic --epochs=5 -T=50 -v` aleatoric uncertainty for classification using extra output for predicted variance used to generate noise in logits space (https://arxiv.org/pdf/1703.04977.pdf)

