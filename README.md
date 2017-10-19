This is a repository for educational resources, examples, and utils related to Bayesian machine learning. 


# Structure
-  `docs`:  an Jupyter notebook with publications, tutorials and blog posts on Bayesian learning
-  `bayesian`: a collection of utils
  -  `bayesian/callbacks.py`: Keras callback for testing MC dropout during training
  -  `bayesian/metrics.py`: 'some uncertainty metrics for bayesian models'
  -  `bayesian/objectives.py`: 'optimization objectives for heteroscedatic uncertainty'
  -  `bayesian/droupout_bayesian_model.py`: a class derived from Keras's Model with predict_stochastic method (MC droupout)
  -  `bayesian/utils.py`: various helper functions
- `examples`: training a bayesian model for univariate regression problem (`bayesian-regr.ipynb`) and MNIST classification (`bayesian-classification.ipynb`). Some simple usage examples, details in Examples section


# Usage

## callbacks.py

ModelTest is is a validation callback, can be used as a callback to Keras fit function to compare MC dropout predictions accuracy with regular predictions.
Can be used on a regular Keras model, but monkey-patches it to have `predict_stochastic method` (from BayesianDropoutModel)
Usage shown in `python examples\callback.py`

## metrics.py

Selected prediction quality metrics for classification and regression

## objectives.py

-  `bayesian_mean_squared_error` for regression tasks (from https://arxiv.org/pdf/1703.04977.pdf)
-  `bayesian_categorical_crossentropy_original` for classification tasks (from https://arxiv.org/pdf/1703.04977.pdf)
-  `bayesian_categorical_crossentropy_elu` for classification tasks (weighted metric from https://medium.com/towards-data-science/building-a-bayesian-deep-learning-classifier-ece1845bc09)

## droupout_bayesian_model.py

Keras Model subclass that automatically creates the predict_stochastic method if needed (for MC dropout).
Usage shown in `python examples\bayesian_*.py` scripts and notebooks

# Examples

*Hint: every example has a built it help, type `python examples/*.py -h`.*

-  `examples/callback.py --epochs=100 -T=50 -v` univariate regression with epistemic uncertainty using MC dropout
-  `examples/bayesian_regression_epistemic.py --epochs=100 -T=50 -v` same thing, but using BayesianDropoutModel instead of callback.
-  `examples/bayesian_regression_heteroscedastic.py --epochs=100 -T=50 -v` aleatoric uncertainty using extra output for predicted variance (https://arxiv.org/pdf/1703.04977.pdf)
-  `examples/bayesian_classification_heteroscedastic.py --epochs=5 -T=50 -v` aleatoric uncertainty for classification using extra output for predicted variance used to generate noise in logits space (https://arxiv.org/pdf/1703.04977.pdf)

