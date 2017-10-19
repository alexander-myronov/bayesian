
# A collection of resources on Bayesian ML in R&D

Feel free to contribute. If you add a link, please include the author and title and possibly a short summary and conslusions with references to your links

### Prerequisites ###

1. general ML (supervised learning, classification, regression, optimization of the cost function, neural nets)
1. Gaussian process regression (kriging). I recommend Neil Lawrence:
    https://www.youtube.com/watch?v=ewJ3AxKclOg

### Motivation
1. "In May 2016 there was the first fatality from an assisted
driving system, caused by the perception system confusing
the white side of a trailer for bright sky (NHTSA, 2017).
In a second recent example, an image classification system
erroneously  identified  two  African  Americans  as  gorillas
(Guynn, 2015),  raising concerns of racial discrimination"
1. https://www.youtube.com/watch?v=rekIC0G_bYA G. Hinton explains why we need Bayesian learning
1. https://alexgkendall.com/computer_vision/bayesian_deep_learning_for_safe_ai/ A. Kendall explains uncertainty

Conclusions:
- (1) our beliefs about complexity of the models should not be affected by the amount of data available - overfitting does not exist in Bayesian setting
- (2) 2 types of uncertainty:
  - Epistemic uncertainty captures our ignorance about which model generated our collected data - model uncertainty.
    Can be reduced given more data
  - Aleatoric uncertainty captures our uncertainty with respect to information which our data cannot explain. 
    This  could  be  for  example  sensor  noise  or  motion  noise,  resulting  in  uncertainty  which  cannot  be  reduced  even  if  more  data  were  to  be  collected.
    Aleatoric uncertainty  can  further  be  categorized  into homoscedastic uncertainty, uncertainty which stays constant for different inputs, and heteroscedastic uncertainty.
    Heteroscedastic  uncertainty  depends  on  the  inputs  to  the  model,  with some  inputs  potentially  having  more  noisy  outputs  than others. Heteroscedatic true data distribution has different variance in different regions of the    domain. Our model should be less confident in regions of higher variance. 

### Overview

1. Z. Ghahramani "Probabilistic machine learning and artificial intelligence"
1. http://bayesiandeeplearning.org/slides/nips16bayesdeep.pdf Z. Ghahramani "A history of Bayesian neural networks"
1. http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf Yarin Gal "Uncertainty in deep learning"
1. http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html more stuff from Yarin Gal, interactive demos
1. https://arxiv.org/pdf/1703.04977.pdf A. Kendall, Y. Gal "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"

Conclusions: 
- (1, 2, 3) Uncertainty can be predicted
- (1, 2, 3) The goal is to estimate the posterior distribution of parameters given data
- (1, 2, 3) This posterior distribution is intractable to compute, it can be approximated using Variational Inference
- (3) Training a nn with dropout and weight decay is analogous to variational inference - such net is already Bayesian
- (3) Epistemic uncertainty can be estimated using a technique called Monte-Carlo dropout, when predictions are repeated using different dropout masks at testing times. This gives us not a single prediction, but rather a distribution
- (4) Aleatoric uncertainty can be estimated using a separate output in a neural net to predict variance and a modified cost function. Intuition: 
  1. imagine regression model which simultaneously predicts mean and variance
  1. cost function is weighted MSE
  1. the weight is dependent on the variance, lower when predicted variance is high
  1. such model can deal with a single prediction error in 2 ways: either reduce the error or increase the predicted variance (meaning reduce the weight of this example). 
  1. variance regularization is required, otherwise it will be optimal for the model to assign high variance to all examples (thus giving very small weights). 


### Tutorials/code/libs

1. https://github.com/yaringal/BayesianRNN callbacks.py has an implementation of MC dropout for epistemic uncertainty
1. https://github.com/yaringal/HeteroscedasticDropoutUncertainty explanation of aleatoric uncertainty (technically h. uncertainty is a part of a. uncertainty, read more in motivation section number 2); javascript implementation
1. https://medium.com/towards-data-science/building-a-bayesian-deep-learning-classifier-ece1845bc09 Implementation of aleatoric and epistemic variance prediction for classification
1. edwardlib.org/tutorials/ edward - python lib to perform Variational Inference
1. https://alpha-i.co/blog/MNIST-for-ML-beginners-The-Bayesian-Way.html Tutorial on using edward lib on MNIST
1. https://www.youtube.com/watch?v=I09QVNrUS3Q A. Rowan talks about using MC dropout and compares it to VI in edward

Conclusions: 
- 2 practical ways of doing Bayesian learning in python
    1. edward - more mathematically rigorous (4), easier to use (5), possibly worse results due to certain independence assumptions in VI algorthms (6), can utilize GPU (1)
    2. MC dropout (1) + variance prediction (2, 3) in Keras - optimization objective are sometimes different among papers, need to write own code (this will be hopefully done and available in this repo), possibly better results (6), can utilize GPU

