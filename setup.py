from setuptools import setup

setup(name='bayesian',
      version='0.1',
      description='A collection of utils and examples on Bayesian learning',
      author='Oleksandr Myronov',
      author_email='oleksandr.myronov@ardigen.com',
      packages=['bayesian'],
      install_requires=['keras', 'numpy', 'scipy'],
      zip_safe=False)