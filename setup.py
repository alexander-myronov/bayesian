from setuptools import setup

setup(name='bayesian',
      version='0.2',
      description='A collection of utils and examples on Bayesian learning',
      author='Oleksandr Myronov',
      author_email='oleksandr.myronov@ardigen.com',
      packages=['src/bayesian'],
      install_requires=['keras', 'numpy', 'scipy'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      zip_safe=False)
