# Bayesian Image Reconstruction using the Metropolis-Hastings Algorithm

## Install

1. Have Python 3.7
2. ``pip install virtualenv``
3. ``virtualenv env``
4. Activate the environment (System dependent; on Windows execute ``env\Scripts\activate.bat``)
5. ``pip install -r requirements.txt``
6. ``python GPnd.py``
7. Enjoy report

## Structure

1. ``GPnd.py`` is the main script. Here the class ``wpCN`` runs the markov chain. Currently, the assumed Gaussian noise is a multiple of the identity.

2. Operators provides the classes for the covariance operator, the forward operator and the *deep Gaussian process*.
3. ``plotting.py`` produces a report from a pickled wpCN object and stores it in results.
