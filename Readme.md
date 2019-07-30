# Bayesian Image Reconstruction using the Metropolis-Hastings Algorithm

## Install

1. Have Python 3.7
2. Open a commad promt.
3. Navigate to the directroy, You want this project to reside in.
4. ``pip install virtualenv``
5. ``virtualenv env``
6. Activate the environment (System dependent; on Windows execute ``env\Scripts\activate.bat``)
7. ``pip install -r requirements.txt``
8. ``python GPnd.py``
9. Enjoy report

## Structure

1. ``GPnd.py`` is the main script. Here the class ``wpCN`` runs the markov chain. Currently, the assumed Gaussian noise is a multiple of the identity.

2. Operators provides the classes for the covariance operator, the forward operator and the *deep Gaussian process*.
3. ``plotting.py`` produces a report from a pickled wpCN object and stores it in results.

![Example Report][report]

[report]: https://github.com/AlexanderNenninger/wpCN_Image_Reconstruction/blob/master/results/results_2019-07-29T08-13-20_n30000.pdf
