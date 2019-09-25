from datetime import datetime
import dill as pickle
from pathlib import Path
from copy import deepcopy

import numpy as np
from skimage.io import imread

import GPnd
from GPnd import *

from plotting import MAP_Estimator



if __name__=='__main__':
    f_path = Path('chains/2019-09-22T16-01-08_n100000.pkl')
    with open(f_path, 'rb') as f:
        chain = pickle.load(f)

    image_path = Path('data/head.png')
    image = dataLoading.import_image(image_path, size=chain.size)
    data = chain.T(image)
    fbp = chain.T.inv(data)

    print(f_path)
    print(
        'Data Shape: %s\n'%(data.shape,),
        'L_2 Errors: \n',
        '   Filtered Back projections: %s\n'%(np.linalg.norm(fbp-image)/np.product(image.shape),),
        '   MCMC Reconstruction: %s\n'%(np.linalg.norm(chain.reconstruction-image)/np.product(image.shape)),
    )        