from datetime import datetime
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
from skimage.io import imread

import GPnd
from GPnd import *

print(mpl.matplotlib_fname())

def plot_result_2d(image, chain, savefig=False):
    
    C = chain.C
    size = chain.size
    T = chain.T
    data = T(image)
    fbp = T.inv(data)
    theta = T.theta
    
    description = 'Size: %s, #Samples: %s, Computation Time: %ss'%(size, len(chain.samples), int(chain.t_delta))
    if savefig:
        plt.switch_backend('pdf')
    title_fontsize = 'x-small'
    fig = plt.figure(dpi=300, tight_layout=True)
    fig.set_size_inches(8.27, 11.69, forward=True)
    
    plt.figtext(0.02, .99, description, fontsize = 'small')

    ax = np.zeros(9, dtype=object)
    gs = fig.add_gridspec(5, 3, height_ratios=[3,2,3,2,2])
    ax[0] = fig.add_subplot(gs[0, 0])
    ax[1] = fig.add_subplot(gs[0, 1])
    ax[2] = fig.add_subplot(gs[0, 2])
    ax[3] = fig.add_subplot(gs[1, :])
    ax[4] = fig.add_subplot(gs[2, 0])
    ax[5] = fig.add_subplot(gs[2, 1])
    ax[6] = fig.add_subplot(gs[2, 2])
    ax[7] = fig.add_subplot(gs[3, :])
    ax[8] = fig.add_subplot(gs[4, :])

    ax[0].imshow(C.C[C.shape[0]//2,C.shape[1]//2])
    ax[0].set_title('Slice through Covariance Operator', fontsize = title_fontsize)
    # ax[0].set_aspect('equal')

    ax[1].imshow(chain.samples[-1][0])
    ax[1].set_title('Last Sample', fontsize = title_fontsize)
    # ax[1].set_aspect('equal')

    ax[2].imshow(image)
    ax[2].set_title('Truth', fontsize = title_fontsize)
    # ax[2].set_aspect('equal')
    
    for i, d in enumerate(data.T):
        ax[3].plot(d, label = '%s°'%int(theta[i]))
    ax[3].legend(loc='upper right')
    ax[3].set_title('Measurement (Sinogram)', fontsize = title_fontsize)
    # ax[3].set_aspect('auto', 'box')
    
    ax[4].imshow(chain.reconstruction)
    ax[4].set_title('MCMC Reconstruction (Sample Mean)', fontsize = title_fontsize)
    # ax[4].set_aspect('equal')
    
    ax[5].imshow(chain.var)
    ax[5].set_title('MCMC Sample Variance', fontsize = title_fontsize)
    # # ax[5].set_aspect('equal')

    ax[6].imshow(fbp)
    ax[6].set_title('FBP Reconstruction', fontsize = title_fontsize)
    # ax[5].set_aspect('equal')

    ax[7].plot([s[1] for s in chain.samples])
    ax[7].set_title('Heightscale', fontsize = title_fontsize)
    # ax[6].set_aspect('auto', 'box')
    
    ax[8].plot([b[0] for b in chain.betas], label='Layer 1')
    ax[8].plot([b[1] for b in chain.betas], label='Layer 0')
    ax[8].legend(loc='upper right')
    ax[8].set_title('Jump Size', fontsize = title_fontsize)
    # ax[7].set_aspect('auto', 'box')


    for x in ax.flat:
        for tick in x.xaxis.get_major_ticks():
            tick.label.set_fontsize('xx-small')
        for tick in x.yaxis.get_major_ticks():
            tick.label.set_fontsize('xx-small')   
    return fig

if __name__=='__main__':
    f_path = Path('2019-06-27T10-38-37_n100000.pkl')
    with open(f_path, 'rb') as f:
        chain = pickle.load(f)

    im_path = Path('data\phantom.png')
    image = imread(im_path, as_gray=True)

    fig = plot_result_2d(image, chain, savefig=True)
    fig_name = 'results_%s.pdf'%f_path.name.replace('.pkl','')
    plt.savefig(fig_name, papertype='a4', orientation='portrait', dpi=300)