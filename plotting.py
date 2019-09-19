from datetime import datetime
import dill as pickle
from pathlib import Path
from copy import deepcopy

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

#animations
from matplotlib import animation

import numpy as np
from skimage.io import imread

import GPnd
from GPnd import *

print(mpl.matplotlib_fname())

def plot_result_2d(image, chain, savefig=False):
    
    Cov = chain.Cov
    C = chain.dgp.C
    size = chain.size
    T = chain.T
    data = T(image)
    data += chain.noise * np.random.standard_normal(data.shape)
    fbp = T.inv(data)
    theta = T.theta

    description = 'Size: %s, Depth: %s, #Samples: %s, Computation Time: %ss'%(size, chain.depth, len(chain.samples), int(chain.t_delta))
    if savefig:
        plt.switch_backend('pdf')
    title_fontsize = 'x-small'
    
    #Layout
    fig = plt.figure(dpi=300, tight_layout=True)
    fig.set_size_inches(8.27, 11.69, forward=True)
    
    plt.figtext(0.02, .99, description, fontsize = 'small')

    ax = np.zeros(12, dtype=object)
    gs = fig.add_gridspec(5, 3, height_ratios=[3,2,3,2,2])
    ax[0] = fig.add_subplot(gs[0, 0])
    ax[1] = fig.add_subplot(gs[0, 1])
    ax[2] = fig.add_subplot(gs[0, 2])
    ax[3] = fig.add_subplot(gs[1, :])
    ax[4] = fig.add_subplot(gs[2, 0])
    ax[5] = fig.add_subplot(gs[2, 1])
    ax[6] = fig.add_subplot(gs[2, 2])
    ax[7] = fig.add_subplot(gs[3, 0:2])
    ax[8] = fig.add_subplot(gs[3, 2:])
    ax[9] = fig.add_subplot(gs[4, 0])
    ax[10] = fig.add_subplot(gs[4, 1])
    ax[11] = fig.add_subplot(gs[4, 2])

    #Truth
    im = ax[0].imshow(image)
    ax[0].set_title('Truth', fontsize = title_fontsize)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='xx-small')

    #MCMC Reconstruction
    im = ax[1].imshow(chain.reconstruction)
    ax[1].set_title('MCMC Reconstruction (Sample Mean)', fontsize = title_fontsize)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='xx-small')

    #FBP Reconstruction
    im = ax[2].imshow(fbp)
    ax[2].set_title('FBP Reconstruction', fontsize = title_fontsize)
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='xx-small')
    
    #Data
    for i, d in enumerate(data.T):
        ax[3].plot(d, label = '%s°'%int(theta[i]))
    if len(data.T) <=10:
        ax[3].legend(loc='upper right', fontsize='xx-small')
    ax[3].set_title('Data (Sinogram)', fontsize = title_fontsize)

    #Maximum Acceptance Pobability
    u, err = MAP_Estimator(chain, image)
    im = ax[4].imshow(u)
    ax[4].set_title('MAP sample, phi=%s'%err, fontsize = title_fontsize)
    divider = make_axes_locatable(ax[4])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='xx-small')

    #Chain Variance
    im = ax[5].imshow(chain.var)
    ax[5].set_title('MCMC Sample Variance', fontsize = title_fontsize)
    divider = make_axes_locatable(ax[5])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='xx-small')

    #Spatial Error
    im = ax[6].imshow(np.abs(chain.reconstruction-image))
    ax[6].set_title('Reconstruction Error', fontsize = title_fontsize)
    divider = make_axes_locatable(ax[6])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='xx-small')

    # Random Projection
    v = np.random.standard_normal(chain.shape)
    v /= np.linalg.norm(v)/np.prod(chain.shape)
    ax[7].plot([np.tensordot(s, v)/np.prod(chain.shape) for s in chain.samples])
    ax[7].set_title('Projection of Chain onto a Random Unit-Vector', fontsize = title_fontsize)
    
    # Step Size
    ax[8].plot(chain.betas)
    ax[8].set_title('Jump Size', fontsize = title_fontsize)

    #Operator Slices
    im = ax[9].imshow(C.C[0,0], norm=colors.LogNorm(vmin=C.C[0,0].min()+.00001, vmax=C.C[0,0].max()))
    ax[9].set_title('Covariance Operator, i=1, j=1', fontsize = title_fontsize)
    divider = make_axes_locatable(ax[9])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='xx-small')

    im = ax[10].imshow(C.C[C.shape[0]//2,C.shape[1]//2], norm=colors.LogNorm(vmin=C.C[C.shape[0]//2,C.shape[1]//2].min()+.00001, vmax=C.C[C.shape[0]//2,C.shape[1]//2].max()))
    ax[10].set_title('Covariance Operator, i=%s, j=%s'%(C.shape[0]//2,C.shape[1]//2), fontsize = title_fontsize)
    divider = make_axes_locatable(ax[10])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='xx-small')

    im = ax[11].imshow(C.C[C.shape[0]-1, C.shape[1]-1], norm=colors.LogNorm(vmin=C.C[C.shape[0]-1, C.shape[1]-1].min()+.00001, vmax=C.C[C.shape[0]-1, C.shape[1]-1].max()))
    ax[11].set_title('Covariance Operator, i=%s, j=%s'%(C.shape[0], C.shape[1]), fontsize = title_fontsize)
    divider = make_axes_locatable(ax[11])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize='xx-small')
    
    for x in ax.flat:
        for tick in x.xaxis.get_major_ticks():
            tick.label.set_fontsize('xx-small')
        for tick in x.yaxis.get_major_ticks():
            tick.label.set_fontsize('xx-small')   
    return fig

def MAP_Estimator(chain, image):
    T  = chain.T
    data = T(image)
    y_hat = [T(u) for u in chain.samples]
    errs = [chain.phi(y,data) for y in y_hat]
    return chain.samples[np.argmin(errs)], np.round(np.min(errs), 4)

def animate_chain(chain, video_path, savefig=False, speed:int = 50):
    #setup the figure and axes
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[4,2])
    
    ax = []    
    #Samples
    ax.append(fig.add_subplot(gs[0, 0]))
    img = ax[0].imshow(chain.samples[0])
    ax[0].set_title('Current Sample')
    
    #Samples mean
    ax.append(fig.add_subplot(gs[0, 1]))
    mean = ax[1].imshow(chain.samples[0])
    ax[1].set_title('Mean up to Current Sample')
    
    #Beta
    ax.append(
        fig.add_subplot(
            gs[1, :],
            xlim=(0, 1000), 
            ylim=(0,1))
    )
    idx = np.arange(0, len(chain.betas))
    line, = ax[2].plot(idx[:0], chain.betas[:0])
    ax[2].set_title('Jump Size beta')
    
    def _init():
        img.set_data(chain.samples[0])
        mean.set_data(chain.samples[0])
        line.set_data(idx[:0], chain.betas[:0])
        return img, line, mean,
    
    def _animate(i):
        #update xlimits on graph
        xmin, xmax = ax[2].get_xlim()
        if i*speed >= xmax:
            ax[2].set_xlim(xmin, 2*xmax)
            ax[2].figure.canvas.draw()
        
        img.set_data(chain.samples[i*speed])
        mean.set_data(np.mean(chain.samples[:(i*speed)+1], axis=0))
        line.set_data(idx[:i*speed], chain.betas[:i*speed])
        return img, line, mean,
    
    anim = animation.FuncAnimation(
        fig,
        _animate, 
        init_func=_init,
        frames=len(chain.samples)//speed, 
        blit=True,
        interval=1
    )
    if savefig:
        anim.save(str(video_path.resolve()), fps=30, bitrate=4000, extra_args=['-vcodec', 'libx264'])
    else:
        plt.show()


if __name__=='__main__':
    f_path = Path('chains/2019-09-19T13-21-18_n200000.pkl')
    with open(f_path, 'rb') as f:
        chain = pickle.load(f)

    image_path = Path('data/phantom.png')
    image = dataLoading.import_image(image_path, size=chain.size)

    video_path = Path("/".join(['results', 'animations', f_path.stem]) + ".mp4")
    animate_chain(chain, video_path, savefig=True)



    # fig = plot_result_2d(image, chain, savefig=True)
    # fig_name = 'results_%s.pdf'%f_path.name.replace('.pkl','')
    # plt.savefig('results\\'+fig_name, papertype='a4', orientation='portrait', dpi=300)
