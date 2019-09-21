'''
This is the NICE code accompanying my bachelor's thesis. I was able to cut out a lot of unnecessary stuff.
Ideas:
    1. Adapt beta based on acceptance probability every 100 iterations - Done
    2. Adapt the length scale parameters during the iteration, maybe for the first 10,000 iterations, then fix or reduce update rate - Scrapped DGPs all the way
    3. Implement more layers, but I need a powerful server for that and need to switch to an ONB representation of the function space
    4. Performance Metrics - Need to decide with Tim which ones he wants to see

Next Steps:
    1. Modularize the code more, so functionality is easily added - done
    2. Implement idea Nr.2 - done
    3. Implement Idea Nr.4 - done
    5. Document Code - done
    
After Thesis (maybe): 
    1. Implement different Priors - Besov is an option!; Might need to switch to basis representations of the space then.
    2. Use Sparse tensors
    4. Use a coarser tiling for deeper layers
'''
import copy
from pathlib import Path
import time
from datetime import datetime
import dill as pickle

import numpy as np
from matplotlib import pyplot as plt
#from scipy.stats import lognorm

from utils.operators import mOp, CovOp, mDevice, RadonTransform, DGP
from utils.logPdfs import lognorm_pdf, exp_pdf, log_acceptance_ratio
from utils import dataLoading
import plotting

def update_beta(beta, acc_prob, target):
	'small function for updating beta'
	beta += .005*(acc_prob-target)
	return np.clip(beta, 2**(-15), 1-2**(-15))

class wpCN(object):
	def F(self, x):
		return np.exp(x)

	# def DGP(self, x, xi):
	# 	u = self.C(xi[0])
	# 	for i in range(1, self.depth):
	# 		nu = self.C(xi[i])
	# 		u = self.L(u[i-1], nu, x)
	# 	return u
 
	def phi(self, x, y):
		return np.sum((x-y)**2) * self.dx / self.noise

	def __init__(self, ndim, size, noise, Covariance: CovOp , T: mDevice):
		self.ndim = ndim
		self.size = size
		self.shape = (size,)*ndim
		
		self.depth = 1
		self.noise = noise
		self.Cov = Covariance
		self.Cov.update_tensor()
		self.T = T

		self.dx = 1/T.len
		self.xi = []
		for i in range(self.depth):
			self.xi.append(np.random.standard_normal(self.shape))
		
		self.dgp = DGP(self.Cov, np.exp, self.depth)
		self.u = self.dgp.sample(self.xi)
		self.m = T(self.u)

		self.Temperature = 1
		self.beta = .5
		self.samples = []
		self.probs = []
		self.betas  = []
  	
	def infer(self, data):
		#base layer
		xi_hat = [np.sqrt(1 - self.beta**2) * self.xi[i] + self.beta * np.random.standard_normal(self.shape) for i in range(self.depth)]
		u_hat = self.dgp.sample(xi_hat)
		m_hat = self.T(u_hat)
		logProb = min(self.phi(self.m, data) - self.phi(m_hat, data), 0) * self.Temperature
		if np.random.rand() <= np.exp(logProb):
			self.xi = xi_hat
			self.u = u_hat
			self.m = m_hat
		self.beta = update_beta(self.beta, np.exp(logProb), .23)
	

	def sample(self, data, niter = 10000):
		t_start = time.time()
		i=0
		while i <= niter:
			i+=1
			self.infer(data)
			self.samples.append(self.u)
			self.betas.append(self.beta)
			# Kill chain if beta leaves a sensible region
			if i%100==0:
				if self.beta < 0.005:
					self.Temperature /= 2
					self.samples = []
					self.betas = []
					self.beta = .5
					print('Temperature decreased to: ', self.Temperature)
					i = 0
				if self.beta > 0.995:
					self.Temperature *= 2
					self.samples = []
					self.betas = []
					self.beta = .5
					print('Temperature increased to: ', self.Temperature)
					i = 0
				print(i, 'Beta: ', self.beta)
		t_end = time.time()
		self.t_delta = (t_end - t_start)
		print('#Samples:%s, Time: %ss'%(niter, self.t_delta))
		self.reconstruction = np.mean(self.samples, axis = 0)
		self.var = np.var(self.samples, axis = 0)


if __name__=='__main__':	

	image_path = Path('data/phantom.png')
	size = 48
	image = dataLoading.import_image(image_path, size=size)

	ndim = image.ndim
	shape = (size,)*ndim

	C = CovOp(ndim, size, sigma=.1, ro=.01)

	T = RadonTransform(ndim, size, np.linspace(0, 180, 10))

	noise = .0001
	
	data = T(image)
	data += noise * np.random.standard_normal(data.shape)
	fbp = T.inv(data)
	print('Shape of Data:', data.shape)

	chain = wpCN(ndim, size, noise, C, T)

	n_iter = 100000
	chain.sample(data, n_iter)

	f_name = '%s_n%s.pkl'%(datetime.now().replace(microsecond=0).isoformat().replace(':','-'),str(n_iter))
	with open('chains/' + f_name, 'wb') as f:
		pickle.dump(chain, f)

	fig = plotting.plot_result_2d(image, chain, savefig=True)
	fig_name = 'results_%s.pdf'%f_name.replace('.pkl','')
	plt.savefig('results/'+ fig_name, papertype='a4', orientation='portrait', dpi=300)