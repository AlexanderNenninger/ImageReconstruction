import numpy as np
from skimage.transform import radon, iradon

from copy import deepcopy
'''
Define Operators. mOp is for taking measurements, CovOp is for mapping to the right function space. 
'''
class mOp(object):
    'F([0,1]^ndim) -> R'
    def f(self, x):
        return np.exp(-x**2/self.sigma**2/2)

    def __init__(self, ndim, size, mean, sigma=1):
        self.tensor_cached = False
        self.ndim = ndim
        self.shape = (size,)*ndim
        self.size = size
        self.sigma = sigma
        self.mean = mean
        self.F = np.zeros(self.shape)

    def __call__(self, x):
        if self.ndim == 0:
           return self.F*x
        elif self.ndim == 1:
            return np.dot(self.F, x)/self.size
        return np.tensordot(self.F, x)/self.size**self.ndim
    
    def update_tensor(self):
        shape = np.array(self.shape)
        mean = np.array(self.mean)
        it = np.nditer(self.F, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = np.array(it.multi_index)
            d = np.linalg.norm(idx/shape - mean)
            it[0] = self.f(d)
            it.iternext()
        self.F/=self.F.sum()
        self.tensor_cached = True



class CovOp(object):
    'F[0,1]^ndim->C[0,1]^ndim'
    def f(self, r):
        return np.exp(-r/self.ro)#(1 + np.sqrt(3)*r / self.ro) * np.exp(-np.sqrt(3) * r / self.ro)
    
    def dist(self, x,y):
        return np.sum((x-y)**2)
    
    def __init__(self, ndim, size, sigma=1, ro=1):
        self.tensor_cached = False
        self.inverse_cached = False
        self.ndim = ndim
        self.size = size
        self.shape = (size,)*2**ndim
        self.C = np.zeros(self.shape)
        self.Inv = np.zeros(self.shape)
        self.ro = ro * size
        self.sigma = sigma      
	
    def __call__(self, x):
        if not self.tensor_cached:
            self.update_tensor
        if self.ndim == 0:
            return self.sigma * self.C * x
        elif self.ndim == 1:
            return self.sigma * np.dot(self.C, x)
        return self.sigma * np.tensordot(self.C, x)
    
    def update_tensor(self):
        it = np.nditer(self.C, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = np.array(it.multi_index)
            d = self.dist(idx[:idx.shape[0]//2], idx[idx.shape[0]//2:])
            it[0] = self.f(d)
            it.iternext()
        self.tensor_cached = True
        #missing cholesky decomposition
    
    def update_inverse(self):
        if self.ndim==1:
            self.Inv = np.linalg.inv(self.C)
        elif self.ndim>1:
            self.Inv = np.linalg.tensorinv(self.C)
        else:
            self.Inv = 1/self.C
        self.inverse_cached = True
    
    def inv(self, x):
        if self.ndim == 0:
            return self.Inv * x / self.sigma
        elif self.ndim == 1:
            return np.dot(self.Inv, x) / self.sigma
        return np.tensordot(self.Inv, x) / self.sigma
    
    def set_f(self ,f):
        self.inverse_cached = False
        self.tensor_cached = False
        self.f = f
        
    def set_dist(self, dist):
        self.inverse_cached = False
        self.tensor_cached = False
        self.dist = dist
        
class DGP(object):
    def __init__(self, CovOp, F, depth):
        self.depth = depth
        self.CovOp = CovOp
        self.C = deepcopy(self.CovOp)
        self.F = F
        self.sigma = CovOp.sigma
        self.u = []
    def sample(self, xi):
        self.u = []
        self.u.append(self.CovOp(xi[0]))
        for i in range(1, self.depth):
            Sigma = self.F(self.u[i-1])
            Q = lambda x,y: np.sqrt(np.inner(x-y, 2*(x-y)/(Sigma[tuple(x)] + Sigma[tuple(y)])))
            self.C.set_dist(Q)
            self.C.update_tensor()
            self.C.sigma = self.CovOp.sigma/self.C.C.ravel()[0]
            self.u.append(self.C(xi[i]))
        return self.u[-1]


class mDevice(object):
    'Measuring Device - Holds list of mOps'
    def __init__(self, functionals: list):
        self.functionals = functionals
        self.len = len(self.functionals)
    def __call__(self, x):    
        'Makes measuring multiple values easy'
        if type(self.functionals) == mOp:
            return self.functionals(x)
        m = [f(x) for f in self.functionals]
        return np.array(m)

class RadonTransform(object):
    def __init__(self, ndim, size, theta):
        self.theta = theta
        self.len = radon(np.ones((size,)*ndim), self.theta, circle=False).size
    def __call__(self, x):
        return radon(x, theta=self.theta, circle=False)
    def inv(self, x):
        return iradon(x, self.theta, circle=False)

if __name__=='__main__':
    import matplotlib.pyplot as plt
    size = 40
    ndim = 2
    depth = 3
        
    F = lambda x: np.exp(x)
    
    Cov = CovOp(ndim, size, 1, .2)
    Cov.update_tensor()
    dgp = DGP(Cov, F, depth)   

    xi = [np.random.standard_normal((size,)*ndim) for i in range(depth)] 
    samples = dgp.sample(xi)

    fig, ax = plt.subplots(ncols = depth)
    for i, s in enumerate(samples):
        im = ax[i].imshow(s)
    #fig.colorbar(im)
    plt.show()
    