# Implementation of heteroscedastic GPR 

#  imports

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
import time
from numpy.linalg import cholesky
from numpy import transpose as T
from numpy.linalg import inv, det, lstsq, solve, norm
from numpy import matmul as mul, exp
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist, squareform
# fit to Goldberg dataset 
# =============================================================================
numpoints = 100
x = np.linspace(0,1,numpoints)
sd = np.linspace(0.5,1.5,numpoints)
y = np.zeros(numpoints)
n = np.size(x)
for i in range(numpoints):
    y[i] = 2*np.sin(2*np.pi*x[i]) + np.random.normal(0, sd[i])
y = y.reshape(-1,1)
x = x.reshape(-1,1)
# =============================================================================
# 
# x = np.linspace(0,1,numpoints)
# n = np.size(x)
# sd = 0.5
# y = np.zeros(numpoints)
# for i in range(numpoints):
#     y[i] = 2*np.sin(2*np.pi*x[i]) + np.random.normal(0, sd)
# y = y.reshape(-1,1)
# x = x.reshape(-1,1)
# =============================================================================
plt.plot(x,y,'+')
plt.show()
# %% ========================= SKLearn for comparison =========================
#kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e3)) + W(1.0, (1e-5, 1e5))
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e3)) 
print("Initial kernel: %s" % kernel)
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 0, normalize_y=False, alpha=np.amax(sd))
#gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 9, normalize_y=False)
start = time.time()
gpr.fit(x, y)
print("Optimised kernel: %s" % gpr.kernel_)
end = time.time()
print("SKLearn fitting took " + str(end - start) + "s")

hyperparameter = np.exp(gpr.kernel_.theta)
alpha = hyperparameter[0]**0.5 # k1
rho = hyperparameter[1]**0.5 # k2
#sigmakernel = hyperparameter[2]**0.5
ystar, sigma = gpr.predict(x, return_std=True )
sigma = np.reshape(sigma,(np.size(sigma), 1))
sigmaave = np.mean(sigma) 
numsig = 2
ystarp = ystar + numsig*sigma
ystarm = ystar - numsig*sigma 
labelp = "GPR mean + " + str(numsig) +  " sigma"
labelm = "GPR mean - " + str(numsig) + " sigma"
labelfill = str(numsig) + " sigma"
plt.plot(x,y,'+', label = 'data')
plt.plot(x,ystar, label = 'GPR mean')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([ystarp,
                        (ystarm)[::-1]]),
         alpha=0.3, fc='g', ec='None', label=labelfill)
plt.title("SKlearn")
plt.legend()
plt.show() 
# %% ========================= my implementation of GPR =========================
# initialise model and optimise hyperparameters via maximum likelihood
sig = np.amax(sd) # set constant noise level as the maximum noise in dataset 
def RBF(X,Y,k1,k2):
    X = np.atleast_2d(X)
    if Y is None:
        dists = pdist(X / k2, metric='sqeuclidean')
        K = np.exp(-.5 * dists)
        # convert from upper-triangular matrix to square matrix
        K = squareform(K)
        np.fill_diagonal(K, 1)
        # return gradient 
        K_gradient = (K * squareform(dists))[:, :, np.newaxis]
        #K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 \  # anisotropic case, see https://github.com/scikit-learn/scikit-learn/blob/95d4f0841d57e8b5f6b2a570312e9d832e69debc/sklearn/gaussian_process/kernels.py
        #            / (k2 ** 2)
        #K_gradient *= K[..., np.newaxis]
        return k1*K, K_gradient
    else:
        dists = cdist(X / k2, Y / k2,metric='sqeuclidean')
        K = np.exp(-.5 * dists)
        return k1*K
# =============================================================================
# def lml(params):
#     #print(params)
#     [k1, k2] = params
#     Ky = RBF(x,x,k1,k2)[0] + (sig**2)*np.identity(n) # calculate initial kernel with noise 
#     return -(-0.5*mul(mul(T(y),inv(Ky)), y) - 0.5*np.log((det(Ky))) - 0.5*n*np.log(2*np.pi)) # marginal likelihood - (5.8)
# def lmlg(params):
#     k1, k2 = params
#     Ky = RBF(x,x,k1,k2)[0] + (sig**2)*np.identity(n) # calculate initial kernel with noise 
#     al = mul(inv(Ky),y)
#     dKdk1 = RBF(x,x,k1,k2)[0]*(1/k1)
#     #dKdk2 = mul(RBF(x,x,k1,k2)[0],RBF(x,x,k1,k2)[1]*(1/k2))
#     dKdk2 = RBF(x,x,k1,k2)[0]*RBF(x,x,k1,k2)[1]*(1/k2)
#     lmlg1 = -(0.5*np.trace(mul(mul(al,T(al)) - inv(Ky), dKdk1)))
#     lmlg2 = -(0.5*np.trace(mul(mul(al,T(al)) - inv(Ky), dKdk2)))
#     return np.ndarray((2,), buffer=np.array([lmlg1,lmlg2]), dtype = float)
# =============================================================================
Kyinv = 0.0
Kf = 0.0
def lml(params):
    #print(params)  # show progress of fit
    [k1, k2] = params
    global Kf
    #Kf = RBF(x,x,k1,k2)[0] 
    Kf = RBF(x,None,k1,k2**0.5)[0]
    Ky = Kf + (sig**2)*np.identity(n) # calculate initial kernel with noise
    global Kyinv
    Kyinv = inv(Ky)
    return -(-0.5*mul(mul(T(y),Kyinv), y) - 0.5*np.log((det(Ky))) - 0.5*n*np.log(2*np.pi)) # marginal likelihood - (5.8)

def lmlg(params):
    k1, k2 = params
    al = mul(Kyinv,y)
    dKdk1 = Kf*(1/k1)
    dKdk2 = RBF(x,None,k1,k2**0.5)[1].reshape(n,n)
    lmlg1 = -(0.5*np.trace(mul(mul(al,T(al)) - Kyinv, dKdk1)))
    lmlg2 = -(0.5*np.trace(mul(mul(al,T(al)) - Kyinv, dKdk2)))
    return np.ndarray((2,), buffer=np.array([lmlg1,lmlg2]), dtype = float)
# %% run optimiser
# =============================================================================
# numh = 2
# num_restarts = 0
# k1i = np.random.uniform(1.0,10.0,num_restarts+1) # set initial values for the hyperparameters 
# k2i = np.random.uniform(1.0,10.0,num_restarts+1)
# results = []
# start = time.time()
# for i in range(num_restarts+1):
#     ki = np.ndarray((numh,), buffer=np.array([k1i[i],k2i[i]]), dtype = float)
#     res = minimize(lml,ki,method = 'L-BFGS-B',jac=lmlg,bounds = ((1e-5,1e5),(1e-5,1e5)))
#     if res.success:
#         results.append(res.x)
#         print("Success")
#     else:
#         raise ValueError(res.message)
# end = time.time()
# print("My GPR fitting took " + str(end - start) + "s")
# =============================================================================
# %% generate GPR model using R+W algorithm 2.1
#k1 = results[0][0]
#k2 = results[0][1]
def GPRfit(xs,k1,k2):
    #Kst = RBF2(xtest,x,k1,k2)[0]
    Ky = RBF(x,None,k1,k2**0.5)[0] + (sig**2)*np.identity(n)
    Ks = RBF(xs, x, k1, k2**0.5)
    Kss = RBF(xs, None, k1, k2)[0]
    L = cholesky(Ky)
    al = solve(T(L), solve(L,y))
    fmst = mul(Ks,al)
    v = solve(L,T(Ks))
    varfmst = Kss - mul(T(v),v)
    lmlopt = -0.5*mul(T(y),al) - np.trace(np.log(L)) - 0.5*n*np.log(2*np.pi)
    return fmst, varfmst, lmlopt
#xs = x[0:10]
#fmst, varfmst, lmlopt = GPRfit(x,k1,k2)
#fmst2, var2, lml2 = GPRfit(xs,k1,k2)
# %% plotting
# =============================================================================
# plt.plot(x,y,'+', label = 'data')
# plt.plot(x,fmst, label = 'Josh')
# plt.plot(x,ystar, label = 'SK Learn')
# plt.legend()
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# plt.savefig('gprsklearnvsjosh.pdf', dpi=200)
# plt.show()
# =============================================================================

# %% ========================= Heteroscedastic GPR implementation =========================
# using algorithm from Most Likely Heteroscedastic Gaussian Process Regression - Kristian Kersting et. al. 
# I will use my implementation only to start with for consistency - could use SKLearn for steps 1-3 for speed
# Step 1: homoscedastic GP on D

numh = 2 # number of hyperparameters in kernel function 
k1is1 = 1.0
k2is1 = 1.0
kis1 = np.ndarray((numh,), buffer=np.array([k1is1,k2is1]), dtype = float)
s1res = minimize(lml,kis1,method = 'L-BFGS-B',jac=lmlg,bounds = ((1e-5,1e5),(1e-5,1e5)))
step1res = []
if s1res.success:
    step1res.append(s1res.x)
    print("Success")
else:
    raise ValueError(step1res.message)
k1s1 = step1res[0][0]
k2s1 = step1res[0][1]
# Step 2: estimate empirical noise levels 





















