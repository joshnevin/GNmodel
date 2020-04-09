# Implementation of heteroscedastic GPR 

#  imports

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
import time
from numpy.linalg import cholesky
from numpy import transpose as T
from numpy.linalg import inv, det, lstsq, solve
from numpy import matmul as mul
from scipy.optimize import minimize
# fit to Goldberg dataset 
# =============================================================================
# numpoints = 100
# x = np.linspace(0,1,numpoints)
# sd = np.linspace(0.5,1.5,numpoints)
# y = np.zeros(numpoints)
# n = np.size(x)
# for i in range(numpoints):
#     y[i] = 2*np.sin(2*np.pi*x[i]) + np.random.normal(0, sd[i])
# y = y.reshape(-1,1)
# x = x.reshape(-1,1)
# =============================================================================
numpoints = 100
x = np.linspace(0,1,numpoints)
n = np.size(x)
sd = 0.5
y = np.zeros(numpoints)
for i in range(numpoints):
    y[i] = 2*np.sin(2*np.pi*x[i]) + np.random.normal(0, sd)
y = y.reshape(-1,1)
x = x.reshape(-1,1)
plt.plot(x,y,'+')
plt.show()
# %% SKLearn for comparison 
#kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e3)) + W(1.0, (1e-5, 1e5))
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e3)) 
print("Initial kernel: %s" % kernel)
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 9, normalize_y=False, alpha=np.amax(sd))
#gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 9, normalize_y=False)
start = time.time()
gpr.fit(x, y)
print("Optimised kernel: %s" % gpr.kernel_)
end = time.time()
print("GPR fitting took " + str(end - start) + "s")

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
# %% my implementation of GPR 
# initialise model and optimise hyperparameters via maximum likelihood
sig = np.amax(sd) # set constant noise level as the maximum noise in dataset 

def RBF(X1, X2, k1, k2):  
    kern = np.zeros((np.shape(X1)[0], np.shape(X2)[0]))
    arg = np.zeros((np.shape(X1)[0], np.shape(X2)[0]))
    for i in range(np.shape(X1)[0]):
        for j in range(np.shape(X2)[0]):    
            kern[j][i] = k1*np.exp(  -((np.linalg.norm(X1[i] - X2[j]))**2)/(2*k2)  )
            arg[j][i] = (np.linalg.norm(X1[i] - X2[j]))**2/(2*k2) 
    return kern, arg

def RBF2(xs, X, k1, k2): # verified to be consistent with MATLAB gradfunc
    kern = np.zeros((np.shape(X)[0], 1))
    for i in range(np.shape(X)[0]):
        kern[i] = (k1**2)*np.exp(  -(np.linalg.norm(xs - X[i]))**2/(2*(k2**2))  ) 
    return kern

xtest = x[0:10]
test = RBF2(xtest,xtest,1,1)
def lml(params):
    print(params)
    [k1, k2] = params
    Ky = RBF(x,x,k1,k2)[0] + (sig**2)*np.identity(n) # calculate initial kernel with noise 
    return -(-0.5*mul(mul(T(y),inv(Ky)), y) - 0.5*np.log((det(Ky))) - 0.5*n*np.log(2*np.pi)) # marginal likelihood - (5.8)

def lmlg(params):
    k1, k2 = params
    Ky = RBF(x,x,k1,k2)[0] + (sig**2)*np.identity(n) # calculate initial kernel with noise 
    al = mul(inv(Ky),y)
    dKdk1 = RBF(x,x,k1,k2)[0]*(1/k1)
    #dKdk2 = mul(RBF(x,x,k1,k2)[0],RBF(x,x,k1,k2)[1]*(1/k2))
    dKdk2 = RBF(x,x,k1,k2)[0]*RBF(x,x,k1,k2)[1]*(1/k2)
    lmlg1 = -(0.5*np.trace(mul(mul(al,T(al)) - inv(Ky), dKdk1)))
    lmlg2 = -(0.5*np.trace(mul(mul(al,T(al)) - inv(Ky), dKdk2)))
    return np.ndarray((2,), buffer=np.array([lmlg1,lmlg2]), dtype = float)
# %% run optimiser
numh = 2
num_restarts = 0
k1i = np.random.uniform(1.0,10.0,num_restarts+1) # set initial values for the hyperparameters 
k2i = np.random.uniform(1.0,10.0,num_restarts+1)
results = []
for i in range(num_restarts+1):
    ki = np.ndarray((numh,), buffer=np.array([k1i[i],k2i[i]]), dtype = float)
    res = minimize(lml,ki,method = 'L-BFGS-B',jac=lmlg,bounds = ((1e-5,1e5),(1e-5,1e5)))
    if res.success:
        results.append(res.x)
        print("Success")
    else:
        raise ValueError(res.message)
# %% generate GPR model using R+W algorithm 2.1
k1 = results[0][0]
k2 = results[0][1]

def GPRfit(xtest,k1,k2):
    Kst = RBF2(xtest,x,k1,k2)[0]
    Ky = RBF(x,x,k1,k2)[0] + (sig**2)*np.identity(n)
    L = cholesky(Ky)
    al = solve(T(L), solve(L,y))
    fmst = mul(T(Kst),al)
    v = solve(L,Kst)
    varfmst = RBF(xtest,xtest,k1,k2)[0] - mul(T(v),v)
    lmlopt = -0.5*mul(T(y),al) - np.trace(np.log(L)) - 0.5*n*np.log(2*np.pi)
    return fmst, varfmst, lmlopt
  
fmst, varfmst, lmlopt = GPRfit(xtest,k1,k2)

# %% plotting
plt.plot(x,y,'+', label = 'data')
plt.plot(x,fmst, label = 'Josh')
plt.plot(x,ystar, label = 'SK Learn')
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.savefig('gprsklearnvsjosh.pdf', dpi=200)
plt.show()




















