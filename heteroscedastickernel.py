# Implementation of heteroscedastic GPR 

#  imports

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from sklearn.preprocessing import scale
import time
from numpy.linalg import cholesky
from numpy import transpose as T
from numpy.linalg import inv, det, lstsq, solve, norm
from numpy import matmul as mul, exp
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist, squareform
# fit to Goldberg dataset 
#np.savetxt('xhetdata.csv', x, delimiter=',') # to freeze data
#np.savetxt('yhetdata.csv', y, delimiter=',')

# %%
#numpoints = 1000
#x = np.genfromtxt(open("xhetdata.csv", "r"), delimiter=",", dtype =float)
#y = np.genfromtxt(open("yhetdata.csv", "r"), delimiter=",", dtype =float)
x = np.genfromtxt(open("PchdBmopts.csv", "r"), delimiter=",", dtype =float)
y = np.genfromtxt(open("SNRpath1.csv", "r"), delimiter=",", dtype =float)
# =============================================================================
# x = np.linspace(0,1,numpoints)
# sd = np.linspace(0.5,1.5,numpoints)
# y = np.zeros(numpoints)
# n = np.size(x)
# for i in range(numpoints):
#     y[i] = 2*np.sin(2*np.pi*x[i]) + np.random.normal(0, sd[i])
# =============================================================================
y = y.reshape(-1,1)
x = x.reshape(-1,1)
n = np.size(x)
y = scale(y)

# %% ========================= SKLearn for comparison =========================
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e3)) + W(1.0, (1e-5, 1e5))
#kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e3)) 
print("Initial kernel: %s" % kernel)
#gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 0, normalize_y=False, alpha=np.var(y)**0.5)
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 9, normalize_y=False)
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
plt.savefig('ASklearncomp.pdf', dpi=200)
plt.show() 
# %% ========================= my implementation of GPR =========================
# initialise model and optimise hyperparameters via maximum likelihood

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
Kyinv = 0.0
Kf = 0.0
def lml(params,y,sig):
    #print(params)  # show progress of fit
    [k1, k2] = params
    global Kf
    #Kf = RBF(x,x,k1,k2)[0] 
    Kf = RBF(x,None,k1,k2**0.5)[0]
    Ky = Kf + (sig**2)*np.identity(n) # calculate initial kernel with noise
    global Kyinv
    Kyinv = inv(Ky)
    return -(-0.5*mul(mul(T(y),Kyinv), y) - 0.5*np.log((det(Ky))) - 0.5*n*np.log(2*np.pi)) # marginal likelihood - (5.8)

def lmlg(params,y,sig):
    k1, k2 = params
    al = mul(Kyinv,y)
    dKdk1 = Kf*(1/k1)
    dKdk2 = RBF(x,None,k1,k2**0.5)[1].reshape(n,n)
    lmlg1 = -(0.5*np.trace(mul(mul(al,T(al)) - Kyinv, dKdk1)))
    lmlg2 = -(0.5*np.trace(mul(mul(al,T(al)) - Kyinv, dKdk2)))
    return np.ndarray((2,), buffer=np.array([lmlg1,lmlg2]), dtype = float)

def lmlgtest(params,y,sig):
    k1, k2 = params
    al = mul(Kyinv,y)
    dKdk1 = Kf*(1/k1)
    dKdk2 = RBF(x,None,k1,k2**0.5)[1].reshape(n,n)
    lmlg1 = -(0.5*np.trace(mul(mul(al,T(al)) - Kyinv, dKdk1)))
    lmlg2 = -(0.5*np.trace(mul(mul(al,T(al)) - Kyinv, dKdk2)))
    return np.ndarray((2,), buffer=np.array([lmlg1,lmlg2]), dtype = float)

# heteroscedastic versions of functions 
Kyinvh = 0.0
Kfh =  0.0 
def lmlh(params,y,R):
    #print(params)  # show progress of fit
    [k1, k2] = params
    global Kfh
    #Kf = RBF(x,x,k1,k2)[0] 
    Kfh = RBF(x,None,k1,k2**0.5)[0]
    Ky = Kfh + R # calculate initial kernel with noise
    global Kyinvh
    Kyinvh = inv(Ky)
    return -(-0.5*mul(mul(T(y),Kyinvh), y) - 0.5*np.log((det(Ky))) - 0.5*n*np.log(2*np.pi)) # marginal likelihood - (5.8)

def lmlgh(params,y,R):
    k1, k2 = params
    al = mul(Kyinvh,y)
    dKdk1 = Kfh*(1/k1)
    dKdk2 = RBF(x,None,k1,k2**0.5)[1].reshape(n,n)
    lmlg1 = -(0.5*np.trace(mul(mul(al,T(al)) - Kyinvh, dKdk1)))
    lmlg2 = -(0.5*np.trace(mul(mul(al,T(al)) - Kyinvh, dKdk2)))
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
def GPRfit(xs,k1,k2,sig):
    #Kst = RBF2(xtest,x,k1,k2)[0]
    Ky = RBF(x,None,k1,k2**0.5)[0] + (sig**2)*np.identity(n)
    Ks = RBF(xs, x, k1, k2**0.5)
    Kss = RBF(xs, None, k1, k2**0.5)[0]
    L = cholesky(Ky)
    al = solve(T(L), solve(L,y))
    fmst = mul(Ks,al)
    varfmst = np.empty([n,1])
    for i in range(np.size(xs)):
        v = solve(L,T(Ks[:,i]))
        varfmst[i] = Kss[i,i] - mul(T(v),v)
    lmlopt = -0.5*mul(T(y),al) - np.trace(np.log(L)) - 0.5*n*np.log(2*np.pi)
    return fmst, varfmst[::-1], lmlopt

def GPRfith(xs,k1,k2,R,Rs):
    #Kst = RBF2(xtest,x,k1,k2)[0]
    Ky = RBF(x,None,k1,k2**0.5)[0] + R
    Ks = RBF(xs, x, k1, k2**0.5)
    Kss = RBF(xs, None, k1, k2)[0]
    L = cholesky(Ky)
    al = solve(T(L), solve(L,y))
    fmst = mul(Ks,al)
    varfmst = np.empty([n,1])
    for i in range(np.size(xs)):
        v = solve(L,T(Ks[:,i]))
        varfmst[i] = Kss[i,i] + Rs[i,i] - mul(T(v),v)
    lmlopt = -0.5*mul(T(y),al) - np.trace(np.log(L)) - 0.5*n*np.log(2*np.pi)
    return fmst, varfmst[::-1], lmlopt
# %% ========================= Heteroscedastic GPR implementation =========================
# using algorithm from Most Likely Heteroscedastic Gaussian Process Regression - Kristian Kersting et. al. 
# I will use my implementation only to start with for consistency - could use SKLearn for steps 1-3 for speed
# Step 1: homoscedastic GP1 on D - (x,y)
numh = 2 # number of hyperparameters in kernel function 
k1is1 = 1.0
k2is1 = 1.0
#sig1 = np.amax(sd) # set constant noise level as the maximum noise in dataset 
sig1 = np.var(y)**0.5
kis1 = np.ndarray((numh,), buffer=np.array([k1is1,k2is1]), dtype = float)
s1res = minimize(lml,kis1,args=(y,sig1),method = 'L-BFGS-B',jac=lmlg,bounds = ((1e-5,1e5),(1e-5,1e5)))
step1res = []
if s1res.success:
    step1res.append(s1res.x)
    print("Success -- step 1 complete")
else:
    raise ValueError(s1res.message)
    #print("Hyperparameter optimisation failed")
k1s1 = step1res[0][0]
k2s1 = step1res[0][1]
fmst, varfmst, lmlopt = GPRfit(x,k1s1,k2s1,sig1)
sigs1 = varfmst**0.5
fmstps1 = fmst + numsig*sigs1
fmstms1 = fmst - numsig*sigs1
labelfillj = "2 sigma Josh"
plt.plot(x,y,'+')
plt.plot(x,fmst, label = 'Josh')
plt.plot(x,ystar, label = 'SK Learn')
# =============================================================================
# plt.fill(np.concatenate([x, x[::-1]]),
#          np.concatenate([fmstps1,
#                         (fmstms1)[::-1]]),
#          alpha=0.3, fc='r', ec='None', label=labelfillj)
# plt.fill(np.concatenate([x, x[::-1]]),
#          np.concatenate([ystarp,
#                         (ystarm)[::-1]]),
#          alpha=0.3, fc='g', ec='None', label=labelfill)
# =============================================================================
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$y$')
# plt.savefig('gprsklearnvsjosh.pdf', dpi=200)
plt.show()

# %% Steps 2-4 in a loop - iterate until convergence 

def hetloop(fmst,ct):
    k1is3 = k2is3 = k1is4 = k2is4 = 1.0
    avediff = 100.0
    notconverged = True
    while notconverged:
        # Step 2: estimate empirical noise levels z 
        z = np.log((y - fmst)**2)  # no average as we have a 1D dataset - would in general be a vector 
        #  Step 3: estimate GP2 on D' - (x,z)
        sig3 = np.var(z)**0.5
        kis3 = np.ndarray((numh,), buffer=np.array([k1is3,k2is3]), dtype = float)
        s3res = minimize(lml,kis3,args=(z,sig3),method = 'L-BFGS-B',jac=lmlg,bounds = ((1e-5,1e5),(1e-5,1e5)))
        step3res = []
        if s3res.success:
            step3res.append(s3res.x)
        else:
            k1is3 = np.random.uniform(0.01,100)
            k2is3 = np.random.uniform(0.01,100)
            print("error - reinitialising hyperparameters")
            continue
            #raise ValueError(s3res.message)
        k1s3 = step3res[0][0]
        k2s3 = step3res[0][1]
        fmst3, varfmst3, lmlopt3 = GPRfit(x,k1s3,k2s3,sig3)
# =============================================================================
#         plt.plot(x,z,'+')
#         plt.plot(x,fmst3)
#         plt.show()
#         print(sig3)
# =============================================================================
        # Step 4: train heteroscedastic GP3 using predictive mean of G2 to predict log noise levels r
        r = exp(fmst3)
        R = r*np.identity(n)
        kis4 = np.ndarray((numh,), buffer=np.array([k1is4,k2is4]), dtype = float)
        s4res = minimize(lmlh,kis4,args=(y,R),method = 'L-BFGS-B',jac=lmlgh,bounds = ((1e-5,1e5),(1e-5,1e5)))
        step4res = []
        if s4res.success:
            step4res.append(s4res.x)
        else:
            #raise ValueError(s4res.message)
            k1is4 = np.random.uniform(0.01,100)
            k2is4 = np.random.uniform(0.01,100)
            print("error - reinitialising hyperparameters")
            continue 
        k1s4 = step4res[0][0]
        k2s4 = step4res[0][1]
        fmst4, varfmst4, lmlopt4 = GPRfith(x,k1s4,k2s4,R,R)
        # test for convergence 
        avediff = sum(np.where(abs(fmst - fmst4)<ct,0,abs(fmst - fmst4)))
        if avediff != 0:
            print("average deviation = " + str(avediff))
            fmst = fmst4 # if not converged, set GP1 = GP3 and repeat steps 2-4
            k1is3 = k1s3 # start optimiser at previous local optimum 
            k2is3 = k2s3
            k1is4 = k1s4
            k2is4 = k2s4
        else:
            notconverged = False
    return fmst4,varfmst4, lmlopt4

fmst4,varfmst4, lmlopt4 = hetloop(fmst,1e-3)

#  plotting 

sigs4 = varfmst4**0.5
fmstps4 = fmst4 + numsig*sigs4
fmstms4 = fmst4 - numsig*sigs4

# %%

plt.plot(x,y,'+', label = 'data')
plt.plot(x,fmst4, label = 'heteroscedastic GP')
plt.plot(x,ystar, label = 'SKLearn homoscedastic GP')
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([fmstps4,
                        (fmstms4)[::-1]]),
         alpha=0.3, fc='r', ec='None', label=labelfillj)
plt.savefig('Aheteroscedasticgdata.pdf', dpi=200)
plt.show()





