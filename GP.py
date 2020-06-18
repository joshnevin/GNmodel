

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from sklearn.preprocessing import StandardScaler
import time

#import matplotlib
#matplotlib.rc_file_defaults()   # use to return to Matplotlib defaults 

y = np.genfromtxt(open("tsSNRN0.csv", "r"), delimiter=",", dtype =float)
numpoints = np.size(y,1)
x = np.linspace(0,numpoints-1,numpoints)


def GPtrain(x,y):
    y = y.reshape(-1,1)
    x = x.reshape(-1,1)
    scaler = StandardScaler().fit(y)
    y = scaler.transform(y)
    #kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e3)) + W(1.0, (1e-5, 1e5))
    kernel = C(1.0, (1e-3, 1e2)) * RBF(1.0, (1e-2, 1e2)) 
    #print("Initial kernel: %s" % kernel)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 9, normalize_y=False, alpha=np.var(y))
    #gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 9, normalize_y=False)
    gpr.fit(x, y)
    print("Optimised kernel: %s" % gpr.kernel_)
    #print("SKLearn fitting took " + str(end - start) + "s")
    #hyperparameter = np.exp(gpr.kernel_.theta)
    #alpha = hyperparameter[0]**0.5 # k1
    #rho = hyperparameter[1]**0.5 # k2
    #sigmakernel = hyperparameter[2]**0.5
    ystar, sigma = gpr.predict(x, return_std=True )
    sigma = np.reshape(sigma,(np.size(sigma), 1)) 
    sigma = (sigma**2 + 1)**0.5  
    
    ystarp = ystar + sigma
    ystari = scaler.inverse_transform(ystar)
    ystarpi = scaler.inverse_transform(ystarp)
    sigmai = np.mean(ystarpi - ystari)
    
    return ystari, sigmai
prmn = np.empty([np.size(y,0),np.size(y,1)])
sigma = np.empty([np.size(y,0),1])

start = time.time()
for i in range(np.size(y,0)):
    print("link " + str(i))
    prmnt, sigma[i] = GPtrain(x,y[i])
    prmn[i] = prmnt.reshape(100)
end = time.time()

print("GP training took " + str(end-start))   


# %%  

testind = 1 
numsig = 2
labelp = "GPR mean + " + str(numsig) +  " sigma"
labelm = "GPR mean - " + str(numsig) + " sigma"
labelfill = str(numsig) + " sigma" 
ystarp = prmn[testind] + numsig*sigma[testind]   
ystarm = prmn[testind] - numsig*sigma[testind]
 
plt.plot(x,y[testind],'+', label = 'data')
plt.plot(x,prmn[testind], label = 'GPR mean')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([ystarp,
                        (ystarm)[::-1]]),
         alpha=0.3, fc='g', ec='None', label=labelfill)
plt.title("SKlearn")
plt.legend()
#plt.savefig('ASklearncomp.pdf', dpi=200)
plt.show() 