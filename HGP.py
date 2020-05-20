# Implementation of heteroscedastic GPR 

#  imports

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from sklearn.preprocessing import StandardScaler
import time
from numpy.linalg import cholesky
from numpy import transpose as T
from numpy.linalg import inv, det, solve
from numpy import matmul as mul, exp
from numpy.random import normal
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import norm
from scipy import special
import multiprocessing
from GHquad import GHquad
import matplotlib
import math
#matplotlib.rc_file_defaults()   # use to return to Matplotlib defaults 

#x = np.genfromtxt(open("Pchripple.csv", "r"), delimiter=",", dtype =float)
#y = np.genfromtxt(open("SNRripple.csv", "r"), delimiter=",", dtype =float)
#x = np.genfromtxt(open("Pchripple10.csv", "r"), delimiter=",", dtype =float)
#y = np.genfromtxt(open("SNRripple10.csv", "r"), delimiter=",", dtype =float)

SNR = np.genfromtxt(open("linkSNR.csv", "r"), delimiter=",", dtype =float)
Pch = np.genfromtxt(open("linkPch.csv", "r"), delimiter=",", dtype =float)

    # data used by Goldberg 
    # =============================================================================
    #numpoints = 100
    #x = np.linspace(0,1,numpoints)
    #sd = np.linspace(0.5,1.5,numpoints)
    #y = np.zeros(numpoints)
    #n = np.size(x)
    #for i in range(numpoints):
    #    y[i] = 2*np.sin(2*np.pi*x[i]) + np.random.normal(0, sd[i])
    # =============================================================================
    
    # data used by Yuan and Wahba 
    # =============================================================================
    #numpoints = 200
    #x = np.linspace(0,1,numpoints)  
    #ymean = 2*(exp(-30*(x-0.25)**2) + np.sin(np.pi*x**2)) - 2
    #ysd = exp(np.sin(2*np.pi*x))
    #y = np.random.normal(ymean, ysd)
    # =============================================================================
    
    # data used by Williams 
    #numpoints = 200
    #x = np.linspace(0,np.pi,numpoints)
    #wmean = np.sin(2.5*x)*np.sin(1.5*x)
    #wsd = 0.01 + 0.25*(1 - np.sin(2.5*x))**2
    #y = np.random.normal(wmean, wsd)
    
def HGPfunc(x,y,plot):


    y = y.reshape(-1,1)
    x = x.reshape(-1,1)
    if plot:
        plt.plot(x,y,'+')
        plt.xlabel("Pch (dBm)")
        plt.ylabel("SNR (dB)")
        plt.savefig('Adataset.png', dpi=200)
        plt.show()
    n = np.size(x)
    scaler = StandardScaler().fit(y)
    y = scaler.transform(y)
    numsig = 2
    def sqexp(X,Y,k1,k2):
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
    # heteroscedastic versions of functions 
    global Kyinvh
    Kyinvh = 0.0
    global Kfh
    Kfh =  0.0 
    def lmlh(params,y,R):
        #print(params)  # show progress of fit
        [k1, k2] = params
        global Kfh
        Kfh = sqexp(x,None,k1,k2**0.5)[0]
        Ky = Kfh + R # calculate initial kernel with noise
        global Kyinvh
        Kyinvh = inv(Ky)
        return -(-0.5*mul(mul(T(y),Kyinvh), y) - 0.5*np.log((det(Ky))) - 0.5*n*np.log(2*np.pi)) # marginal likelihood - (5.8)
    def lmlgh(params,y,R):
        k1, k2 = params
        al = mul(Kyinvh,y)
        dKdk1 = Kfh*(1/k1)
        dKdk2 = sqexp(x,None,k1,k2**0.5)[1].reshape(n,n)
        lmlg1 = -(0.5*np.trace(mul(mul(al,T(al)) - Kyinvh, dKdk1)))
        lmlg2 = -(0.5*np.trace(mul(mul(al,T(al)) - Kyinvh, dKdk2)))
        return np.ndarray((2,), buffer=np.array([lmlg1,lmlg2]), dtype = float)
    def GPRfith(xs,k1,k2,R,Rs):
        Ky = sqexp(x,None,k1,k2**0.5)[0] + R
        Ks = sqexp(xs, x, k1, k2**0.5)
        Kss = sqexp(xs, None, k1, k2)[0]
        L = cholesky(Ky)
        al = solve(T(L), solve(L,y))
        fmst = mul(Ks,al)
        varfmst = np.empty([n,1])
        for i in range(np.size(xs)):
            v = solve(L,T(Ks[:,i]))
            varfmst[i] = Kss[i,i] - mul(T(v),v)  + Rs[i,i]
        lmlopt = -0.5*mul(T(y),al) - np.trace(np.log(L)) - 0.5*n*np.log(2*np.pi)
        #return fmst, varfmst[::-1], lmlopt
        return fmst, varfmst, lmlopt
    def hypopth(y, numrestarts, R):
        numh = 2 # number of hyperparameters in kernel function 
        k1s4 = np.empty([numrestarts,1])
        k2s4 = np.empty([numrestarts,1])
        for i in range(numrestarts):    
            k1is4 = np.random.uniform(1e-2,1e2)
            k2is4 = np.random.uniform(1e-1,1e3)
            kis4 = np.ndarray((numh,), buffer=np.array([k1is4,k2is4]), dtype = float)
            s4res = minimize(lmlh,kis4,args=(y,R),method = 'L-BFGS-B',jac=lmlgh,bounds = ((1e-2,1e3),(0.5,1e3)))
            step4res = []
            if s4res.success:
                step4res.append(s4res.x)
            else:
                #raise ValueError(s4res.message)
                k1is4 = np.random.uniform(1e-2,1e3)
                k2is4 = np.random.uniform(0.5,1e3)
                print("error in hypopth() - reinitialising hyperparameters")
                continue 
            k1s4[i] = step4res[0][0]
            k2s4[i] = step4res[0][1]
        lmltest = [lmlh([k1s4[i],k2s4[i]],y,R) for i in range(numrestarts)]
        k1f = k1s4[np.argmin(lmltest)]
        k2f = k2s4[np.argmin(lmltest)]
            #lml(params,y,sig)
        return k1f, k2f
    def hetloopSK(fmst,varfmst,numiters,numrestarts):
        s = 200
        #k1is3, k2is3, k1is4,k2is4  =  np.random.uniform(1e-2,1e2,4)
        MSE = np.empty([numiters,1])
        NLPD = np.empty([numiters,1])
        fmstf = np.empty([numiters,n])
        varfmstf = np.empty([numiters,n])
        lmloptf = np.empty([numiters,1])
        rf = np.empty([numiters,n])
        i = 0
        while i < numiters:        
            breakwhile = False
            # Step 2: estimate empirical noise levels z 
            #k1is4,k2is4  = np.random.uniform(1e-2,1e2,2)
            k1is3, k1is4  =  np.random.uniform(1e-2,1e2,2)
            k2is3, k2is4  =  np.random.uniform(1e-1,1e2,2)
            z = np.empty([n,1])
            for j in range(n):
                #np.random.seed()
                normdraw = normal(fmst[j], varfmst[j]**0.5, s).reshape(s,1)
                z[j] = np.log((1/s)*0.5*sum((y[j] - normdraw)**2))
                if math.isnan(z[j]): # True for NaN values
                    breakwhile = True
                    break
            if breakwhile:
                print("Nan value in z -- restarting iter "+ str(i))
                continue
            #  Step 3: estimate GP2 on D' - (x,z)
            kernel2 = C(k1is3, (1e-2, 1e2)) * RBF(k2is3, (0.5, 1e2)) 
            gpr2 = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer = numrestarts, normalize_y=False, alpha=np.var(z))
            
            gpr2.fit(x, z)
            ystar2, sigma2 = gpr2.predict(x, return_std=True )
            sigma2 = (sigma2**2 + 1)**0.5
        # Step 4: train heteroscedastic GP3 using predictive mean of G2 to predict log noise levels r
            r = exp(ystar2)
            R = r*np.identity(n)
            k1s4, k2s4 = hypopth(y,numrestarts,R)
            fmst4, varfmst4, lmlopt4 = GPRfith(x,k1s4,k2s4,R,R)
            # test for convergence 
            MSE[i] = (1/n)*sum(((y-fmst4)**2)/np.var(y))
            #NLPD[i] = sum([(1/n)*(-np.log(norm.pdf(x[j], fmst4[j], varfmst4[j]**0.5))) for j in range(n) ])
            test3 = np.empty([n,1])
            for k in range(n):
                test3[k] = -np.log(norm.pdf(x[k], fmst[k], varfmst[k]**0.5))
            print("NLPD argument " + str(norm.pdf(x[k], fmst[k], varfmst[k]**0.5)))
            NLPD[i] = sum(test3)*(1/n)
            print("MSE = " + str(MSE[i]))
            print("NLPD = " + str(NLPD[i]))
            print("finished iteration " + str(i+1))
            fmstf[i,:] = fmst4.reshape(n)
            varfmstf[i,:] = varfmst4.reshape(n)
            lmloptf[i] = lmlopt4
            fmst = fmst4
            varfmst = varfmst4
            rf[i,:] = r.reshape(n)
            #k1is3 = k1s4
            #k2is3 = k2s4
            i = i + 1
        return fmstf,varfmstf, lmloptf, MSE, rf, NLPD #  , NLPD 
    #kernel1 = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e3)) + W(1.0, (1e-5, 1e5))
    #gpr1 = GaussianProcessRegressor(kernel=kernel1, n_restarts_optimizer = 0, normalize_y=True)
    kernel1 = C(1.0, (1e-1, 1e1)) * RBF(1.0, (1e-2, 1e2)) 
    gpr1 = GaussianProcessRegressor(kernel=kernel1, n_restarts_optimizer = 10, normalize_y=False, alpha=np.var(y))
    gpr1.fit(x, y)
    ystar1, sigma1 = gpr1.predict(x, return_std=True )
    var1 = (sigma1**2 + np.var(y))
    #sigma1 = np.reshape(sigma1,(np.size(sigma1), 1))
    
    # %%
    numiters = 20
    numrestarts = 5
    start_time = time.time()
    fmstf,varfmstf, lmloptf, MSE, rf,NLPD = hetloopSK(ystar1,var1,numiters,numrestarts)
    duration = time.time() - start_time
    # %% learning curve
    if plot:
        numiterations = np.linspace(1,numiters,numiters,dtype=int)
        plt.plot(numiterations,MSE,'+')
        plt.plot(numiterations,MSE)
        plt.ylabel('MSE')  
        plt.xlabel('Iterations')
        plt.savefig('AMSE.png', dpi=200)
        plt.show()
        
        # =============================================================================
        plt.plot(numiterations,NLPD,'+')
        plt.plot(numiterations,NLPD)
        plt.ylabel('NLPD')  
        plt.xlabel('Iterations')
        plt.savefig('ANLPD.png', dpi=200)
        plt.show()
        # =============================================================================
        
        plt.plot(numiterations, lmloptf,'+')
        plt.plot(numiterations, lmloptf)
        plt.ylabel('LML')  
        plt.xlabel('Iterations')
        plt.savefig('ALML.png', dpi=200)
        plt.show()
    
    # %%   plotting
    ind = numiters - 1
    #ind = 
    fmst4 = fmstf[ind]
    varfmst4 = varfmstf[ind]
    lmlopt4 = lmloptf[ind]
     
    sigs4 = varfmst4**0.5
    fmstps4 = fmst4 + numsig*sigs4
    fmstms4 = fmst4 - numsig*sigs4
    
    
    numsig2 = 3
    fmstps42 = fmst4 + numsig2*sigs4
    fmstms42 = fmst4 - numsig2*sigs4
    
    numsig3 = 4
    fmstps43 = fmst4 + numsig3*sigs4
    fmstms43 = fmst4 - numsig3*sigs4
    
    numsig4 = 5
    fmstps44 = fmst4 + numsig4*sigs4
    fmstms44 = fmst4 - numsig4*sigs4
    
    # %%
    
    font = {'family' : 'normal',
            'size'   : 14}
    plt.rc('font', **font)
    # =============================================================================
    # labelfillj2 = str(numsig) + '$\sigma$'
    # # =============================================================================
    yi = scaler.inverse_transform(y)
    fmst4i = scaler.inverse_transform(fmst4)
    fmstps4i = scaler.inverse_transform(fmstps4)
    fmstms4i = scaler.inverse_transform(fmstms4)
    
    fmstps4i2 = scaler.inverse_transform(fmstps42)
    fmstms4i2 = scaler.inverse_transform(fmstms42)
    fmstps4i3 = scaler.inverse_transform(fmstps43)
    fmstms4i3 = scaler.inverse_transform(fmstms43)
    fmstps4i4 = scaler.inverse_transform(fmstps44)
    fmstms4i4 = scaler.inverse_transform(fmstms44)
    
    if plot:
        plt.plot(x, yi,'+')
        plt.plot(x, fmst4i, label = 'HGPR')
        # =============================================================================
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([fmstps4i,
                                (fmstms4i)[::-1]]),
                 alpha=0.3, fc='b', ec='None', label='$2 \sigma$')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([fmstps4i2,
                                (fmstps4i)[::-1]]),
                 alpha=0.3, fc='r', ec='None', label='$3 \sigma$')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([fmstms4i,
                                (fmstms4i2)[::-1]]),
                 alpha=0.3, fc='r', ec='None')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([fmstps4i3,
                                (fmstps4i2)[::-1]]),
                 alpha=0.3, fc='y', ec='None', label='$4 \sigma$')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([fmstms4i2,
                                (fmstms4i3)[::-1]]),
                 alpha=0.3, fc='y', ec='None')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([fmstps4i4,
                                (fmstps4i3)[::-1]]),
                 alpha=0.3, fc='g', ec='None', label='$5 \sigma$')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([fmstms4i3,
                                (fmstms4i4)[::-1]]),
                 alpha=0.3, fc='g', ec='None')
        #plt.axis([x[0],x[-1],11.3,14.7])
        plt.xlabel('$P_{ch}$(dBm)')  
        plt.ylabel('SNR(dB)')
        plt.legend(loc="lower right",ncol=3)
        plt.savefig('AHGPfit.pdf', dpi=200,bbox_inches='tight')
        plt.show()
        
        
        # %% ECOC sigma plot 
        plt.plot(x,sigs4,color='k',LineStyle='-.',label='GP1')
        plt.plot(x,rf[ind]**0.5,color='k',LineStyle='-',label='$\sqrt{r(x)}$ 1')
        #plt.plot(x,sigs420,color='k',LineStyle='--',label='GP2')
        #plt.plot(x,rf20**0.5,color='k',LineStyle=':',label='$\sqrt{r(x)}$ 2')
        plt.xlabel('$P_{ch}$(dBm)')  
        plt.ylabel('$\sigma$(dB)')
        plt.legend()
        #plt.savefig('NoisefunctionECOCcombined.pdf', dpi=200,bbox_inches='tight')
        plt.savefig('Noisefunctionnum50.pdf', dpi=200,bbox_inches='tight')
        plt.show()
    # %% BER transform
    
    M = 16
    def BERcalc(M, SNR):
        if M == 4: 
            BER = 0.5*special.erfc(SNR**0.5)
            #BERrs = 0.5*special.erfc(SNRRSr**0.5)
            #BERrs2 = 0.5*special.erfc(SNRRS2r**0.5)
             
        elif M == 16:
            BER = (3/8)*special.erfc(((2/5)*SNR)**0.5) + (1/4)*special.erfc(((18/5)*SNR)**0.5) - (1/8)*special.erfc((10*SNR)**0.5)
            #BERrs = (3/8)*special.erfc(((2/5)*SNRRSr)**0.5) + (1/4)*special.erfc(((18/5)*SNRRSr)**0.5) - (1/8)*special.erfc((10*SNRRSr)**0.5)
            #BERrs2 = (3/8)*special.erfc(((2/5)*SNRRS2r)**0.5) + (1/4)*special.erfc(((18/5)*SNRRS2r)**0.5) - (1/8)*special.erfc((10*SNRRS2r)**0.5)
             
        elif M == 64:
            BER = (7/24)*special.erfc(((1/7)*SNR)**0.5) + (1/4)*special.erfc(((9/7)*SNR)**0.5) - (1/24)*special.erfc(((25/7)*SNR)**0.5) - (1/24)*special.erfc(((25/7)*SNR)**0.5) + (1/24)*special.erfc(((81/7)*SNR)**0.5) - (1/24)*special.erfc(((169/7)*SNR)**0.5) 
            #BERrs = (7/24)*special.erfc(((1/7)*SNRRSr)**0.5) + (1/4)*special.erfc(((9/7)*SNRRSr)**0.5) - (1/24)*special.erfc(((25/7)*SNRRSr)**0.5) - (1/24)*special.erfc(((25/7)*SNRRSr)**0.5) + (1/24)*special.erfc(((81/7)*SNRRSr)**0.5) - (1/24)*special.erfc(((169/7)*SNRRSr)**0.5) 
            #BERrs2 = (7/24)*special.erfc(((1/7)*SNRRS2r)**0.5) + (1/4)*special.erfc(((9/7)*SNRRS2r)**0.5) - (1/24)*special.erfc(((25/7)*SNRRS2r)**0.5) - (1/24)*special.erfc(((25/7)*SNRRS2r)**0.5) + (1/24)*special.erfc(((81/7)*SNRRS2r)**0.5) - (1/24)*special.erfc(((169/7)*SNRRS2r)**0.5) 
        return BER
    
       
    By= BERcalc(M,yi)
    Bfmst = BERcalc(M,fmst4i)
    Bfmstps = BERcalc(M, fmstps4i)
    Bfmstps2 = BERcalc(M, fmstps4i2)
    Bfmstps3 = BERcalc(M, fmstps4i3)
    Bfmstps4 = BERcalc(M, fmstps4i4)
    Bfmstms = BERcalc(M, fmstms4i)
    Bfmstms2 = BERcalc(M, fmstms4i2)
    Bfmstms3 = BERcalc(M, fmstms4i3)
    Bfmstms4 = BERcalc(M, fmstms4i4)
    
    
    if plot:
        #ax = plt.subplot(111)
        plt.plot(x,By,'+')
        plt.plot(x, Bfmst, label = 'HGP')
        # =============================================================================
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([Bfmstps,
                                (Bfmstms)[::-1]]),
                 alpha=0.3, fc='b', ec='None', label='$2 \sigma$')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([Bfmstps2,
                                (Bfmstps)[::-1]]),
                 alpha=0.3, fc='r', ec='None', label='$3 \sigma$')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([Bfmstms,
                                (Bfmstms2)[::-1]]),
                 alpha=0.3, fc='r', ec='None')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([Bfmstps3,
                                (Bfmstps2)[::-1]]),
                 alpha=0.3, fc='y', ec='None', label='$4 \sigma$')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([Bfmstms2,
                                (Bfmstms3)[::-1]]),
                 alpha=0.3, fc='y', ec='None')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([Bfmstps4,
                                (Bfmstps3)[::-1]]),
                 alpha=0.3, fc='g', ec='None', label='$5 \sigma$')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([Bfmstms3,
                                (Bfmstms4)[::-1]]),
                 alpha=0.3, fc='g', ec='None')
        #ax.set_yticklabels(["{:.1e}".format(t) for t in ax.get_yticks()])
        #plt.axis([x[0],x[-1],2e-4,1.15e-3])
        plt.xlabel('$P_{ch}$(dBm)')  
        plt.ylabel('BER (AU)')
        #plt.yticks([])
        plt.legend()
        #plt.savefig('AHGPber.pdf', dpi=200)
        plt.show()
    
    
    # %% ================================ Mutual information transform ===========================================
    MIcalc = False 
    # import constellation shapes from MATLAB-generated csv files 
    if MIcalc:  
        Qam4r = np.genfromtxt(open("qam4r.csv", "r"), delimiter=",", dtype =float)
        Qam4i = np.genfromtxt(open("qam4i.csv", "r"), delimiter=",", dtype =float)
        Qam16r = np.genfromtxt(open("qam16r.csv", "r"), delimiter=",", dtype =float)
        Qam16i = np.genfromtxt(open("qam16i.csv", "r"), delimiter=",", dtype =float)
        Qam32r = np.genfromtxt(open("qam32r.csv", "r"), delimiter=",", dtype =float)
        Qam32i = np.genfromtxt(open("qam32i.csv", "r"), delimiter=",", dtype =float)
        Qam64r = np.genfromtxt(open("qam64r.csv", "r"), delimiter=",", dtype =float)
        Qam64i = np.genfromtxt(open("qam64i.csv", "r"), delimiter=",", dtype =float)
        Qam128r = np.genfromtxt(open("qam128r.csv", "r"), delimiter=",", dtype =float)
        Qam128i = np.genfromtxt(open("qam128i.csv", "r"), delimiter=",", dtype =float)
        
        Qam4 = Qam4r + 1j*Qam4i
        Qam16 = Qam16r + 1j*Qam16i
        Qam32 = Qam32r + 1j*Qam32i
        Qam64 = Qam64r + 1j*Qam64i
        Qam128 = Qam128r + 1j*Qam128i
        #  ================================ Estimate MI ================================ 
        # set modulation format order and number of terms used in Gauss-Hermite quadrature
        M = 16
        L = 6
        
        def MIGHquad(SNR):
            if M == 4:
                Ps = np.mean(np.abs(Qam4**2))
                X = Qam4
            elif M == 16:
                Ps = np.mean(np.abs(Qam16**2))
                X = Qam16
            elif M == 32:
                Ps = np.mean(np.abs(Qam32**2))
                X = Qam32
            elif M == 64:
                Ps = np.mean(np.abs(Qam64**2))
                X = Qam64
            elif M == 128:
                Ps = np.mean(np.abs(Qam128**2))
                X = Qam128
            else:
                print("unrecogised M")
            sigeff2 = Ps/(10**(SNR/10))
            Wgh = GHquad(L)[0]
            Rgh = GHquad(L)[1]
            sum_out = 0
            for ii in range(M):
                sum_in = 0
                for l1 in range(L):      
                    sum_inn = 0
                    for l2 in range(L):
                        sum_exp = 0
                        for jj in range(M):  
                            arg_exp = np.linalg.norm(X[ii]-X[jj])**2 + 2*(sigeff2**0.5)*np.real( (Rgh[l1]+1j*Rgh[l2])*(X[ii]-X[jj]));
                            sum_exp = np.exp(-arg_exp/sigeff2) + sum_exp
                        sum_inn = Wgh[l2]*np.log2(sum_exp) + sum_inn
                    sum_in = Wgh[l1]*sum_inn + sum_in
                sum_out = sum_in + sum_out
            return np.log2(M)- (1/(M*np.pi))*sum_out 
        
        def findMI(SNR):
            with multiprocessing.Pool() as pool:
                Ixy = pool.map(MIGHquad, SNR) 
            return Ixy
        
        MIy = findMI(yi)
        MIfmst = findMI(fmst4i)
        MIfmstps = findMI(fmstps4i)
        MIfmstps2 = findMI(fmstps4i2)
        MIfmstps3 = findMI(fmstps4i3)
        MIfmstps4 = findMI(fmstps4i4)
        MIfmstms = findMI(fmstms4i)
        MIfmstms2 = findMI(fmstms4i2)
        MIfmstms3 = findMI(fmstms4i3)
        MIfmstms4 = findMI(fmstms4i4)
        
        # %%
        
        plt.plot(x, MIy,'+')
        plt.plot(x, MIfmst, label = 'HGP')
        # =============================================================================
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([MIfmstps,
                                (MIfmstms)[::-1]]),
                 alpha=0.3, fc='b', ec='None', label='$2 \sigma$')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([MIfmstps2,
                                (MIfmstps)[::-1]]),
                 alpha=0.3, fc='r', ec='None', label='$3 \sigma$')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([MIfmstms,
                                (MIfmstms2)[::-1]]),
                 alpha=0.3, fc='r', ec='None')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([MIfmstps3,
                                (MIfmstps2)[::-1]]),
                 alpha=0.3, fc='y', ec='None', label='$4 \sigma$')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([MIfmstms2,
                                (MIfmstms3)[::-1]]),
                 alpha=0.3, fc='y', ec='None')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([MIfmstps4,
                                (MIfmstps3)[::-1]]),
                 alpha=0.3, fc='g', ec='None', label='$5 \sigma$')
        plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([MIfmstms3,
                                (MIfmstms4)[::-1]]),
                 alpha=0.3, fc='g', ec='None')
        plt.xlabel('$P_{ch}$(dBm)')  
        plt.ylabel('MI (bits/sym)')
        plt.legend()
        plt.savefig('AHGPMI.pdf', dpi=200)
        plt.show()
        
    print("HGP fitting duration: " + str(duration)) 
    
    return fmst4i, fmstps4i, fmstms4i, sigs4, rf[ind]**0.5
   



# %%

x = Pch

prmn = np.empty([np.size(SNR,0),np.size(SNR,1)])
prmnp = np.empty([np.size(SNR,0),np.size(SNR,1)])
prmnn = np.empty([np.size(SNR,0),np.size(SNR,1)])
sig = np.empty([np.size(SNR,0),np.size(SNR,1)])
sigrf = np.empty([np.size(SNR,0),np.size(SNR,1)])
for i in range(np.size(SNR,0)):
    prmn[i], prmnp[i], prmnn[i], sig[i], sigrf[i] = HGPfunc(x,SNR[i],False)

#test1, test2, test3, test4, test5 = HGPfunc(x,SNR[22],False)


# %% 

# =============================================================================
# plt.plot(Pch,SNR[22],'+')
# plt.plot(Pch,test1)
# plt.fill(np.concatenate([x, x[::-1]]),
#                  np.concatenate([test2,
#                                 (test3)[::-1]]),
#                  alpha=0.3, fc='b', ec='None', label='$2 \sigma$')
# plt.xlabel('$P_{ch}$(dBm)')  
# plt.ylabel('SNR (dB)')
# #plt.legend()
# #plt.savefig('AHGPMI.pdf', dpi=200)
# plt.show()
# 
# plt.plot(x,test4,color='k',LineStyle='-.',label='GP1')
# plt.plot(x,test5,color='k',LineStyle='-',label='$\sqrt{r(x)}$ 1')
# plt.xlabel('$P_{ch}$(dBm)')  
# plt.ylabel('$\sigma$(dB)')
# plt.legend()
# #plt.savefig('NoisefunctionECOCcombined.pdf', dpi=200,bbox_inches='tight')
# #plt.savefig('Noisefunctionnum50.pdf', dpi=200,bbox_inches='tight')
# plt.show()
# =============================================================================



# %%

np.savetxt('prmn.csv', prmn, delimiter=',') 
np.savetxt('prmnp.csv', prmnp, delimiter=',')
np.savetxt('prmnn.csv', prmnn, delimiter=',')
np.savetxt('sig.csv', sig, delimiter=',')
np.savetxt('sigrf.csv', sigrf, delimiter=',')


# %%

plt.plot(x,SNR[0],'+')
plt.plot(x,prmn[0])
plt.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([prmnp[0],
                                (prmnn[0])[::-1]]),
                 alpha=0.3, fc='b', ec='None', label='$2 \sigma$')
plt.xlabel('$P_{ch}$(dBm)')  
plt.ylabel('SNR (dB)')
plt.legend()
#plt.savefig('AHGPMI.pdf', dpi=200)
plt.show()

plt.plot(x,sig[0],color='k',LineStyle='-.',label='GP1')
plt.plot(x,sigrf[0],color='k',LineStyle='-',label='$\sqrt{r(x)}$ 1')
plt.xlabel('$P_{ch}$(dBm)')  
plt.ylabel('$\sigma$(dB)')
plt.legend()
#plt.savefig('NoisefunctionECOCcombined.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('Noisefunctionnum50.pdf', dpi=200,bbox_inches='tight')
plt.show()














   
    