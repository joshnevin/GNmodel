# PERFORMANCE ESTIMATOR
# Author: Josh Nevin 
# Calculation of SNR, BER, MI for AWGN channel using GN model 
# REFS: The GN Model of Non-Linear Propagation in Uncompensated Coherent Optical Systems - Pierluigi Poggiolini
# Information Rate in Ultra-Wideband Optical Fiber Communication Systems Accounting for High-Order Dispersion - Nikita A. Shevchenko et. al.
# Data from NDFIS: NDFIS_cheatsheet_ives.docx
# UCL OTN course notes 
# On the Bit Error Probability of QAM ModulationMichael - P. Fitz and James P. Seymour

# %% ################### imports ####################
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import time 
from scipy import special
#from scipy.stats import iqr
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF
from GHquad import GHquad
from NFmodelGNPy import nf_model
from NFmodelGNPy import lin2db
from NFmodelGNPy import db2lin
#from Routingalgorithm import dijkstra
import random

datagen = False
NDFISimport = True
repairlossimport = True
constellationimport = True
reachcalculation = True
#addnoise = True

asediffvar = 3e-6
TRxSNR = db2lin(26) # TRX SNR [TRxdiffmean = db2lin(1.0)
TRxdiffvar = db2lin(2)
snrdiffvar = 0.5
powerdiffvar = 0.5
#numpoints = 50
numpoints = 100

def main(Ls, Ns, NchNy, NchRS, NchRS2, al, D, PchdBm, NF, gam, addnoise ):
    #################### equipment characteristics ####################
    lam = 1550 # operating wavelength centre [nm]
    f = 299792458/(lam*1e-9) # operating frequency [Hz]
    c = 299792.458 # speed of light in vacuum [nm/ps] -> needed for calculation of beta2
    Rs = 32 # symbol rate [GBaud]
    h = 6.63*1e-34  # Planck's constant [Js]
    BWNy = (NchNy*Rs)/1e3 # full BW of Nyquist signal [THz
    BchRS = 41.6 # channel BW for non-Nyquist signal [GHz]
    BchRS2 = 20.8
    #Df = 5000/(NchRS - 1) # variable channel spacing - total signal BW fixed as 5THz
    Df = 50 # channel frequency spacing for non-Nyquist [GHz]
    Df2 = 25 
    allin = np.log((10**(al/10)))/2 # fibre loss [1/km] -> weird definition, due to exponential decay of electric field instead of power, which is standard 
    #gam = 1.27 # fibre nonlinearity coefficient [1/W*km]
    beta2 = (D*(lam**2))/(2*np.pi*c) # dispersion coefficient at given wavelength [ps^2/km]
    Leff = (1 - np.exp(-2*allin*Ls ))/(2*allin)  # effective length [km]      
    Leffa = 1/(2*allin)  # the asymptotic effective length [km]
    LF = 1 # filtering effects coefficient, takes values 0 < LF <= 1 
    Pch = 1e-3*10**(PchdBm/10)  # ^ [W]
    Gwdm = (Pch*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
    GwdmRS = Pch/(BchRS*1e9)
    GwdmRS2 = Pch/(BchRS2*1e9)
    
    ## equation 13, Gnli(0) for single-span Nyquist-WDM 
    GnliEq13 = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))
    ## equation 15, Gnli(0) for single-span non-Nyquist-WDM 
    GnliEq15 = 1e24*(8/27)*(gam**2)*(GwdmRS**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))
    GnliEq15v2 = 1e24*(8/27)*(gam**2)*(GwdmRS2**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS2**2)*(NchRS2**((2*BchRS2)/Df2))  ) )/(np.pi*beta2*Leffa ))

    ## implement multiple spans using (7) 
    def nthHarmonic(N) : 
        harmonic = 1.00
        for i in range(2, N + 1) : 
            harmonic += 1 / i 
        return harmonic 
    if Ns == 1:
        epsilon = 0
        epsilonRS = 0
        epsilonRS2 = 0
    else:
        #epsilon = 0.05 # exponent from (7) 
        epsilon = (np.log(1 + (2*Leffa*(1 - Ns + Ns*nthHarmonic(int(Ns-1))))/( Ns*Ls*np.arcsinh((np.pi**2)*0.5*beta2*Leffa*BWNy**2 )   )  ))/(np.log(Ns))
        epsilonRS = (3/10)*np.log(1 + (6*Leffa)/(Ls*np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**2)**(BchRS/Df))    ))
        epsilonRS2 = (3/10)*np.log(1 + (6*Leffa)/(Ls*np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS2**2)*(NchRS2**2)**(BchRS2/Df2))    ))

    #epsilonRS = (3/10)*(np.log((1 + 6*Leffa)/( Ls*np.arcsinh((np.pi**2)*0.5*beta2*Leffa*BchRS**2*(NchRS**2)**(BchRS/Df)) )/np.log(Ns))
    GnliEq13mul = GnliEq13*Ns**(1+epsilon)   # implementation of (7) 
    #GnliEq13mulinc = GnliEq13*Ns  # incoherent NLI noise accumulation case (epsilon = 0)
    GnliEq15mul = GnliEq15*Ns**(1+epsilonRS)   # implementation of (7) 
    GnliEq15mul2 = GnliEq15v2*Ns**(1+epsilonRS2)   # implementation of (7) 
 
    # ASE noise bit 
    G = al*Ls
    NFl = 10**(NF/10) 
    Gl = 10**(G/10) 
    OSNRmeasBW = 12.478*1e9 # OSNR measurement BW [Hz]
    Pasech = NFl*h*f*(Gl - 1)*Rs*1e9*Ns # [W] the ASE noise power in one Nyquist channel across all spans
    PasechO = NFl*h*f*(Gl - 1)*OSNRmeasBW*Ns # [W] the ASE noise power in the OSNR measurement BW, defined as Dlam = 0.1nm
    PasechRS = NFl*h*f*(Gl - 1)*BchRS*1e9*Ns # [W] the ASE noise power in one non-Nyquist channel across all spans
    PasechRS2 = NFl*h*f*(Gl - 1)*BchRS2*1e9*Ns
    
    # SNR calc + plotting
    SNRanalytical = np.zeros(numpoints)
    SNRanalyticalO = np.zeros(numpoints)
    SNRanalyticalRS = np.zeros(numpoints)
    SNRanalyticalRS2 = np.zeros(numpoints)
    OSNRanalytical = np.zeros(numpoints)
    for i in range(numpoints):
        if addnoise:
            np.random.seed()
            Pasepert = np.random.normal(0,asediffvar)            
            TRxpert = np.random.normal(0,TRxdiffvar)            
            if np.size(Pasech) > 1 and np.size(GnliEq13mul) > 1 and np.size(Pch)==1:
                SNRanalytical[i] = 10*np.log10( 1/( 1/((LF*Pch)/(Pasech[i]+Pasepert + GnliEq13mul[i]*Rs*1e9) )  + (1/(TRxSNR+TRxpert)) )  ) 
                SNRanalyticalO[i] = 10*np.log10((LF*Pch)/(PasechO[i] + GnliEq13mul[i]*OSNRmeasBW))
                OSNRanalytical[i] = 10*np.log10((LF*Pch)/PasechO[i])
                SNRanalyticalRS[i] = 10*np.log10((LF*Pch)/(PasechRS[i] + GnliEq15mul[i]*BchRS*1e9))
                SNRanalyticalRS2[i] = 10*np.log10((LF*Pch)/(PasechRS2[i] + GnliEq15mul2[i]*BchRS2*1e9))
                
            elif np.size(Pasech) > 1 and np.size(GnliEq13mul) == 1 and np.size(Pch)==1:    
                
                SNRanalytical[i] = 10*np.log10( 1/( 1/((LF*Pch)/(Pasech[i]+Pasepert + GnliEq13mul*Rs*1e9) )  + (1/(TRxSNR+TRxpert)) )  ) 
                SNRanalyticalO[i] = 10*np.log10((LF*Pch)/(PasechO[i] + GnliEq13mul*OSNRmeasBW))
                SNRanalyticalRS2[i] = 10*np.log10((LF*Pch)/(PasechRS2[i] + GnliEq15mul2*BchRS2*1e9))
                SNRanalyticalRS[i] = 10*np.log10((LF*Pch)/(PasechRS[i] + GnliEq15mul*BchRS*1e9))
                OSNRanalytical[i] = 10*np.log10((LF*Pch)/PasechO[i])
            
            elif np.size(Pasech) == 1 and np.size(GnliEq13mul) > 1 and np.size(Pch)==1:    
            
                SNRanalytical[i] = 10*np.log10( 1/( 1/((LF*Pch)/(Pasech + Pasepert + GnliEq13mul[i]*Rs*1e9) )  + (1/(TRxSNR+TRxpert)) )  ) 
                SNRanalyticalO[i] = 10*np.log10((LF*Pch)/(PasechO + GnliEq13mul[i]*Rs*1e9))
                #SNRanalytical[i] = 10*np.log10((LF*Pch)/(Pasech + GnliEq13mul[i]*OSNRmeasBW))
                SNRanalyticalRS[i] = 10*np.log10((LF*Pch)/(PasechRS + GnliEq15mul[i]*BchRS*1e9))
                SNRanalyticalRS2[i] = 10*np.log10((LF*Pch)/(PasechRS2 + GnliEq15mul2[i]*BchRS2*1e9))
                OSNRanalytical[i] = 10*np.log10((LF*Pch)/PasechO)       
            
            elif np.size(Pasech) == 1 and np.size(GnliEq13mul) > 1 and np.size(Pch) > 1:    
            
                SNRanalytical[i] = 10*np.log10( 1/( 1/((LF*Pch[i])/(Pasech+Pasepert + GnliEq13mul[i]*Rs*1e9) )  + (1/(TRxSNR+TRxpert)) )  ) 
                SNRanalyticalO[i] = 10*np.log10((LF*Pch[i])/(PasechO + GnliEq13mul[i]*OSNRmeasBW))
                SNRanalyticalRS[i] = 10*np.log10((LF*Pch[i])/(PasechRS + GnliEq15mul[i]*BchRS*1e9))
                SNRanalyticalRS2[i] = 10*np.log10((LF*Pch[i])/(PasechRS2 + GnliEq15mul2[i]*BchRS2*1e9))
                OSNRanalytical[i] = 10*np.log10((LF*Pch[i])/PasechO)      
                
            elif np.size(Pasech) == 1 and np.size(GnliEq13mul) == 1 and np.size(Pch) == 1:    
            
                SNRanalytical[i] = 10*np.log10( 1/( 1/((LF*Pch)/(Pasech+Pasepert + GnliEq13mul*Rs*1e9) )  + (1/(TRxSNR+TRxpert)) )  ) 
                SNRanalyticalO[i] = 10*np.log10((LF*Pch)/(PasechO + GnliEq13mul*OSNRmeasBW))
                SNRanalyticalRS[i] = 10*np.log10((LF*Pch)/(PasechRS + GnliEq15mul*BchRS*1e9))
                SNRanalyticalRS2[i] = 10*np.log10((LF*Pch)/(PasechRS2 + GnliEq15mul2*BchRS2*1e9))   
                OSNRanalytical[i] = 10*np.log10((LF*Pch)/PasechO)
            else:
                #SNRanalyticalRS[i] = 10*np.log10((LF*Pch[i])/(PasechRS[i] + GnliEq15mul[i]*BchRS*1e9))
                SNRanalytical[i] = 10*np.log10( 1/( 1/((LF*Pch[i])/(Pasech[i]+Pasepert + GnliEq13mul[i]*Rs*1e9) )  + (1/(TRxSNR+TRxpert)) )  ) 
                SNRanalyticalO[i] = 10*np.log10((LF*Pch[i])/(PasechO[i] + GnliEq13mul[i]*OSNRmeasBW))
                SNRanalyticalRS2[i] = 10*np.log10((LF*Pch[i])/(PasechRS2[i] + GnliEq15mul2[i]*BchRS2*1e9))
                SNRanalyticalRS[i] = 10*np.log10((LF*Pch[i])/(PasechRS[i] + GnliEq15mul[i]*BchRS*1e9))
                OSNRanalytical[i] = 10*np.log10((LF*Pch[i])/PasechO[i])  
        else:
            if np.size(Pasech) > 1 and np.size(GnliEq13mul) > 1 and np.size(Pch)==1:
                SNRanalytical[i] = 10*np.log10((LF*Pch)/(Pasech[i] + GnliEq13mul[i]*Rs*1e9))
                SNRanalyticalO[i] = 10*np.log10((LF*Pch)/(PasechO[i] + GnliEq13mul[i]*OSNRmeasBW))
                OSNRanalytical[i] = 10*np.log10((LF*Pch)/PasechO[i])
                SNRanalyticalRS[i] = 10*np.log10((LF*Pch)/(PasechRS[i] + GnliEq15mul[i]*BchRS*1e9))
                SNRanalyticalRS2[i] = 10*np.log10((LF*Pch)/(PasechRS2[i] + GnliEq15mul2[i]*BchRS2*1e9))
                
            elif np.size(Pasech) > 1 and np.size(GnliEq13mul) == 1 and np.size(Pch)==1:    
                
                SNRanalytical[i] = 10*np.log10((LF*Pch)/(Pasech[i] + GnliEq13mul*Rs*1e9))
                SNRanalyticalO[i] = 10*np.log10((LF*Pch)/(PasechO[i] + GnliEq13mul*OSNRmeasBW))
                SNRanalyticalRS2[i] = 10*np.log10((LF*Pch)/(PasechRS2[i] + GnliEq15mul2*BchRS2*1e9))
                SNRanalyticalRS[i] = 10*np.log10((LF*Pch)/(PasechRS[i] + GnliEq15mul*BchRS*1e9))
                OSNRanalytical[i] = 10*np.log10((LF*Pch)/PasechO[i])
            
            elif np.size(Pasech) == 1 and np.size(GnliEq13mul) > 1 and np.size(Pch)==1:    
            
                SNRanalytical[i] = 10*np.log10((LF*Pch)/(Pasech + GnliEq13mul[i]*Rs*1e9))
                SNRanalyticalO[i] = 10*np.log10((LF*Pch)/(PasechO + GnliEq13mul[i]*Rs*1e9))
                #SNRanalytical[i] = 10*np.log10((LF*Pch)/(Pasech + GnliEq13mul[i]*OSNRmeasBW))
                SNRanalyticalRS[i] = 10*np.log10((LF*Pch)/(PasechRS + GnliEq15mul[i]*BchRS*1e9))
                SNRanalyticalRS2[i] = 10*np.log10((LF*Pch)/(PasechRS2 + GnliEq15mul2[i]*BchRS2*1e9))
                OSNRanalytical[i] = 10*np.log10((LF*Pch)/PasechO)       
            
            elif np.size(Pasech) == 1 and np.size(GnliEq13mul) > 1 and np.size(Pch) > 1:    
            
                SNRanalytical[i] = 10*np.log10((LF*Pch[i])/(Pasech + GnliEq13mul[i]*Rs*1e9))
                SNRanalyticalO[i] = 10*np.log10((LF*Pch[i])/(PasechO + GnliEq13mul[i]*OSNRmeasBW))
                SNRanalyticalRS[i] = 10*np.log10((LF*Pch[i])/(PasechRS + GnliEq15mul[i]*BchRS*1e9))
                SNRanalyticalRS2[i] = 10*np.log10((LF*Pch[i])/(PasechRS2 + GnliEq15mul2[i]*BchRS2*1e9))
                OSNRanalytical[i] = 10*np.log10((LF*Pch[i])/PasechO)      
                
            elif np.size(Pasech) == 1 and np.size(GnliEq13mul) == 1 and np.size(Pch) == 1:    
            
                SNRanalytical[i] = 10*np.log10((LF*Pch)/(Pasech + GnliEq13mul*Rs*1e9))
                SNRanalyticalO[i] = 10*np.log10((LF*Pch)/(PasechO + GnliEq13mul*OSNRmeasBW))
                SNRanalyticalRS[i] = 10*np.log10((LF*Pch)/(PasechRS + GnliEq15mul*BchRS*1e9))
                SNRanalyticalRS2[i] = 10*np.log10((LF*Pch)/(PasechRS2 + GnliEq15mul2*BchRS2*1e9))   
                OSNRanalytical[i] = 10*np.log10((LF*Pch)/PasechO)
            else:
                #SNRanalyticalRS[i] = 10*np.log10((LF*Pch[i])/(PasechRS[i] + GnliEq15mul[i]*BchRS*1e9))
                SNRanalytical[i] = 10*np.log10((LF*Pch[i])/(Pasech[i] + GnliEq13mul[i]*Rs*1e9))
                SNRanalyticalO[i] = 10*np.log10((LF*Pch[i])/(PasechO[i] + GnliEq13mul[i]*OSNRmeasBW))
                SNRanalyticalRS2[i] = 10*np.log10((LF*Pch[i])/(PasechRS2[i] + GnliEq15mul2[i]*BchRS2*1e9))
                SNRanalyticalRS[i] = 10*np.log10((LF*Pch[i])/(PasechRS[i] + GnliEq15mul[i]*BchRS*1e9))
                OSNRanalytical[i] = 10*np.log10((LF*Pch[i])/PasechO[i])
    return SNRanalytical, SNRanalyticalRS, SNRanalyticalRS2, OSNRanalytical, GnliEq13, epsilon, SNRanalyticalO

# %% ================================ NDFIS data stuff  ===========================================
if NDFISimport:
    NDFISdata = np.genfromtxt(open("NDFISdata.csv", "r"), delimiter=",", dtype =float)
    NDFISdata = np.delete(NDFISdata,0,0 )
    # extract loss and overall loss and remove links for which loss or overall loss is missing 
    NDFISdatanozero = np.delete(NDFISdata,[16,17,20,21],0 )  
    NDFISlossnz = NDFISdatanozero.T[2]
    NDFISlossoverallnz = NDFISdatanozero.T[4]
    NDFISdispnz = NDFISdatanozero.T[6]
    # calculate mean and variance 
    NDFISlossnzmean = np.mean(NDFISlossnz)
    NDFISlossoverallnzmean = np.mean(NDFISlossoverallnz)
    NDFISdispnzmean = np.mean(NDFISdispnz)
    NDFISlossnzvar = np.var(NDFISlossnz)
    NDFISlossoverallnzvar = np.var(NDFISlossoverallnz)
    NDFISdispnzvar = np.var(NDFISdispnz)
    # work out standard deviation = sqrt(variance) as a % of the mean 
    lossvarperc = ((NDFISlossnzvar**0.5)/NDFISlossnzmean)*100  
    lossvarpercoverall = ((NDFISlossoverallnzvar**0.5)/NDFISlossoverallnzmean)*100  
    dispvarperc = ((NDFISdispnzvar**0.5)/NDFISdispnzmean)*100 
    
if repairlossimport:
    # import repair loss data generated by David's implementation of J. A. Nagel,
    #“Statistical Analysis of Single-Mode Fiber Field Splice Losses,” in OFC 2009, p. JWA3.'
    repairlossdata = np.genfromtxt(open("Astotal.csv", "r"), delimiter=",", dtype =float)
    nbinsrep = int(np.genfromtxt(open("Astotalnbins.csv", "r"), delimiter=",", dtype =float))
    countsrep, binsrep = np.histogram(repairlossdata, nbinsrep)

# %% ======================== NDFIS loss fitting stuff ======================== 

# find bin width from Friedman-Diaconis rule 
# =============================================================================
# Binwid = (2*iqr(NDFISlossnz))/(np.size(NDFISlossnz)**(1/3))
# Binnum = (np.max(NDFISlossnz) - np.min(NDFISlossnz))/Binwid
# Binnum = int(np.around(Binnum, decimals=0))
# Binwido = (2*iqr(NDFISlossoverallnz))/(np.size(NDFISlossoverallnz)**(1/3))
# Binnumo = (np.max(NDFISlossoverallnz) - np.min(NDFISlossoverallnz))/Binwido
# Binnumo = int(np.around(Binnumo, decimals=0))
# =============================================================================
# calculate bins for histogram stuff 
#counts, bins = np.histogram(NDFISlossnz, Binnum)
#countso, binso = np.histogram(NDFISlossoverallnz, Binnumo)
counts, bins = np.histogram(NDFISlossnz, 20)
countso, binso = np.histogram(NDFISlossoverallnz, 20)
countsd, binsd = np.histogram(NDFISdispnz, 20)

# how to reconstruct histogram 
# =============================================================================
# xloss = [bins[i]*np.ones(np.size(counts))  for i in range(np.size(bins))   ]
# xloss = np.reshape(xloss, np.size(xloss))
# 
# yloss = [counts[i]*np.ones(np.size(bins))  for i in range(np.size(counts))   ]
# yloss = np.reshape(yloss, np.size(yloss))
# 
# plt.plot(xloss,yloss)
# plt.xlabel("loss (dB/km)")
# plt.ylabel("freqeuncy")
# plt.savefig('NDFISlosshist.png', dpi=200)
# plt.show()
# =============================================================================
# %% random draws from histogram 

countsnorm = np.zeros(np.size(counts))
for i in range(np.size(countsnorm)):
    countsnorm[i] = counts[i]/np.sum(counts)
countsnormo = np.zeros(np.size(countso))
for i in range(np.size(countsnormo)):
    countsnormo[i] = countso[i]/np.sum(countso)
countsnormd = np.zeros(np.size(countsd))
for i in range(np.size(countsnormd)):
    countsnormd[i] = countsd[i]/np.sum(countsd)
countsnormrep = np.zeros(np.size(countsrep))
for i in range(np.size(countsnormrep)):
    countsnormrep[i] = countsrep[i]/np.sum(countsrep)
def histdraw(counts, bins):
    np.random.seed()
    randnum = np.random.uniform(0,1)
    #print("random draw " + str(randnum))
    for i in range(np.size(counts)):
        if randnum < np.sum(np.split(counts, [i+1])[0]):
            #print("checksum " + str(np.sum(np.split(countsnorm, [i+1])[0])))
            drawnum = np.random.uniform(bins[i],bins[i+1])
            break
    return drawnum
# %%  ================================ set params and call main ================================
#dev = 1  # % deviation from baseline 
PchdBm = np.linspace(-10, 10, num = numpoints, dtype =float) 
#PchdBm = 0
#alpha = 0.25
Nspans = 3
Lspans = 100
num_breaks = 0 # set the number of fibre breaks
num_years = (num_breaks*374)/(Nspans*Lspans) # expected number of years for this number of fibre breaks given a rate of 1 break/374km/year
#alpha = NDFISlossnzmean
alpha = 0.2
#alpha = np.zeros(numpoints)
#for i in range(numpoints):
#    alpha[i] = histdraw(countsnorm, bins)
#alpha = NDFISlossnzmean
#Disp = np.random.normal(NDFISdispnzmean, NDFISdispnzvar**0.5, numpoints)
NLco = 1.27
NchRS = 101
#Disp = NDFISdispnzmean
Disp = 16.7
gain_target = alpha*Lspans
# =============================================================================
# gain_max = 26  # operator model example from GNPy - see eqpt_config.json 
# gain_min = 15
# nf_min = 6
# nf_max = 10 
# nf1 = nf_model(gain_min,gain_max,nf_min,nf_max )[0]
# nf2 = nf_model(gain_min,gain_max,nf_min,nf_max )[1]
# deltap = nf_model(gain_min,gain_max,nf_min,nf_max )[2]
# g1max = nf_model(gain_min,gain_max,nf_min,nf_max )[3]
# g1min = nf_model(gain_min,gain_max,nf_min,nf_max )[4]
# g1a = gain_target - deltap - (gain_max - gain_target)
# NF = lin2db(db2lin(nf1) + db2lin(nf2)/db2lin(g1a))  
# =============================================================================
NF = 5.5

# %% SNR variation due to ripple emulation

ripplepertmax = 0.1
ripplepertmin = -0.1
ripplepertmaxpowerdep = 0.4
ripplepertminpowerdep = -1.0

def rippledatagen(PchdBm,numspans):
    Gnli = np.empty([numspans,numpoints])
    ep = np.empty([numspans,1])
    lam = 1550
    Rs = 32
    f = 299792458/(lam*1e-9) # operating frequency [Hz]
    h = 6.63*1e-34  # Planck's constant [Js] 
    Pun = main(Lspans, 1, 157, 101, 201, alpha, Disp, PchdBm, NF, NLco,False)[0]
    Popt = PchdBm[np.argmax(Pun)] 
    Poptr = np.linspace(Popt-1.5, Popt+1.5, numpoints)
    Poptran = Poptr  +  np.random.uniform(ripplepertmin, ripplepertmax, numpoints)
    #Poptran = Poptr  +  (10**(np.random.uniform(ripplepertmaxpowerdep, ripplepertminpowerdep, numpoints)/10) - 1)*Poptr
    #Poptran = lin2db(db2lin(Poptr)+(10**(np.random.uniform(ripplepertmaxpowerdep, ripplepertminpowerdep, numpoints)/10) - 1)*db2lin(Poptr))
    for i in range(numspans):
        Gnli[i] = main(Lspans, 1, 157, 101, 201, alpha, Disp, Poptran, NF, NLco,False)[4]
        Poptran = Poptran  +  np.random.uniform(ripplepertmin, ripplepertmax, numpoints)
        #Poptran = Poptr  +  (10**(np.random.uniform(ripplepertmaxpowerdep, ripplepertminpowerdep, numpoints)/10) - 1)*Poptr
        #Poptran = lin2db(db2lin(Poptr)  + (10**(np.random.uniform(ripplepertmaxpowerdep, ripplepertminpowerdep, numpoints)/10) - 1)*db2lin(Poptr))
    ep = main(Lspans, numspans, 157, 101, 201, alpha, Disp, Poptran, NF, NLco,False)[5]    
    Pase = NF*h*f*(db2lin(gain_target) - 1)*Rs*1e9*numspans
    Gnli = np.sum(Gnli,axis=0)*(Nspans**ep)
    #Pch = 1e-3*db2lin(PchdBm/10)  # ^ [W]
    Pch = 1e-3*10**(Poptran/10)  # ^ [W]
    return lin2db((Pch)/(Pase + Gnli*Rs*1e9)), Popt
    
SNRripple, Popt = rippledatagen(PchdBm,10)
Pchripple = np.linspace(Popt-1.5, Popt+1.5, numpoints)


Poptr = np.linspace(Popt-1.5, Popt+1.5, numpoints)
test  = (10**(np.random.uniform(ripplepertmaxpowerdep, ripplepertminpowerdep, numpoints)/10) - 1)
test2  = (10**(-0.4/10) - 1)
plt.plot(Pchripple, SNRripple,'+')
plt.title('Power excursion data')
plt.xlabel("Pch(dBm)")
plt.ylabel("SNR(dB)")
plt.savefig('Apowerexcursion.png', dpi=200)
plt.show()
#np.savetxt('Pchripple10.csv', Pchripple, delimiter=',') 
#np.savetxt('SNRripple10.csv', SNRripple, delimiter=',') 

#np.savetxt('Pchnum75.csv', Pchripple, delimiter=',') 
#np.savetxt('SNRnum75.csv', SNRripple, delimiter=',') 

# %% routing alg bit
# import trained GP models
prmn = np.genfromtxt(open("prmn.csv", "r"), delimiter=",", dtype =float)
sigma = np.genfromtxt(open("sig.csv", "r"), delimiter=",", dtype =float)
sigrf = np.genfromtxt(open("sigrf.csv", "r"), delimiter=",", dtype =float)

# Dijkstra implementation: following https://www.youtube.com/watch?v=IG1QioWSXRI
def dijkstra(graph,start,goal):
    shortest_distance = {}
    predecessor = {}
    unseenNodes = graph
    infinity = 9999999
    path = []
    for node in unseenNodes:
        shortest_distance[node] = infinity
    shortest_distance[start] = 0
    while unseenNodes:
        minNode = None
        for node in unseenNodes:
            if minNode is None:
                minNode = node
            elif shortest_distance[node] < shortest_distance[minNode]:
                minNode = node
 
        for childNode, weight in graph[minNode].items():
            if weight + shortest_distance[minNode] < shortest_distance[childNode]:
                shortest_distance[childNode] = weight + shortest_distance[minNode]
                predecessor[childNode] = minNode
        unseenNodes.pop(minNode)
    currentNode = goal
    while currentNode != start:
        try:
            path.insert(0,currentNode)
            currentNode = predecessor[currentNode]
        except KeyError:
            print('Path not reachable')
            break
    path.insert(0,start)
    #if shortest_distance[goal] != infinity:
        #print('Shortest distance is ' + str(shortest_distance[goal]))
        #print('And the path is ' + str(path))
    return shortest_distance[goal], path
# define the network topology 

nodes = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14']
numnodes = np.size(nodes)
graph = {'1':{'2':2100,'3':3000,'8':4800},'2':{'1':2100,'3':1200,'4':1500},'3':{'1':3000,'2':1200,'6':3600},    
         '4':{'2':1500,'5':1200,'11':3900},'5':{'4':1200,'6':2400,'7':1200}, '6':{'3':3600,'5':2400,'10':2100,'14':3600},
         '7':{'5':1200,'8':1500,'10':2700}, '8':{'1':4800,'7':1500,'9':1500}, '9':{'8':1500,'10':1500,'12':600,'13':600},
         '10':{'6':2100,'7':2700,'9':1500}, '11':{'4':3900,'12':1200,'13':1500}, '12':{'9':600,'11':1200,'14':600},
         '13':{'9':600,'11':1500,'14':300}, '14':{'6':3600,'12':600,'13':300}
         } 
graphnorm = {'1':{'2':2100,'3':3000,'8':4800},'2':{'1':2100,'3':1200,'4':1500},'3':{'1':3000,'2':1200,'6':3600},    
         '4':{'2':1500,'5':1200,'11':3900},'5':{'4':1200,'6':2400,'7':1200}, '6':{'3':3600,'5':2400,'10':2100,'14':3600},
         '7':{'5':1200,'8':1500,'10':2700}, '8':{'1':4800,'7':1500,'9':1500}, '9':{'8':1500,'10':1500,'12':600,'13':600},
         '10':{'6':2100,'7':2700,'9':1500}, '11':{'4':3900,'12':1200,'13':1500}, '12':{'9':600,'11':1200,'14':600},
         '13':{'9':600,'11':1500,'14':300}, '14':{'6':3600,'12':600,'13':300}
         }                 
edges = {'1':{'2':0,'3':1,'8':2},'2':{'1':3,'3':4,'4':5},'3':{'1':6,'2':7,'6':8},    
         '4':{'2':9,'5':10,'11':11},'5':{'4':12,'6':13,'7':14}, '6':{'3':15,'5':16,'10':17,'14':18},
         '7':{'5':19,'8':20,'10':21}, '8':{'1':22,'7':23,'9':24}, '9':{'8':25,'10':26,'12':27,'13':28},
         '10':{'6':29,'7':30,'9':31}, '11':{'4':32,'12':33,'13':34}, '12':{'9':35,'11':36,'14':37},
         '13':{'9':38,'11':39,'14':40}, '14':{'6':41,'12':42,'13':43}
         }
numedges = 44
# %%
edgelens = np.empty([numedges,1])
count = 0
for key in graph:
    for key2 in graph.get(key):
        #print(graph.get(key).get(key2))
        edgelens[count] = graph.get(key).get(key2)
        count = count + 1

# %%
PchdBm = np.linspace(-10,10,numpoints)
#ripplepertmax = 0.1  # for fixed perturbation between spans 
#ripplepertmin = -0.1
gain_target2 = alpha*Lspans
numlam = 20 # initial expected number of wavelengths 
if datagen:
    def routingdatagen2(edgelen):
        lam = 1550
        Rs = 32
        f = 299792458/(lam*1e-9) # operating frequency [Hz]
        h = 6.63*1e-34  # Planck's constant [Js] 
        Pun = main(Lspans, 1, numlam, 101, 201, alpha, Disp, PchdBm, NF, NLco,False)[0] # }
        Popt = PchdBm[np.argmax(Pun)]                                                   # }could do this outside of function
        Poptind = int(np.argmax(Pun))                                                   # }
        Poptr = np.linspace(Popt-1.5, Popt+1.5, numpoints)
        #ripplepert = 0.1
        ripplepert = np.random.uniform(0.1,0.2)
        Poptran = Poptr  +  np.random.uniform(-ripplepert, ripplepert, numpoints)
        numspans = int(edgelen/Lspans)
        Gnli = np.empty([numspans,numpoints])
        for i in range(numspans):
            Gnli[i] = main(Lspans, 1, numlam, 101, 201, alpha, Disp, Poptran, NF, NLco,False)[4]
            ripplepert = np.random.uniform(0.1,0.2)
            Poptran = Poptran + np.random.uniform(-ripplepert, ripplepert, numpoints)
            if i == numspans-2:
                    Poptfinal = Poptran  # select the power that was put into the last span to calculate SNR
        Gnli = np.sum(Gnli,axis=0)
        Pase = NF*h*f*(db2lin(alpha*Lspans) - 1)*Rs*1e9*numspans
        Pch = 1e-3*10**(Poptfinal/10) 
        return lin2db((Pch)/(Pase*np.ones(numpoints) + Gnli*Rs*1e9)), Popt, Poptind
        
    linkSNR = np.empty([numedges,numpoints])
    for i in range(numedges):
        linkSNR[i], linkPopt, linkPoptind = routingdatagen2(edgelens[i])
    
    linkPch = np.transpose(np.linspace(linkPopt-1.5, linkPopt+1.5, numpoints).reshape(100,1))
    linkPchtest = np.linspace(linkPopt-1.5, linkPopt+1.5, numpoints)
    
    #np.savetxt('linkSNR.csv', linkSNR, delimiter=',') 
    #np.savetxt('linkPch.csv', linkPch, delimiter=',') 

if datagen == False:
    Pun = main(Lspans, 1, numlam, 101, 201, alpha, Disp, PchdBm, NF, NLco,False)[0] # }
    Popt = PchdBm[np.argmax(Pun)]                                                   # }could do this outside of function
    linkPoptind = int(np.argmax(Pun)) 
    linkSNR = np.genfromtxt(open("linkSNR.csv", "r"), delimiter=",", dtype =float)
    linkPch = np.genfromtxt(open("linkPch.csv", "r"), delimiter=",", dtype =float)


# %%
    
plt.plot(linkPch, linkSNR[41],'o',color='y')
plt.plot(linkPch[linkPoptind], linkSNR[41][linkPoptind],'+',color = 'r')
plt.xlabel("Pch (dBm)")
plt.ylabel("SNR (dB)")
plt.savefig('testSNR.png', dpi=200,bbox_inches='tight')
plt.show()

# %%

def BERcalc(M, SNR):
        if M == 4: 
            BER = 0.5*special.erfc(SNR**0.5)
        elif M == 16:
            BER = (3/8)*special.erfc(((2/5)*SNR)**0.5) + (1/4)*special.erfc(((18/5)*SNR)**0.5) - (1/8)*special.erfc((10*SNR)**0.5)
        elif M == 64:
            BER = (7/24)*special.erfc(((1/7)*SNR)**0.5) + (1/4)*special.erfc(((9/7)*SNR)**0.5) - (1/24)*special.erfc(((25/7)*SNR)**0.5) - (1/24)*special.erfc(((25/7)*SNR)**0.5) + (1/24)*special.erfc(((81/7)*SNR)**0.5) - (1/24)*special.erfc(((169/7)*SNR)**0.5) 
        return BER
    
def getlinklen(shpath,graph,edges):
        linklen = np.empty([len(shpath)-1,1])
        link = []
        for i in range(len(shpath)-1):
            linklen[i] = float((graph.get(shpath[i])).get(shpath[i+1]))
            link.append((edges.get(shpath[i])).get(shpath[i+1]))
        return linklen, link
    
def requestgen(graph):
            src = random.choice(list(graph.keys()))
            des = random.choice(list(graph.keys()))
            while des == src:
                des = random.choice(list(graph.keys()))
            return src, des

# pre-determine 100 random requests 
numreq = 100      
rsrc = []
rdes = []
for i in range(numreq):
    rsc, rds = requestgen(graph)
    rsrc.append(rsc)
    rdes.append(rds)   
    
# %%
# generate shortest path between each pair of nodes and store the path and distance

def basicrta(FT, graph, edges, Rsource, Rdest, showres):

    dis = []
    path = []
    for i in range(numnodes):    
        for j in range(numnodes): 
            d, p = dijkstra({'1':{'2':2100,'3':3000,'8':4800},'2':{'1':2100,'3':1200,'4':1500},'3':{'1':3000,'2':1200,'6':3600},    
             '4':{'2':1500,'5':1200,'11':3900},'5':{'4':1200,'6':2400,'7':1200}, '6':{'3':3600,'5':2400,'10':2100,'14':3600},
             '7':{'5':1200,'8':1500,'10':2700}, '8':{'1':4800,'7':1500,'9':1500}, '9':{'8':1500,'10':1500,'12':600,'13':600},
             '10':{'6':2100,'7':2700,'9':1500}, '11':{'4':3900,'12':1200,'13':1500}, '12':{'9':600,'11':1200,'14':600},
             '13':{'9':600,'11':1500,'14':300}, '14':{'6':3600,'12':600,'13':300}
             }, nodes[i], nodes[j])
            if i == j:
                continue  # don't include lightpaths of length 0
            else:
                dis.append(d)
                path.append(p)
    pathdists = []
    links = []                                                
    for i in range(np.size(path)):
        pathdists.append(getlinklen(path[i],graphnorm,edges)[0])
        links.append(getlinklen(path[i],graph,edges)[1])
    
    estlam = np.zeros([numedges,numlam]) # 0 for empty, 1 for occupied
    reqlams = 0
    conten = 0
    numreq = np.size(Rsource)
    randdist = []
    for i in range(numreq):
        # update for online learning 
        #  choose random source and destination nodes 

        #Rsource, Rdest = requestgen(graph)
    
        # find corresponding path index
        def srcdestcheck(path,src,dest):
            if path[0] == src and path[-1] == dest:
                return True  
        
        randpathind = [j for j in range(np.size(path)) if  srcdestcheck(path[j], Rsource[i], Rdest[i])][0]
        #print("selected request index: " + str(randpathind))
        randedges = links[randpathind]  # selected edges for request 
        randdist.append(sum(pathdists[randpathind]))
        lamconten = False
        testlamslot = [np.where(estlam[randedges[k]]==0) for k in range(np.size(randedges))]
        for g in range(np.size(randedges)):
            if np.size(testlamslot[g]) == 0:
                #print("all wavelength slots are full")
                lamconten = True
                continue
        if lamconten:
            conten = conten + 1
            continue
        lamslot = [np.where(estlam[randedges[k]]==0)[0][0] for k in range(np.size(randedges))] # first available wavelength slot for this edge
        
        # need to check SNR for all the edges in the path
        edgesuc = 0
        for j in range(np.size(linkSNR[randedges],0)):
            if linkSNR[randedges][j][linkPoptind] > FT:
                #print("reach satisfied on edge " + str(randedges[j]))
                edgesuc = edgesuc + 1
            else:
                #print("request " + str(i) + " denied -- insufficient reach on edge " + str(randedges[j]))
                break   
        if edgesuc == np.size(linkSNR[randedges],0):
            #print("successfully allocated request " + str(i))
            reqlams = reqlams + 1
            for l in range(len(randedges)):
                estlam[randedges[l]][lamslot[l]] = 1
    ava = (reqlams/numreq)*100 
    tottime = (sum(randdist)*1e3*1.468)/299792458
    if showres:
        print("Normal availability = " + str(ava) + "%") 
        print("Normal total traversal time = " + str('%.2f' % tottime) + "s")
    return ava, estlam, reqlams, tottime, conten

#ava, estl, reql, tottime = basicrta(FT,graph,edges,rsrc,rdes, False)

def varrta(FT,graph,edges,Rsource,Rdest,showres):
    dis = []
    path = []
    for i in range(numnodes):    
        for j in range(numnodes): 
            d, p = dijkstra({'1':{'2':gwt[0][0],'3':gwt[1][0],'8':gwt[2][0]},'2':{'1':gwt[3][0],'3':gwt[4][0],'4':gwt[5][0]},'3':{'1':gwt[6][0],'2':gwt[7][0],'6':gwt[8][0]},    
             '4':{'2':gwt[9][0],'5':gwt[10][0],'11':gwt[11][0]},'5':{'4':gwt[12][0],'6':gwt[13][0],'7':gwt[14][0]}, '6':{'3':gwt[15][0],'5':gwt[16][0],'10':gwt[17][0],'14':gwt[18][0]},
             '7':{'5':gwt[19][0],'8':gwt[20][0],'10':gwt[21][0]}, '8':{'1':gwt[22][0],'7':gwt[23][0],'9':gwt[24][0]}, '9':{'8':gwt[25][0],'10':gwt[26][0],'12':gwt[27][0],'13':gwt[28][0]},
             '10':{'6':gwt[29][0],'7':gwt[30][0],'9':gwt[31][0]}, '11':{'4':gwt[32][0],'12':gwt[33][0],'13':gwt[34][0]}, '12':{'9':gwt[35][0],'11':gwt[36][0],'14':gwt[37][0]},
             '13':{'9':gwt[38][0],'11':gwt[39][0],'14':gwt[40][0]}, '14':{'6':gwt[41][0],'12':gwt[42][0],'13':gwt[43][0]}
             }, nodes[i], nodes[j])
            if i == j:
                continue  # don't include lightpaths of length 0
            else:
                dis.append(d)
                path.append(p)
    pathdists = []
    links = []                                                
    for i in range(np.size(path)):
        pathdists.append(getlinklen(path[i],graphnorm,edges)[0])
        links.append(getlinklen(path[i],graph,edges)[1])
    
    estlam = np.zeros([numedges,numlam]) # 0 for empty, 1 for occupied
    reqlams = 0
    conten = 0
    numreq = np.size(Rsource)
    randdist = []
    for i in range(numreq):
        # update for online learning 
        #  choose random source and destination nodes 
        
        #Rsource, Rdest = requestgen(graph)
        # find corresponding path index
        def srcdestcheck(path,src,dest):
            if path[0] == src and path[-1] == dest:
                return True  

        randpathind = [j for j in range(np.size(path)) if  srcdestcheck(path[j], Rsource[i], Rdest[i])][0]
        #print("selected request index: " + str(randpathind))
        randedges = links[randpathind]  # selected edges for request 
        randdist.append(sum(pathdists[randpathind]))
        lamconten = False
        testlamslot = [np.where(estlam[randedges[k]]==0) for k in range(np.size(randedges))]
        for g in range(np.size(randedges)):
            if np.size(testlamslot[g]) == 0:
                #print("all wavelength slots are full")
                lamconten = True
                continue
        if lamconten:
            conten = conten + 1
            continue
        lamslot = [np.where(estlam[randedges[k]]==0)[0][0] for k in range(np.size(links[randpathind]))] # first available wavelength slot for this edge
        
        # need to check SNR for all the edges in the path
        edgesuc = 0
        for j in range(np.size(linkSNR[randedges],0)):
            if linkSNR[randedges][j][linkPoptind] > FT:
                #print("reach satisfied on edge " + str(randedges[j]))
                edgesuc = edgesuc + 1
            else:
                #print("request " + str(i) + " denied -- insufficient reach on edge " + str(randedges[j]))
                break   
        if edgesuc == np.size(linkSNR[randedges],0):
            #print("successfully allocated request " + str(i))
            reqlams = reqlams + 1
            for l in range(len(randedges)):
                estlam[randedges[l]][lamslot[l]] = 1
    ava = (reqlams/numreq)*100 
    tottime = ((sum(randdist)*1e3*1.468)/299792458)[0]
    if showres:
        print("Variance-aided availability = " + str(ava) + "%") 
        print("Variance-aided total traversal time = " + str('%.2f' % tottime) + "s")
    return ava, estlam, reqlams, tottime, conten
    

# find optimum from the predictive mean
prmnopt = [np.argmax(prmn[i]) for i in range(np.size(prmn,0))]
gsige = [sigma[i][prmnopt[i]] for i in range(np.size(prmn,0))]  # use predictive mean to get optimum Pch
gwte = [edgelens[i]*gsige[i] for i in range(np.size(prmn,0))]
gsig = [sigma[i][linkPoptind] for i in range(np.size(sigma,0))]
gwt = [edgelens[i]*gsig[i] for i in range(np.size(sigma,0))]
graphvar = {'1':{'2':gwt[0][0],'3':gwt[1][0],'8':gwt[2][0]},'2':{'1':gwt[3][0],'3':gwt[4][0],'4':gwt[5][0]},'3':{'1':gwt[6][0],'2':gwt[7][0],'6':gwt[8][0]},    
             '4':{'2':gwt[9][0],'5':gwt[10][0],'11':gwt[11][0]},'5':{'4':gwt[12][0],'6':gwt[13][0],'7':gwt[14][0]}, '6':{'3':gwt[15][0],'5':gwt[16][0],'10':gwt[17][0],'14':gwt[18][0]},
             '7':{'5':gwt[19][0],'8':gwt[20][0],'10':gwt[21][0]}, '8':{'1':gwt[22][0],'7':gwt[23][0],'9':gwt[24][0]}, '9':{'8':gwt[25][0],'10':gwt[26][0],'12':gwt[27][0],'13':gwt[28][0]},
             '10':{'6':gwt[29][0],'7':gwt[30][0],'9':gwt[31][0]}, '11':{'4':gwt[32][0],'12':gwt[33][0],'13':gwt[34][0]}, '12':{'9':gwt[35][0],'11':gwt[36][0],'14':gwt[37][0]},
             '13':{'9':gwt[38][0],'11':gwt[39][0],'14':gwt[40][0]}, '14':{'6':gwt[41][0],'12':gwt[42][0],'13':gwt[43][0]}
             }        
graphvared = {'1':{'2':gwte[0][0],'3':gwte[1][0],'8':gwte[2][0]},'2':{'1':gwte[3][0],'3':gwte[4][0],'4':gwte[5][0]},'3':{'1':gwte[6][0],'2':gwte[7][0],'6':gwte[8][0]},    
             '4':{'2':gwte[9][0],'5':gwte[10][0],'11':gwte[11][0]},'5':{'4':gwte[12][0],'6':gwte[13][0],'7':gwte[14][0]}, '6':{'3':gwte[15][0],'5':gwte[16][0],'10':gwte[17][0],'14':gwte[18][0]},
             '7':{'5':gwte[19][0],'8':gwte[20][0],'10':gwte[21][0]}, '8':{'1':gwte[22][0],'7':gwte[23][0],'9':gwte[24][0]}, '9':{'8':gwte[25][0],'10':gwte[26][0],'12':gwte[27][0],'13':gwte[28][0]},
             '10':{'6':gwte[29][0],'7':gwte[30][0],'9':gwte[31][0]}, '11':{'4':gwte[32][0],'12':gwt[33][0],'13':gwt[34][0]}, '12':{'9':gwte[35][0],'11':gwte[36][0],'14':gwte[37][0]},
             '13':{'9':gwte[38][0],'11':gwte[39][0],'14':gwte[40][0]}, '14':{'6':gwte[41][0],'12':gwte[42][0],'13':gwte[43][0]}
             }    

#avav, estlamv, reqlamsv, tottimev  = varrta(FT,graphvar,edges,rsrc,rdes,False)

# %%
def varrtap(FT,graph,edges,Rsource,Rdest,showres):
    dis = []
    path = []
    for i in range(numnodes):    
        for j in range(numnodes): 
            d, p = dijkstra({'1':{'2':gwte[0][0],'3':gwte[1][0],'8':gwte[2][0]},'2':{'1':gwte[3][0],'3':gwte[4][0],'4':gwte[5][0]},'3':{'1':gwte[6][0],'2':gwte[7][0],'6':gwte[8][0]},    
             '4':{'2':gwte[9][0],'5':gwte[10][0],'11':gwte[11][0]},'5':{'4':gwte[12][0],'6':gwte[13][0],'7':gwte[14][0]}, '6':{'3':gwte[15][0],'5':gwte[16][0],'10':gwte[17][0],'14':gwte[18][0]},
             '7':{'5':gwte[19][0],'8':gwte[20][0],'10':gwte[21][0]}, '8':{'1':gwte[22][0],'7':gwte[23][0],'9':gwte[24][0]}, '9':{'8':gwte[25][0],'10':gwte[26][0],'12':gwte[27][0],'13':gwte[28][0]},
             '10':{'6':gwte[29][0],'7':gwte[30][0],'9':gwte[31][0]}, '11':{'4':gwte[32][0],'12':gwt[33][0],'13':gwt[34][0]}, '12':{'9':gwte[35][0],'11':gwte[36][0],'14':gwte[37][0]},
             '13':{'9':gwte[38][0],'11':gwte[39][0],'14':gwte[40][0]}, '14':{'6':gwte[41][0],'12':gwte[42][0],'13':gwte[43][0]}
             }, nodes[i], nodes[j])
            if i == j:
                continue  # don't include lightpaths of length 0
            else:
                dis.append(d)
                path.append(p)
    pathdists = []
    links = []                                                
    for i in range(np.size(path)):
        pathdists.append(getlinklen(path[i],graphnorm,edges)[0])
        links.append(getlinklen(path[i],graph,edges)[1])
    
    estlam = np.zeros([numedges,numlam]) # 0 for empty, 1 for occupied
    reqlams = 0
    conten = 0
    numreq = np.size(Rsource)
    randdist = []
    conf = []
    for i in range(numreq):
        # update for online learning 
        #  choose random source and destination nodes 
        #Rsource, Rdest = requestgen(graph)
        # find corresponding path index
        def srcdestcheck(path,src,dest):
            if path[0] == src and path[-1] == dest:
                return True  
        randpathind = [j for j in range(np.size(path)) if  srcdestcheck(path[j], Rsource[i], Rdest[i])][0]
        #print("selected request index: " + str(randpathind))
        randedges = links[randpathind]  # selected edges for request 
        randdist.append(sum(pathdists[randpathind]))
        lamconten = False
        testlamslot = [np.where(estlam[randedges[k]]==0) for k in range(np.size(randedges))]
        for g in range(np.size(randedges)):
            if np.size(testlamslot[g]) == 0:
                #print("all wavelength slots are full")
                lamconten = True
                continue
        if lamconten:
            conten = conten + 1
            continue
        lamslot = [np.where(estlam[randedges[k]]==0)[0][0] for k in range(np.size(links[randpathind]))] # first available wavelength slot for this edge
        # need to check SNR for all the edges in the path
        edgesuc = 0
        for j in range(np.size(linkSNR[randedges],0)): # for each edge in the path
            #if linkSNR[randedges][j][prmnopt[randedges[j]]] > FT:
            if prmn[randedges][j][prmnopt[randedges[j]]] > FT:
                conf.append((prmn[randedges][j][prmnopt[randedges[j]]] - FT)/gsige[randedges[j]])
                #print("reach satisfied on edge " + str(randedges[j]))
                edgesuc = edgesuc + 1
            else:
                #print("request " + str(i) + " denied -- insufficient reach on edge " + str(randedges[j]))
                break   
        if edgesuc == np.size(linkSNR[randedges],0):
            #print("successfully allocated request " + str(i))
            reqlams = reqlams + 1
            for l in range(len(randedges)):
                
                estlam[randedges[l]][lamslot[l]] = 1
    ava = (reqlams/numreq)*100 
    tottime = ((sum(randdist)*1e3*1.468)/299792458)[0]
    if showres:
        print("Variance-aided availability = " + str(ava) + "%") 
        print("Variance-aided total traversal time = " + str('%.2f' % tottime) + "s")
    return ava, estlam, reqlams, tottime, conf, conten

test1, test2, test3, test4, test5, test6 = varrtap(2.11,graph,edges,rsrc,rdes,False)



# %%

M = 64
if M == 4:
    FT = 2.11  # QPSK
elif M == 16:
    FT = 4.65   # 16-QAM
elif M == 64:
    FT = 11.6   # 64-QAM
else:
    print("unrecognised modulation format -- set FT for QPSK")
    FT = 2.11  # QPSK



# ========== this loop resets the network loading after numreqs, numtests times  ===========
def testrout(numtests,showres):
#     
    ava = np.empty([numtests,1])
    tottime = np.empty([numtests,1])
    conten = np.empty([numtests,1])
    avav = np.empty([numtests,1])
    tottimev = np.empty([numtests,1])
    contenv = np.empty([numtests,1])
    avavp = np.empty([numtests,1])
    tottimevp = np.empty([numtests,1])
    contenvp = np.empty([numtests,1])
    numreq = 1000 
    for i in range(numtests):
        # pre-determine 100 random requests 
        
        rsrct = []
        rdest = []
        for _ in range(numreq):
            rsct, rdst = requestgen(graph)
            rsrct.append(rsct)
            rdest.append(rdst)
        ava[i], _, _, tottime[i],conten[i]   = basicrta(FT,graphvar,edges,rsrct,rdest,showres)
        avav[i], _, _, tottimev[i], contenv[i]  = varrta(FT,graphvar,edges,rsrct,rdest,showres)
        avavp[i], _, _, tottimevp[i], conf, contenvp[i]  = varrtap(FT,graphvared,edges,rsrct,rdest,showres)
    return ava, tottime, conten, avav, tottimev,contenv, avavp, tottimevp,contenvp
# %% =============================================================================

ava, tottime,conten, avav, tottimev,contenv, avavp, tottimevp,contenvp = testrout(10,False)

avaave = np.mean(ava)
ttave = np.mean(tottime)
wavconave = np.mean(conten)
avaavev = np.mean(avav)
ttavev = np.mean(tottimev)
wavconavev = np.mean(contenv)
avaavevp = np.mean(avavp)
ttavevp = np.mean(tottimevp)
wavconavevp = np.mean(contenvp)

print("Normal average availability " + str(avaave) + "%")
print("Normal average latency " + str('%.2f' % ttave) + "s")
print("Variance-aided average availability " + str(avaavev) + "%")
print("Variance-aided average latency " + str('%.2f' % ttavev) + "s")
print("Variance-aided power-adjusted average availability " + str(avaavevp) + "%")
print("Variance-aided power-adjusted average latency " + str('%.2f' % ttavevp) + "s")

#%% ====================================================================================

GNmodelresbase = main(Lspans, Nspans, 157, 101, 201, alpha, Disp, PchdBm, NF, NLco,False)
#GNmodelresbase = main(100, 10, 157, 101, 201, 0.2, 16.7, 0, 6.0, 1.27)
#GNmodelresp = main(Lspans, Nspans, 157, 101, 201, alpha, Disp, PchdBm, NF, NLco)
#GNmodelresn = main(Lspans, Nspans, 157, 101, 201, alpha, Disp, PchdBm, NF, NLco)

SNRanalyticalbase = GNmodelresbase[0]
SNRanalyticalRSbase = GNmodelresbase[1]
SNRanalyticalRS2base = GNmodelresbase[2]
OSNRanalytical = GNmodelresbase[3]
SNRanalyticalO = GNmodelresbase[6]

numdraws = np.linspace(0,numpoints-1, numpoints)

# %% ================================ Mutual information estimation ===========================================
  
# import constellation shapes from MATLAB-generated csv files 
if constellationimport:  
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

# %% ================================ Estimate MI ================================ 
# set modulation format order and number of terms used in Gauss-Hermite quadrature
M = 4
L = 10

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

MIripple = findMI(SNRripple)

# %%

plt.plot(Pchripple, MIripple,'+')
plt.show()
# =============================================================================
# start_time = time.time()
# IxyNy = findMI(SNRanalyticalbase)
# IxyRS = findMI(SNRanalyticalRSbase)
# IxyRS2 = findMI(SNRanalyticalRS2base)
# duration = time.time() - start_time
# print("MI calculation duration: " + str(duration))
# =============================================================================

# %% ================================== Reach calculation ==================================
if reachcalculation:
# find the BER from: On the Bit Error Probability of QAM Modulation - Michael P. Fitz 
    PchreachdBm = np.linspace(-5,5,numpoints)
    lossreach = NDFISlossoverallnzmean
    dispreach = NDFISdispnzmean
    # =============================================================================
    SNRNYr = main(Lspans, Nspans, 157, 101, 201, lossreach,dispreach, PchreachdBm, NF, NLco,False)[0]
    SNRRSr = main(Lspans, Nspans, 157, 101, 201, lossreach, dispreach, PchreachdBm, NF, NLco,False)[1]
    SNRRS2r = main(Lspans, Nspans, 157, 101, 201, lossreach, dispreach, PchreachdBm, NF, NLco,False)[2]
    # Ny = 0 for Nyquist, 1 for RS and 2 for RS2
    def reachcalc(Ny, P, M):
        FECthreshold = 2e-2
        BER = np.zeros(numpoints)
        Ns = 20 # start at 2 spans because of the denominator of (22) in Poggiolini's GN model paper - divide by ln(Ns) = 0 for Ns = 1
        while BER[0] < FECthreshold:               
            SNR = main(Lspans, Ns, 157, 101, 201, alpha, Disp, P, NF, NLco,False)[Ny]                
            if M == 4: 
                BER = 0.5*special.erfc(SNR**0.5)                    
            elif M == 16:
                BER = (3/8)*special.erfc(((2/5)*SNR)**0.5) + (1/4)*special.erfc(((18/5)*SNR)**0.5) - (1/8)*special.erfc((10*SNR)**0.5)                    
            elif M == 64:
                BER = (7/24)*special.erfc(((1/7)*SNR)**0.5) + (1/4)*special.erfc(((9/7)*SNR)**0.5) - (1/24)*special.erfc(((25/7)*SNR)**0.5) - (1/24)*special.erfc(((25/7)*SNR)**0.5) + (1/24)*special.erfc(((81/7)*SNR)**0.5) - (1/24)*special.erfc(((169/7)*SNR)**0.5)     
            else:
                print("unrecognised modulation format")    
            Ns = Ns + 1
        return Ns
    
    test = reachcalc(0, 0, 64)
    
    
    #reachNy = np.zeros(numpoints)
    #reachRS = np.zeros(numpoints)
    #reachRS2 = np.zeros(numpoints)
    #for i in range(np.size(PchreachdBm)): 
    #    reachNy[i] = reachcalc(0, PchreachdBm[i])
    #    reachRS[i] = reachcalc(1, PchreachdBm[i])
    #    reachRS2[i] = reachcalc(2, PchreachdBm[i])
#        SNRRS = main(Lspans, Nspans, 157, 101, 201, NDFISlossoverallnzmean, NDFISdispnzmean, PchdBm, NF, NLco)[1]
#        SNRRS2 = main(Lspans, Nspans, 157, 101, 201, NDFISlossoverallnzmean, NDFISdispnzmean, PchdBm, NF, NLco)[2]
    
# %% ================================== plotting =================================
GNPysnr = np.genfromtxt(open("SNRGNPy.csv", "r"), delimiter=",", dtype =float)
GNPyosnr = np.genfromtxt(open("OSNRGNPy.csv", "r"), delimiter=",", dtype =float)

# =============================================================================
# #plt.plot(PchdBm, IxyRS,label = '50GHz non-Nyquist ')
# #plt.plot(PchdBm, IxyNy,label = 'Nyquist ')
# #plt.plot(PchdBm, IxyRS2,label = '25GHz non-Nyquist ')
# plt.plot(numdraws, IxyNy,label = 'Nyquist ')
# plt.plot(numdraws, IxyRS,label = '50GHz non-Nyquist ')
# plt.plot(numdraws, IxyRS2,label = '25GHz non-Nyquist ')
# 
# #plt.plot(NDFISlossoverallnz, '*',label = 'Overall')
# plt.legend()
# plt.ylabel('MI (bits) ')
# # # #plt.xlabel('loss coefficient (dB/km)')
# # # #plt.xlabel('EDFA NF (dB)')
# # # #plt.xlabel('Pch (dBm)')
# #plt.xlabel('Pch (dBm)')
# plt.xlabel("draw index")
# # # #plt.xlabel('dispersion (ps/nm*km)')
# plt.title('NDFIS loss distribution MI for 16QAM P=0dBm ')
# plt.grid()
# #plt.savefig('NDFISlossdistribtionMI16qamP0.png', dpi=200)
# plt.show()
# =============================================================================

# %% SNR plotting bit 
#Pchgnpy = np.linspace(-10,10,np.size(GNPysnr))

# =============================================================================
# plt.plot(PchdBm, SNRanalyticalO, label = 'Nyquist')
# plt.plot(PchdBm,GNPysnr, label='GNPy')
# plt.legend()
# plt.ylabel('SNR (dB)')
# plt.xlabel('loss coefficient (dB/km)')
# #plt.xlabel('EDFA NF (dB)')
# plt.xlabel('Pch (dBm)')
# #plt.xlabel('Draw index')
# # #plt.xlabel('dispersion (ps/nm*km)')
# plt.title('SNR 3 spans')
# #plt.grid()
# plt.savefig('JoshvsGNPyvariablegain.png', dpi=200)
# plt.show()
# # # 
# plt.plot(PchdBm, OSNRanalytical, label = 'Josh')
# plt.plot(PchdBm,GNPyosnr, label='GNPy')
# plt.legend()
# plt.ylabel('OSNR (dB)')
# # #plt.xlabel('loss coefficient (dB/km)')
# # # #plt.xlabel('EDFA NF (dB)')
# plt.xlabel('Pch (dBm)')
# # # #plt.xlabel('Draw index')
# # # # #plt.xlabel('dispersion (ps/nm*km)')
# plt.title('OSNR 3 spans')
# #plt.grid()
# plt.savefig('JoshvsGNPyvariablegainOSNR.png', dpi=200)
# plt.show()
# =============================================================================

# %%

#============================================== Non-Nyquist 50GHz ==============================================


# =============================================================================
# plt.plot(PchdBm, SNRanalyticalRSbase, label = 'NN 50GHz 2 channels')
# # #plt.plot(PchdBm, SNRanalyticalRSp, label = 'NN 50GHz p')
# plt.plot(PchdBm, SNRanalyticalRSp, label = 'NN 50GHz 3 channels')
# # #plt.plot(PchdBm, SNRanalyticalRSn, label = 'NN 50GHz n')
# plt.plot(PchdBm, SNRanalyticalRSn, label = 'NN 50GHz 4 channels')
# # #plt.plot(PchdBm, SNRanalyticalRS2base, label = 'Analytical non-Nyquist 25GHz')
# # 
# # #plt.plot(al, SNRMC, label = 'Monte Carlo')
# plt.legend()
# plt.ylabel('SNR (dB)')
# #plt.xlabel('loss coefficient (dB/km)')
# #plt.xlabel('EDFA NF (dB)')
# plt.xlabel('Pch (dBm)')
# #plt.xlabel('dispersion (ps/nm*km)')
# #plt.title('Channel loading NN ' + str(dev) + '%')
# plt.title('Non-Nyquist channel loading large spacing')
# plt.grid()
# plt.savefig('SNRchannelloadingNNlargsplow.png', dpi=200)
# plt.show()
# =============================================================================

    
#============================================== Non-Nyquist 25GHz ==============================================

# =============================================================================
# plt.plot(PchdBm, SNRanalyticalRS2base, label = 'NN 25GHz 201 channels')
# #plt.plot(PchdBm, SNRanalyticalp, label = 'Nyquist p')
# #plt.plot(PchdBm, SNRanalyticalRSp, label = 'NN 50GHz p')
# plt.plot(PchdBm, SNRanalyticalRS2p, label = 'NN 25GHz 101 channels')
# #plt.plot(PchdBm, SNRanalyticaln, label = 'Nyquist n')
# #plt.plot(PchdBm, SNRanalyticalRSn, label = 'NN 50GHz n')
# plt.plot(PchdBm, SNRanalyticalRS2n, label = 'NN 25GHz 41 channels')
# #plt.plot(PchdBm, SNRanalyticalRS2base, label = 'Analytical non-Nyquist 25GHz')
# 
# #plt.plot(al, SNRMC, label = 'Monte Carlo')
# plt.legend()
# plt.ylabel('SNR (dB)')
# #plt.xlabel('loss coefficient (dB/km)')
# #plt.xlabel('EDFA NF (dB)')
# plt.xlabel('Pch (dBm)')
# #plt.xlabel('dispersion (ps/nm*km)')
# #plt.title('Channel loading NN ' + str(dev) + '%')
# plt.title('Non-Nyquist channel loading 25GHz ')
# plt.savefig('SNRchannelloadingNN25.png', dpi=200)
# plt.grid()
# plt.show()
# =============================================================================


#============================================== Nyquist vs non-Nyquist comparison ============================================

# =============================================================================
# plt.plot(PchdBm, SNRanalyticalbase, label = 'Nyquist 157 channels')
# plt.plot(PchdBm, SNRanalyticalRSbase, label = 'NN 50GHz 101 channels')
# plt.plot(PchdBm, SNRanalyticalRS2base, label = 'NN 25GHz 201 channels')
# 
# plt.legend()
# plt.ylabel('SNR (dB)')
# #plt.xlabel('loss coefficient (dB/km)')
# #plt.xlabel('EDFA NF (dB)')
# plt.xlabel('Pch (dBm)')
# #plt.xlabel('dispersion (ps/nm*km)')
# #plt.title('Channel loading NN ' + str(dev) + '%')
# plt.title('SNR channel spacing comparison ')
# plt.savefig('SNRchannelspacingcomparison.png', dpi=200)
# plt.grid()
# plt.show()
# =============================================================================
 


















    
