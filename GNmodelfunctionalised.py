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
#import komm
import multiprocessing
import time 
import pystan
from scipy import special
from scipy.stats import iqr
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF
from GHquad import GHquad
from NFmodelGNPy import nf_model
from NFmodelGNPy import lin2db
from NFmodelGNPy import db2lin
#from scipy.stats import invgamma
#from scipy.stats import gamma

NDFISimport = True
repairlossimport = True
constellationimport = True
reachcalculation = False
#addnoise = True



asediffvar = 1e-6*2
TRxSNR = db2lin(26) # TRX SNR [TRxdiffmean = db2lin(1.0)
TRxdiffvar = 0.252*2
snrdiffvar = 0.5
powerdiffvar = 0.2
#numpoints = 50
numpoints = 1000

def main(Ls, Ns, NchNy, NchRS, NchRS2, al, D, PchdBm, NF, gam, addnoise ):
    
    
    #  ################### equipment characteristics ####################

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
# =============================================================================
#plt.hist(bins[:-1], bins, weights=counts)
#plt.xlabel("loss (dB/km)")
#plt.ylabel("freqeuncy")
#plt.title("fibre loss")
#plt.savefig('NDFISlosshist20.png', dpi=200)
#plt.show()
# =============================================================================

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
num_breaks = 1 # set the number of fibre breaks
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
gain_max = 26  # operator model example from GNPy - see eqpt_config.json 
gain_min = 15
nf_min = 6
nf_max = 10 
nf1 = nf_model(gain_min,gain_max,nf_min,nf_max )[0]
nf2 = nf_model(gain_min,gain_max,nf_min,nf_max )[1]
deltap = nf_model(gain_min,gain_max,nf_min,nf_max )[2]
g1max = nf_model(gain_min,gain_max,nf_min,nf_max )[3]
g1min = nf_model(gain_min,gain_max,nf_min,nf_max )[4]
g1a = gain_target - deltap - (gain_max - gain_target)
NF = lin2db(db2lin(nf1) + db2lin(nf2)/db2lin(g1a))  
#NF = 5.0
# =========================== differing spans bit ==================================

def spanvar(PchdBm):
    alpha = np.zeros(Nspans)
    Disp = np.zeros(Nspans)
    ep = np.zeros(Nspans)
    gain_target = np.zeros(Nspans)
    Gnlispanvar = np.zeros((Nspans,numpoints))
    ep = np.zeros((Nspans,numpoints))
    lam = 1550
    Rs = 32
    f = 299792458/(lam*1e-9) # operating frequency [Hz]
    h = 6.63*1e-34  # Planck's constant [Js] 
    for i in range(Nspans):  # draw the span loss from a distribution 
        alpha[i] = histdraw(countsnorm, bins) # + histdraw(countsnormrep, binsrep)
        Disp[i] = histdraw(countsnormd, binsd)
    gain_target = alpha*Lspans 
    g1a = gain_target - deltap*np.ones(Nspans) - (gain_max*np.ones(Nspans) - gain_target)
    NF = lin2db(db2lin(nf1) + db2lin(nf2)/db2lin(g1a)) 
    NFl = 10**(NF/10) 
    Gl = 10**(gain_target/10)
    
    for i in range(Nspans):
        Gnlispanvar[i] = main(Lspans, Nspans, 157, 101, 201, alpha[i], Disp[i], PchdBm, NF[i], NLco,False)[4]
        ep[i] = main(Lspans, Nspans, 157, 101, 201, alpha[i], Disp[i], PchdBm, NF[i], NLco,False)[5]
    Pasespanvar = NFl*h*f*(Gl - 1)*Rs*1e9
    Pasech = np.sum(Pasespanvar)
    Gnli = np.sum(Gnlispanvar,axis=0)*(Nspans**(np.mean(ep)))
    Pch = 1e-3*10**(PchdBm/10)  # ^ [W]
    return 10*np.log10((Pch)/(Pasech + Gnli*Rs*1e9))

numsweeps = 20
SNRdataset = []
for _ in range(numsweeps):
    SNRdataset = np.append(SNRdataset,spanvar(PchdBm))

PchdBmrs = []
for _ in range(numsweeps):
    PchdBmrs.append(PchdBm)
PchdBmrs = np.reshape(PchdBmrs,numpoints*numsweeps)
#SNRtest1 = spanvar(PchdBm)
#SNRtest2 = spanvar(PchdBm)
#SNRtest3 = spanvar(PchdBm)
#SNRtest4 = spanvar(PchdBm)
# =============================================================================
# plt.plot(PchdBmrs, SNRdataset,'*', label = 'draw one')
# plt.ylabel('SNR (dB)')
# plt.xlabel('Pch (dBm)')
# plt.legend()
# plt.title('SNR vs Pch for NDFIS loss')
# #plt.grid()
# #plt.savefig('SNRvspch10spans.pdf', dpi=200)
# plt.show() 
# =============================================================================

#np.savetxt('SNRvspch1span.csv', SNRdataset, delimiter=',') 
#np.savetxt('PchdBmrs.csv', PchdBmrs, delimiter=',') 

# NSFNET implementation 
path1 = [3000,3600,2100]
path2 = [2100,1200,3600,2100]
path3 = [4800,1500,1500]
path4 = [4800,1500,2700]

np.random.seed(101)

# path 1 - calculate the SNR at each node and return them 
def NSFNETexample(path):
    p = []
    pts = []
    #pn = []
    #pase = []
    for i in range(np.size(path)):
        p.append(main(Lspans, int(path[i]/Lspans), 157, 101, 201, alpha, Disp, PchdBm, NF, NLco,False)[0] )
        Popt = PchdBm[np.argmax(p[i])]
        Popts = np.linspace(Popt-1.5, Popt+1.5, numpoints) +  np.random.normal(0, powerdiffvar, numpoints)
        pts.append(main(Lspans, int(path[i]/Lspans), 157, 101, 201, alpha, Disp, Popts, NF, NLco,True)[0] )
        #pn.append(main(Lspans, int(path[i]/Lspans), 157, 101, 201, alpha, Disp, PchdBm, NF, NLco,False)[0] +  np.random.normal(0, snrdiffvar, numpoints))
        #popt = (main(Lspans, int(path[i]/Lspans), 157, 101, 201, alpha, Disp, PchdBm, NF, NLco)[-1])
    #path1popt = [PchdBm[np.argmax(p[i])] for i in range(np.size(p,0))]
    return p,pts, Popt

path1snr = NSFNETexample(path1)[1][-1]
Popt = NSFNETexample(path1)[2]
drawind = np.linspace(1,numpoints,numpoints)
PchdBmopts = np.linspace(Popt-1.5, Popt+1.5, numpoints)

plt.plot(PchdBmopts, path1snr, '*')
plt.xlabel("Pch")
plt.ylabel("SNR (dB)")
plt.show()

np.savetxt('drawind.csv', drawind, delimiter=',') 
np.savetxt('PchdBmopts.csv', PchdBmopts, delimiter=',') 
np.savetxt('SNRpath1.csv', path1snr, delimiter=',') 

# %%

# BER calculation 
M = 4
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


# check peak BER for each node - must be below FEC threshold 
def BERcheck(pathsnr):
    snrmax = [np.amax(pathsnr[i]) for i in range(np.size(pathsnr,0))]
    pathber = []
    FECthreshold = 2e-2
    for i in range(np.size(snrmax)):
        pathber.append(BERcalc(M, snrmax[i]))
    errorlinks = [pathber.index(ber) for ber in pathber if ber > FECthreshold]
    if len(errorlinks) > 0:
        print("FECthreshold exceeded" )
    return pathber, errorlinks

def BERfullcalc(pathsnr):
    ber = np.zeros((np.size(pathsnr,0),np.size(pathsnr,1)))
    for i in range(np.size(pathsnr,0)):
        ber[i] = [BERcalc(M, db2lin(pathsnr[i][j])) for j in range(np.size(pathsnr,1))]
    return ber
    
#path1bern = BERfullcalc(path1snrn)
#path2bern = BERfullcalc(path2snrn)
#path3bern = BERfullcalc(path3snrn)
#path4bern = BERfullcalc(path4snrn)

#plt.plot(PchdBm, path1ber[-1],'*', label= 'path 1')
# =============================================================================
# plt.plot(PchdBm, path1snr[2], label= 'path 1 span 3')
# plt.plot(PchdBm, path1snr[0], label= 'path 1 span 1')
# plt.plot(PchdBm, path1snr[1], label= 'path 1 span 2')
# plt.xlabel('Pch (dBm)')
# plt.ylabel('SNR (dB)')
# plt.legend()
# plt.show()
# =============================================================================

#plt.plot(PchdBm, path1snr[-1], label= 'path 1')
# =============================================================================
# plt.plot(PchdBm, path1snrn[-1], label= 'path 1 noise')
# plt.xlabel('Pch (dBm)')
# plt.ylabel('SNR (dB)')
# plt.legend()
# plt.show()
# =============================================================================

#np.savetxt('PchdBm.csv', PchdBm, delimiter=',') 
#np.savetxt('SNRpath1.csv', path1snrn[-1], delimiter=',') 
#np.savetxt('BERpath1.csv', path1bern[-1], delimiter=',') 

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

start_time = time.time()
IxyNy = findMI(SNRanalyticalbase)
IxyRS = findMI(SNRanalyticalRSbase)
IxyRS2 = findMI(SNRanalyticalRS2base)

#IxyNy = [MIGHquad(i) for i in SNRanalyticalbase ]
#IxyRS = [MIGHquad(i) for i in SNRanalyticalRSbase ]
#IxyRS2 = [MIGHquad(i) for i in SNRanalyticalRS2base ]

duration = time.time() - start_time
print("MI calculation duration: " + str(duration))

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
    
    
    
    # %% =============================================================================
    # find the reach from BER
    
    # Ny = 0 for Nyquist, 1 for RS and 2 for RS2
    def reachcalc(Ny, P):
        FECthreshold = 2e-2
        BER = np.zeros(numpoints)
        Ns = 2 # start at 2 spans because of the denominator of (22) in Poggiolini's GN model paper - divide by ln(Ns) = 0 for Ns = 1
           
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
    
    reachNy = np.zeros(numpoints)
    reachRS = np.zeros(numpoints)
    reachRS2 = np.zeros(numpoints)
    for i in range(np.size(PchreachdBm)): 
        reachNy[i] = reachcalc(0, PchreachdBm[i])
        reachRS[i] = reachcalc(1, PchreachdBm[i])
        reachRS2[i] = reachcalc(2, PchreachdBm[i])
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


# %% plot of reach vs number of spans 

# =============================================================================
# plt.plot(PchreachdBm, reachNy, label = 'Nyquist' )
# plt.plot(PchreachdBm, reachRS, label = '50 GHz non-nyquist' )
# plt.plot(PchreachdBm, reachRS2, label = '25 GHz non-nyquist' )
# plt.legend()
# plt.ylabel('reach (100 km spans) ')
# plt.xlabel('Pch (dBm)')
# plt.title('Reach vs Pch 64 QAM ')
# plt.grid()
# plt.savefig('reachvspch64qam.png', dpi=200)
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
 


















    
