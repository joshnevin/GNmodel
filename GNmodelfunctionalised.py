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

numpoints = 50

def main(Ls, Ns, NchNy, NchRS, NchRS2, al, D, PchdBm, NF, gam ):
    
    
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
  
        # H1 = 1  
        harmonic = 1.00
  
        # loop to apply the forumula  
        # Hn = H1 + H2 + H3 ... +  
        # Hn-1 + Hn-1 + 1/n  
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
    #Pasech = NFl*h*f*(Gl - 1)*OSNRmeasBW*Ns # [W] the ASE noise power in the OSNR measurement BW, defined as Dlam = 0.1nm
    PasechRS = NFl*h*f*(Gl - 1)*BchRS*1e9*Ns # [W] the ASE noise power in one non-Nyquist channel across all spans
    PasechRS2 = NFl*h*f*(Gl - 1)*BchRS2*1e9*Ns
    # SNR calc + plotting 
    
    
    SNRanalytical = np.zeros(numpoints)
    SNRanalyticalRS = np.zeros(numpoints)
    SNRanalyticalRS2 = np.zeros(numpoints)
    OSNRanalytical = np.zeros(numpoints)
    
    for i in range(numpoints):
        
        if np.size(Pasech) > 1 and np.size(GnliEq13mul) > 1 and np.size(Pch)==1:
            SNRanalytical[i] = 10*np.log10((LF*Pch)/(Pasech[i] + GnliEq13mul[i]*Rs*1e9))
            #SNRanalytical[i] = 10*np.log10((LF*Pch)/(Pasech[i] + GnliEq13mul[i]*OSNRmeasBW))
            OSNRanalytical[i] = 10*np.log10((LF*Pch)/Pasech[i])
            SNRanalyticalRS[i] = 10*np.log10((LF*Pch)/(PasechRS[i] + GnliEq15mul[i]*BchRS*1e9))
            SNRanalyticalRS2[i] = 10*np.log10((LF*Pch)/(PasechRS2[i] + GnliEq15mul2[i]*BchRS2*1e9))
            
        elif np.size(Pasech) > 1 and np.size(GnliEq13mul) == 1 and np.size(Pch)==1:    
            
            SNRanalytical[i] = 10*np.log10((LF*Pch)/(Pasech[i] + GnliEq13mul*Rs*1e9))
            #SNRanalytical[i] = 10*np.log10((LF*Pch)/(Pasech[i] + GnliEq13mul*OSNRmeasBW))
            SNRanalyticalRS2[i] = 10*np.log10((LF*Pch)/(PasechRS2[i] + GnliEq15mul2*BchRS2*1e9))
            SNRanalyticalRS[i] = 10*np.log10((LF*Pch)/(PasechRS[i] + GnliEq15mul*BchRS*1e9))
            OSNRanalytical[i] = 10*np.log10((LF*Pch)/Pasech[i])

        #SNRMC[i] = 10*np.log10((LF*Pch)/(Pasech[i] + GnliMC*Rs*1e9))
        #SNRMC[i] = 10*np.log10((LF*Pch)/(Pasech[i] + GnliMC[i]*Rs*1e9))
        
        elif np.size(Pasech) == 1 and np.size(GnliEq13mul) > 1 and np.size(Pch)==1:    
        
            SNRanalytical[i] = 10*np.log10((LF*Pch)/(Pasech + GnliEq13mul[i]*Rs*1e9))
            #SNRanalytical[i] = 10*np.log10((LF*Pch)/(Pasech + GnliEq13mul[i]*OSNRmeasBW))
            SNRanalyticalRS[i] = 10*np.log10((LF*Pch)/(PasechRS + GnliEq15mul[i]*BchRS*1e9))
            SNRanalyticalRS2[i] = 10*np.log10((LF*Pch)/(PasechRS2 + GnliEq15mul2[i]*BchRS2*1e9))
            OSNRanalytical[i] = 10*np.log10((LF*Pch)/Pasech)       
        
        elif np.size(Pasech) == 1 and np.size(GnliEq13mul) > 1 and np.size(Pch) > 1:    
        
            SNRanalytical[i] = 10*np.log10((LF*Pch[i])/(Pasech + GnliEq13mul[i]*Rs*1e9))
            #SNRanalytical[i] = 10*np.log10((LF*Pch[i])/(Pasech + GnliEq13mul[i]*OSNRmeasBW))
            SNRanalyticalRS[i] = 10*np.log10((LF*Pch[i])/(PasechRS + GnliEq15mul[i]*BchRS*1e9))
            SNRanalyticalRS2[i] = 10*np.log10((LF*Pch[i])/(PasechRS2 + GnliEq15mul2[i]*BchRS2*1e9))
            OSNRanalytical[i] = 10*np.log10((LF*Pch[i])/Pasech)          
        elif np.size(Pasech) == 1 and np.size(GnliEq13mul) == 1 and np.size(Pch) == 1:    
        
            SNRanalytical[i] = 10*np.log10((LF*Pch)/(Pasech + GnliEq13mul*Rs*1e9))
            #SNRanalytical[i] = 10*np.log10((LF*Pch)/(Pasech + GnliEq13mul*OSNRmeasBW))
            SNRanalyticalRS[i] = 10*np.log10((LF*Pch)/(PasechRS + GnliEq15mul*BchRS*1e9))
            SNRanalyticalRS2[i] = 10*np.log10((LF*Pch)/(PasechRS2 + GnliEq15mul2*BchRS2*1e9))   
            OSNRanalytical[i] = 10*np.log10((LF*Pch)/Pasech)
        else:
            #SNRanalyticalRS[i] = 10*np.log10((LF*Pch[i])/(PasechRS[i] + GnliEq15mul[i]*BchRS*1e9))
            SNRanalytical[i] = 10*np.log10((LF*Pch[i])/(Pasech[i] + GnliEq13mul[i]*Rs*1e9))
            SNRanalytical[i] = 10*np.log10((LF*Pch[i])/(Pasech[i] + GnliEq13mul[i]*OSNRmeasBW))
            SNRanalyticalRS2[i] = 10*np.log10((LF*Pch[i])/(PasechRS2[i] + GnliEq15mul2[i]*BchRS2*1e9))
            OSNRanalytical[i] = 10*np.log10((LF*Pch[i])/Pasech[i])
    return SNRanalytical, SNRanalyticalRS, SNRanalyticalRS2, OSNRanalytical, GnliEq13, epsilon


# %% ================================ NDFIS data stuff  ===========================================

NDFISdata = np.genfromtxt(open("NDFISdata.csv", "r"), delimiter=",", dtype =float)
NDFISdata = np.delete(NDFISdata,0,0 )
# extract loss and overall loss 
NDFISloss = NDFISdata.T[2]
NDFISlossoverall = NDFISdata.T[4]
NDFISdisp = NDFISdata.T[6]
# remove links for which loss or overall loss is missing 
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

# %% ======================== NDFIS loss fitting stuff ======================== 
# =============================================================================
# plt.plot(NDFISlossnz ,'*' , label = 'fibre' )
# plt.plot(NDFISlossoverallnz , '+', label = 'overall' )
# plt.legend()
# plt.ylabel('losses (dB/km) ')
# #plt.xlabel(' ')
# plt.title('NDFIS losses')
# plt.grid()
# #plt.savefig('reachvspch64qam.png', dpi=200)
# plt.show()
# =============================================================================

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

# =============================================================================
#plt.hist(bins[:-1], bins, weights=counts)
#plt.xlabel("loss (dB/km)")
#plt.ylabel("freqeuncy")
#plt.title("fibre loss")
#plt.savefig('NDFISlosshist20.png', dpi=200)
#plt.show()
# =============================================================================

# =============================================================================
#plt.hist(binso[:-1], binso, weights=countso)
#plt.xlabel("loss (dB/km)")
#plt.ylabel("freqeuncy")
#plt.title("overall loss")
#plt.savefig('NDFISoveralllosshist20.png', dpi=200)
#plt.show()
# =============================================================================

#plt.hist(NDFISlossnz,bins = 15)
#plt.show()
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

Nbins = np.zeros(np.size(bins))
Nbins[0] = bins[0] - (bins[1] - bins[0]) 
for i in range(0, (np.size(bins)-1),1 ):
    Nbins[i+1] = (bins[i+1] + bins[i])/2
Ncounts = [0] + [i for i in counts]
# =============================================================================
# polfit = np.polyfit(Nbins,Ncounts,4)
# p = np.poly1d(polfit)
# xp = np.linspace(0.186, 0.272, 100)
# plt.plot(Nbins, Ncounts, '*')
# plt.plot(xp, p(xp))
# plt.grid()
# plt.xlabel("loss (dB/km)")
# plt.ylabel("freqeuncy")
# plt.title("order 4")
# plt.savefig('NDFISlossfreqdenpol6FD.png', dpi=200)
# plt.show()
# =============================================================================
Nbinso = np.zeros(np.size(binso))
Nbinso[0] = binso[0] - (binso[1] - binso[0]) 
for i in range(0, (np.size(binso)-1),1 ):
    Nbinso[i+1] = (binso[i+1] + binso[i])/2
Ncountso = [0] + [i for i in countso]
# =============================================================================
# polfito = np.polyfit(Nbinso,Ncountso,12)
# po = np.poly1d(polfito)
# xpo = np.linspace(0.148, 0.429, 100)
# plt.plot(Nbinso, Ncountso, '*')
# plt.plot(xpo, po(xp))
# plt.grid()
# plt.xlabel("loss (dB/km)")
# plt.ylabel("freqeuncy")
# plt.title("order 12")
# plt.savefig('NDFISlossfreqdenpol12overallFD.png', dpi=200)
# plt.show()
# =============================================================================

# %% fit a GPR model to the loss-frequency data 

# =============================================================================
# kernel = 1.0**2 * RBF(length_scale=0.1)
# gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.5, normalize_y=True, n_restarts_optimizer=10)
# gpr.fit(Nbins.reshape(-1,1), Ncounts)
# yGP = gpr.predict(xp.reshape(-1,1))
# 
# plt.plot(xp,yGP)
# plt.plot(Nbins, Ncounts, '*')
# plt.grid()
# plt.xlabel("loss (dB/km)")
# plt.ylabel("freqeuncy")
# plt.title("noise = 0.5")
# plt.savefig('GPRfitnoise05FD.png', dpi=200)
# plt.show() 
#  
# kernelo = 1.0**2 * RBF(length_scale=0.1)
# gpro = GaussianProcessRegressor(kernel=kernelo, alpha=0.5, normalize_y=True, n_restarts_optimizer=10)
# gpro.fit(Nbinso.reshape(-1,1), Ncountso)
# yGPo = gpro.predict(xpo.reshape(-1,1))
# 
# plt.plot(xpo,yGPo)
# plt.plot(Nbinso, Ncountso, '*')
# plt.grid()
# plt.xlabel("loss (dB/km)")
# plt.ylabel("freqeuncy")
# plt.title("noise = 0.5")
# plt.savefig('GPRfitnoise05FDoverall.png', dpi=200)
# plt.show() 
# =============================================================================

# %% random draws from histogram 

countsnorm = np.zeros(np.size(counts))
for i in range(np.size(countsnorm)):
    countsnorm[i] = counts[i]/np.sum(counts)

countsnormo = np.zeros(np.size(countso))
for i in range(np.size(countsnormo)):
    countsnormo[i] = countso[i]/np.sum(countso)

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
#varperc = lossvarperc  # percentage variance from mean of Gaussian 
#alphamean = NDFISlossnzmean  # Gaussian mean 
#alpha = np.random.normal(alphamean, alphamean*varperc*0.01, numpoints)
#alpha = np.random.normal(NDFISlossoverallnzmean, NDFISlossoverallnzvar**0.5, numpoints)
#alpha = NDFISlossnzmean
alpha = NDFISlossnzmean
#alpha = np.zeros(numpoints)
#for i in range(numpoints):
#    alpha[i] = histdraw(countsnorm, bins)
#alpha = NDFISlossnzmean
#Disp = np.random.normal(NDFISdispnzmean, NDFISdispnzvar**0.5, numpoints)
NLco = 1.27
Nspans = 1
Lspans = 100
NchRS = 101
Disp = NDFISdispnzmean
#Disp = 16.7
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
#NF = 4.5


# =========================== differing spans bit ==================================
def spanvar(PchdBm):
    Gnlispanvar = np.zeros(Nspans)
    Pasespanvar = np.zeros(Nspans)
    ep = np.zeros(Nspans)
    lam = 1550
    Rs = 32
    f = 299792458/(lam*1e-9) # operating frequency [Hz]
    h = 6.63*1e-34  # Planck's constant [Js] 
    for i in range(Nspans):
            
        alpha2 = histdraw(countsnorm, bins)
        gain_target = alpha2*Lspans
        g1a = gain_target - deltap - (gain_max - gain_target)
        NF = lin2db(db2lin(nf1) + db2lin(nf2)/db2lin(g1a)) 
        NFl = 10**(NF/10) 
        Gl = 10**(gain_target/10)
        Gnlispanvar[i] = main(Lspans, Nspans, 157, 101, 201, alpha2, Disp, PchdBm, NF, NLco)[4]
        ep[i] = main(Lspans, Nspans, 157, 101, 201, alpha2, Disp, PchdBm, NF, NLco)[5]
        # ASE noise bit 
        Pasespanvar[i] = NFl*h*f*(Gl - 1)*Rs*1e9
        
    Pasech = np.sum(Pasespanvar)
    Gnli = np.sum(Gnlispanvar)*(Nspans**(np.mean(ep)))
    Pch = 1e-3*10**(PchdBm/10)  # ^ [W]
    
    return 10*np.log10((Pch)/(Pasech + Gnli*Rs*1e9))

numpointsdg = 100
#PchdBmdata = np.random.uniform(-10,10, numpointsdg)
PchdBmdata = np.linspace(-10,10,numpointsdg)
start_time = time.time()
def datagenpar(Pch):
    with multiprocessing.Pool() as pool:
        SNR = pool.map(spanvar, Pch) 
    return SNR
numsweeps = 10
SNRdataset = []
for _ in range(numsweeps):
    SNRdataset.append(datagenpar(PchdBmdata))

SNRdatasetrs = np.reshape(SNRdataset,numpointsdg*numsweeps)
PchdBmrs = []
for _ in range(numsweeps):
    PchdBmrs.append(PchdBmdata)
PchdBmrs = np.reshape(PchdBmrs,numpointsdg*numsweeps)

duration = time.time() - start_time
print(duration)
 
#%% =============================================================================
#plt.plot(PchdBmdata,SNRdataset[0],'*', label= 'sweep 1')
#plt.plot(PchdBmdata,SNRdataset[1],'o', label= 'sweep 2')
#plt.plot(PchdBmdata,SNRdataset[2],'^', label= 'sweep 3')
#plt.plot(PchdBmdata,SNRdataset[3],'.', label= 'sweep 4')
#plt.plot(PchdBmdata,SNRdataset[4],'>', label= 'sweep 5')
plt.plot(PchdBmrs,SNRdatasetrs,'*', label= 'SNR Dataset')
plt.ylabel('SNR (dB)')
plt.xlabel('Pch (dBm)')
plt.legend()
plt.title('SNR vs Pch for NDFIS loss')
plt.grid()
plt.savefig('SNRvspch10sweeps.pdf', dpi=200)
plt.show()
# =============================================================================

plt.plot(PchdBmdata,SNRdataset[0],'*', label= 'SNR Dataset')
plt.ylabel('SNR (dB)')
plt.xlabel('Pch (dBm)')
plt.legend()
plt.title('SNR vs Pch for NDFIS loss')
plt.grid()
plt.savefig('SNRvspchsweep0.pdf', dpi=200)
plt.show()


#%% ====================================================================================

GNmodelresbase = main(Lspans, Nspans, 157, 101, 201, alpha, Disp, PchdBm, NF, NLco)
#GNmodelresbase = main(100, 10, 157, 101, 201, 0.2, 16.7, 0, 6.0, 1.27)
#GNmodelresp = main(Lspans, Nspans, 157, 101, 201, alpha, Disp, PchdBm, NF, NLco)
#GNmodelresn = main(Lspans, Nspans, 157, 101, 201, alpha, Disp, PchdBm, NF, NLco)

SNRanalyticalbase = GNmodelresbase[0]
SNRanalyticalRSbase = GNmodelresbase[1]
SNRanalyticalRS2base = GNmodelresbase[2]
OSNRanalytical = GNmodelresbase[3]

#SNRanalyticalp = GNmodelresp[0]
#SNRanalyticalRSp = GNmodelresp[1]
#SNRanalyticalRS2p = GNmodelresp[2]
#SNRanalyticaln = GNmodelresn[0]
#SNRanalyticalRSn = GNmodelresn[1]
#SNRanalyticalRS2n = GNmodelresn[2]

numdraws = np.linspace(0,numpoints-1, numpoints)





# %% ================================ Mutual information estimation ===========================================
  
# import constellation shapes from MATLAB-generated csv files 
    
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
#Ixy = [MIGHquad(M,L,i) for i in SNR ]
duration = time.time() - start_time
print(duration)

# %% ================================== Reach calculation ==================================

# find the BER from: On the Bit Error Probability of QAM Modulation - Michael P. Fitz 

PchreachdBm = np.linspace(-5,5,numpoints)
lossreach = NDFISlossoverallnzmean
dispreach = NDFISdispnzmean
# =============================================================================
SNRNYr = main(Lspans, Nspans, 157, 101, 201, lossreach,dispreach, PchreachdBm, NF, NLco)[0]
SNRRSr = main(Lspans, Nspans, 157, 101, 201, lossreach, dispreach, PchreachdBm, NF, NLco)[1]
SNRRS2r = main(Lspans, Nspans, 157, 101, 201, lossreach, dispreach, PchreachdBm, NF, NLco)[2]

# BER calculation 
M = 16
if M == 4: 
    BERny = 0.5*special.erfc(SNRNYr**0.5)
    BERrs = 0.5*special.erfc(SNRRSr**0.5)
    BERrs2 = 0.5*special.erfc(SNRRS2r**0.5)
     
elif M == 16:
    BERny = (3/8)*special.erfc(((2/5)*SNRNYr)**0.5) + (1/4)*special.erfc(((18/5)*SNRNYr)**0.5) - (1/8)*special.erfc((10*SNRNYr)**0.5)
    BERrs = (3/8)*special.erfc(((2/5)*SNRRSr)**0.5) + (1/4)*special.erfc(((18/5)*SNRRSr)**0.5) - (1/8)*special.erfc((10*SNRRSr)**0.5)
    BERrs2 = (3/8)*special.erfc(((2/5)*SNRRS2r)**0.5) + (1/4)*special.erfc(((18/5)*SNRRS2r)**0.5) - (1/8)*special.erfc((10*SNRRS2r)**0.5)
     
elif M == 64:
    BERny = (7/24)*special.erfc(((1/7)*SNRNYr)**0.5) + (1/4)*special.erfc(((9/7)*SNRNYr)**0.5) - (1/24)*special.erfc(((25/7)*SNRNYr)**0.5) - (1/24)*special.erfc(((25/7)*SNRNYr)**0.5) + (1/24)*special.erfc(((81/7)*SNRNYr)**0.5) - (1/24)*special.erfc(((169/7)*SNRNYr)**0.5) 
    BERrs = (7/24)*special.erfc(((1/7)*SNRRSr)**0.5) + (1/4)*special.erfc(((9/7)*SNRRSr)**0.5) - (1/24)*special.erfc(((25/7)*SNRRSr)**0.5) - (1/24)*special.erfc(((25/7)*SNRRSr)**0.5) + (1/24)*special.erfc(((81/7)*SNRRSr)**0.5) - (1/24)*special.erfc(((169/7)*SNRRSr)**0.5) 
    BERrs2 = (7/24)*special.erfc(((1/7)*SNRRS2r)**0.5) + (1/4)*special.erfc(((9/7)*SNRRS2r)**0.5) - (1/24)*special.erfc(((25/7)*SNRRS2r)**0.5) - (1/24)*special.erfc(((25/7)*SNRRS2r)**0.5) + (1/24)*special.erfc(((81/7)*SNRRS2r)**0.5) - (1/24)*special.erfc(((169/7)*SNRRS2r)**0.5) 
# %% =============================================================================
# find the reach from BER

# Ny = 0 for Nyquist, 1 for RS and 2 for RS2
def reachcalc(Ny, P):
    FECthreshold = 2e-2
    BER = np.zeros(numpoints)
    Ns = 2 # start at 2 spans because of the denominator of (22) in Poggiolini's GN model paper - divide by ln(Ns) = 0 for Ns = 1
       
    while BER[0] < FECthreshold:
            
        SNR = main(Lspans, Ns, 157, 101, 201, NDFISlossoverallnzmean, NDFISdispnzmean, P, NF, NLco)[Ny]
            
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

GNPysnr = np.genfromtxt(open("SNRGNPyvargainsysmar0ch2.csv", "r"), delimiter=",", dtype =float)
GNPyosnr = np.genfromtxt(open("OSNRGNPyvargainsysmar0ch2.csv", "r"), delimiter=",", dtype =float)


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
# 
# =============================================================================
# plt.plot(PchdBm, SNRanalyticalbase, label = 'Nyquist')
# #plt.plot(PchdBm,GNPysnr, label='GNPy')
# plt.legend()
# plt.ylabel('SNR (dB)')
# #plt.xlabel('loss coefficient (dB/km)')
# #plt.xlabel('EDFA NF (dB)')
# plt.xlabel('Pch (dBm)')
# #plt.xlabel('Draw index')
# # #plt.xlabel('dispersion (ps/nm*km)')
# plt.title('SNR vs Pch 0.1 nm variable gain 2 channels')
# plt.grid()
# #plt.savefig('JoshvsGNPyvariablegain2ch.png', dpi=200)
# plt.show()
# 
# plt.plot(PchdBm, OSNRanalytical, label = 'Josh')
# #plt.plot(PchdBm,GNPyosnr, label='GNPy')
# plt.legend()
# plt.ylabel('OSNR (dB)')
# #plt.xlabel('loss coefficient (dB/km)')
# #plt.xlabel('EDFA NF (dB)')
# plt.xlabel('Pch (dBm)')
# #plt.xlabel('Draw index')
# # #plt.xlabel('dispersion (ps/nm*km)')
# plt.title('OSNR vs Pch 0.1 nm variable gain 2 channels')
# plt.grid()
# #plt.savefig('JoshvsGNPyvariablegainOSNR2ch.png', dpi=200)
# plt.show()
# 
# =============================================================================


# 
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
 


















    
