################### imports ####################
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
from GNf import GNmain
import random
from dijkstra import dijkstra
import matplotlib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from sklearn.preprocessing import StandardScaler
import cProfile
from scipy.special import erfc


def BERcalc(M, SNR): # linear SNR here is energy per symbol - need to convert to energy per bit to use these formulae - hence the division by log2(M)
        if M == 2: 
            BER = 0.5*special.erfc(((2*SNR)/np.log2(2))**0.5)
        elif M == 4: 
            BER = 0.5*special.erfc(((2*SNR)/np.log2(4))**0.5)    
        elif M == 16:
            BER = (3/8)*special.erfc(((2/5)*(SNR/np.log2(16)))**0.5) + (1/4)*special.erfc(((18/5)*(SNR/np.log2(16)))**0.5) - (1/8)*special.erfc((10*(SNR/np.log2(16)))**0.5)
        elif M == 64:
            BER = (7/24)*special.erfc(((1/7)*(SNR/np.log2(64)))**0.5) + (1/4)*special.erfc(((9/7)*(SNR/np.log2(64)))**0.5) - (1/24)*special.erfc(((25/7)*(SNR/np.log2(64)))**0.5) + (1/24)*special.erfc(((81/7)*(SNR/np.log2(64)))**0.5) - (1/24)*special.erfc(((169/7)*(SNR/np.log2(64)))**0.5) 
        else:
            print("unrecognised modulation format")
        return BER
    

PchdBm = np.linspace(-6,6,500)  # 500 datapoints for higher resolution of Pch
TRxb2b = 26 # fron UCL paper: On the limits of digital back-propagation in the presence of transceiver noise, Lidia Galdino et al.
numpoints = 100
LspansA = 100
alpha = 0.2
NLco = 1.27
NchRS = 101
Disp = 16.7
linklen = np.linspace(100, 5000, 50)
numlens = len(linklen)

def marginsnrtest(edgelen, Lspans, numlam, NF, alpha):
    lam = 1550 # operating wavelength centre [nm]
    f = 299792458/(lam*1e-9) # operating frequency [Hz]
    c = 299792.458 # speed of light in vacuum [nm/ps] -> needed for calculation of beta2
    Rs = 32 # symbol rate [GBaud]
    h = 6.63*1e-34  # Planck's constant [Js]
    NchNy = numlam
    BWNy = (NchNy*Rs)/1e3 # keep Nyquist spacing for non-saturated channel
    #BWNy = (157*Rs)/1e3 # full 5THz BW
    Numspans = int(edgelen/Lspans)
    al = alpha
    D = Disp
    gam = NLco
    Ls = Lspans
    allin = np.log((10**(al/10)))/2 # fibre loss [1/km] -> weird definition, due to exponential decay of electric field instead of power, which is standard 
    #gam = 1.27 # fibre nonlinearity coefficient [1/W*km]
    beta2 = (D*(lam**2))/(2*np.pi*c) # dispersion coefficient at given wavelength [ps^2/km]
    Leff = (1 - np.exp(-2*allin*Ls ))/(2*allin)  # effective length [km]      
    Leffa = 1/(2*allin)  # the asymptotic effective length [km]
    
    numpch = len(PchdBm) # account for higher resolution of PchdBm - not equal to numpoints
    Pchsw = 1e-3*10**(PchdBm/10)  # [W]
    Gwdmsw = (Pchsw*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
    GnliEq13sw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))
    G = al*Ls
    NFl = 10**(NF/10) 
    Gl = 10**(G/10) 
    Pasesw = NFl*h*f*(Gl - 1)*Rs*1e9 # [W] the ASE noise power in one Nyquist channel across all spans
    snrsw = (Pchsw)/(Pasesw*np.ones(numpch) + GnliEq13sw*Rs*1e9)
    Popt = PchdBm[np.argmax(snrsw)]                                                   
    
    Gwdm = (1e-3*10**(Popt/10)*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
    Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))*Numspans
    Pase = NF*h*f*(db2lin(alpha*Ls) - 1)*Rs*1e9*Numspans
    Pch = 1e-3*10**(Popt/10) 
    snr = (  (Pch/(Pase + Gnli*Rs*1e9))**(-1) + db2lin(26)**(-1)  )**(-1)
    return lin2db(snr)
def fmdatagentest(edgelens,Lspans, numlam, NF, alpha):
    marginSNR = np.empty([np.size(edgelens),1])
    for i in range(np.size(edgelens)):
        marginSNR[i] = marginsnrtest(edgelens[i],Lspans, numlam, NF, alpha)
    return marginSNR

testlen = 1000.0     # all ageing effects modelled using values in: Faster return of investment in WDM networks when elastic transponders dynamically fit ageing of link margins, Pesic et al.
years = np.linspace(0,10,21) # define how many years are in the lifetime of the network and the resolution 
numyears = np.size(years)
sd = np.linspace(0.04, 0.08, np.size(years)) # added SNR uncertainty SD - assumed to double over lifetime
numlam = np.linspace(30, 150, np.size(years)) # define the channel loading over the network lifetime 
NF = np.linspace(4.5,5.5,np.size(years)) # define the NF ageing of the amplifiers 
alpha = 0.2 + 0.00163669*years # define the fibre ageing due to splice losses over time 
trxaging = ((1 + 0.05*years)*2).reshape(np.size(years),1) # define TRx ageing 
oxcaging = ((0.03 + 0.007*years)*2).reshape(np.size(years),1) # define filter ageing, assuming two filters per link, one at Tx and one at Rx
# =============================================================================

# find the worst-case margin required                         
fmD = sd[-1]*5 # D margin is defined as 5xEoL SNR uncertainty SD that is added
fmDGI = sd[0]*5 
fmT = trxaging[-1] + oxcaging[-1] + fmD # total static margin: amplifier ageing, NL loading and fibre ageing included in GN
fmTGI = trxaging[0] + oxcaging[0] + fmDGI # total static margin: use the BoL values for GP initial planning case



def GPtrain(x,y):
    y = y.reshape(-1,1)
    x = x.reshape(-1,1)
    scaler = StandardScaler().fit(y)
    y = scaler.transform(y)
    #kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e3)) + W(1.0, (1e-5, 1e5))
    kernel = C(0.03, (1e-3, 1e1)) * RBF(0.01, (1e-2, 1e2)) 
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 20, normalize_y=False, alpha=np.var(y))
    gpr.fit(x, y)
    #print("Optimised kernel: %s" % gpr.kernel_)
    ystar, sigma = gpr.predict(x, return_std=True )
    sigma = np.reshape(sigma,(np.size(sigma), 1)) 
    sigma = (sigma**2 + 1)**0.5  
    ystarp = ystar + sigma
    ystari = scaler.inverse_transform(ystar)
    ystarpi = scaler.inverse_transform(ystarp)
    sigmai = np.mean(ystarpi - ystari)
    return ystari, sigmai


def fmsnr(edgelen, Lspans, numlam, NF, alpha, yearind):
        Ls = Lspans
        NchNy = numlam
        D = Disp
        gam = NLco
        lam = 1550 # operating wavelength centre [nm]
        f = 299792458/(lam*1e-9) # operating frequency [Hz]
        c = 299792.458 # speed of light in vacuum [nm/ps] -> needed for calculation of beta2
        Rs = 32 # symbol rate [GBaud]
        h = 6.63*1e-34  # Planck's constant [Js]
        BWNy = (NchNy*Rs)/1e3 
        #BWNy = (157*Rs)/1e3 # full 5THz BW
        allin = np.log((10**(alpha/10)))/2 # fibre loss [1/km] -> weird definition, due to exponential decay of electric field instead of power, which is standard 
        beta2 = (D*(lam**2))/(2*np.pi*c) # dispersion coefficient at given wavelength [ps^2/km]
        Leff = (1 - np.exp(-2*allin*Ls ))/(2*allin)  # effective length [km]      
        Leffa = 1/(2*allin)  # the asymptotic effective length [km]  
        numspans = int(edgelen/Lspans)
        #Pun = GNmain(Lspans, 1, numlam, 101, 201, alpha, Disp, PchdBm, NF, NLco,False,numpoints)[0] 
        #Popt = PchdBm[np.argmax(Pun)]        
        # ======================= find Popt ===============================
        numpch = len(PchdBm)
        Pchsw = 1e-3*10**(PchdBm/10)  # ^ [W]
        Gwdmsw = (Pchsw*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
        GnliEq13sw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))
        G = alpha*Ls
        NFl = 10**(NF/10) 
        Gl = 10**(G/10) 
        Pasesw = NFl*h*f*(Gl - 1)*Rs*1e9 # [W] the ASE noise power in one Nyquist channel across all spans
        snrsw = (Pchsw)/(Pasesw*np.ones(numpch) + GnliEq13sw*Rs*1e9)
        Popt = PchdBm[np.argmax(snrsw)]  
        # ======================= find Popt ===============================
        Gwdm = (1e-3*10**(Popt/10)*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
        Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))*numspans
        Pase = NF*h*f*(db2lin(alpha*Lspans) - 1)*Rs*1e9*numspans
        Pch = 1e-3*10**(Popt/10) 
        #snr = (Pch/(Pase + Gnli*Rs*1e9)) - db2lin(trxaging[yearind] + oxcaging[yearind])
        snr = (Pch/(Pase + Gnli*Rs*1e9))
        snr = ( snr**(-1) + (db2lin(TRxb2b))**(-1) )**(-1)
        return lin2db(snr) 


def SNRnew(edgelen,numlam, yearind):  # function for generating a new SNR value to test if uncertainty is dealt with
        Ls = LspansA
        NchNy = numlam    # numlam is an input here because the number of wavelengths actually established on a given link is used 
        D = Disp          # rather than the worst-case number 
        gam = NLco
        lam = 1550 # operating wavelength centre [nm]
        f = 299792458/(lam*1e-9) # operating frequency [Hz]
        c = 299792.458 # speed of light in vacuum [nm/ps] -> needed for calculation of beta2
        Rs = 32 # symbol rate [GBaud]
        h = 6.63*1e-34  # Planck's constant [Js]
        BWNy = (NchNy*Rs)/1e3 
        #BWNy = (157*Rs)/1e3 # full 5THz BW
        allin = np.log((10**(alpha[yearind]/10)))/2 # fibre loss [1/km] -> weird definition, due to exponential decay of electric field instead of power, which is standard 
        beta2 = (D*(lam**2))/(2*np.pi*c) # dispersion coefficient at given wavelength [ps^2/km]
        Leff = (1 - np.exp(-2*allin*Ls ))/(2*allin)  # effective length [km]      
        Leffa = 1/(2*allin)  # the asymptotic effective length [km]  
        numspans = int(edgelen/Ls)
        # ===================== find Popt ==========================
        numpch = len(PchdBm)
        Pchsw = 1e-3*10**(PchdBm/10)  # ^ [W]
        Gwdmsw = (Pchsw*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
        GnliEq13sw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))
        G = alpha[yearind]*Ls
        NFl = 10**(NF[yearind]/10) 
        Gl = 10**(G/10) 
        Pasesw = NFl*h*f*(Gl - 1)*Rs*1e9 # [W] the ASE noise power in one Nyquist channel across all spans
        snrsw = (Pchsw)/(Pasesw*np.ones(numpch) + GnliEq13sw*Rs*1e9)
        Popt = PchdBm[np.argmax(snrsw)]  
        # ===================== find Popt ==========================
        Popt = PchdBm[np.argmax(snrsw)]                                                   
        Gwdm = (1e-3*10**(Popt/10)*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
        Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))*numspans
        Pase = NF[yearind]*h*f*(db2lin(alpha[yearind]*Ls) - 1)*Rs*1e9*numspans
        Pch = 1e-3*10**(Popt/10) 
        snr = (Pch/(Pase + Gnli*Rs*1e9)) - db2lin(trxaging[yearind] + oxcaging[yearind])
        snr = ( snr**(-1) + (db2lin(TRxb2b))**(-1) )**(-1)
        #snr = snr + np.random.normal(0,db2lin(sd),numpoints)
        sdnorm = sd[yearind]
        return lin2db(snr) + np.random.normal(0,sdnorm,1)
    

def SNRgen(edgelen,numlam, yearind):  # function for generating a new SNR value to test if uncertainty is dealt with
        Ls = LspansA
        NchNy = numlam    # numlam is an input here because the number of wavelengths actually established on a given link is used 
        D = Disp          # rather than the worst-case number 
        gam = NLco
        lam = 1550 # operating wavelength centre [nm]
        f = 299792458/(lam*1e-9) # operating frequency [Hz]
        c = 299792.458 # speed of light in vacuum [nm/ps] -> needed for calculation of beta2
        Rs = 32 # symbol rate [GBaud]
        h = 6.63*1e-34  # Planck's constant [Js]
        BWNy = (NchNy*Rs)/1e3 
        #BWNy = (157*Rs)/1e3 # full 5THz BW
        allin = np.log((10**(alpha[yearind]/10)))/2 # fibre loss [1/km] -> weird definition, due to exponential decay of electric field instead of power, which is standard 
        beta2 = (D*(lam**2))/(2*np.pi*c) # dispersion coefficient at given wavelength [ps^2/km]
        Leff = (1 - np.exp(-2*allin*Ls ))/(2*allin)  # effective length [km]      
        Leffa = 1/(2*allin)  # the asymptotic effective length [km]  
        numspans = int(edgelen/Ls)
        # ===================== find Popt ==========================
        numpch = len(PchdBm)
        Pchsw = 1e-3*10**(PchdBm/10)  # ^ [W]
        Gwdmsw = (Pchsw*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
        GnliEq13sw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))
        G = alpha[yearind]*Ls
        NFl = 10**(NF[yearind]/10) 
        Gl = 10**(G/10) 
        Pasesw = NFl*h*f*(Gl - 1)*Rs*1e9 # [W] the ASE noise power in one Nyquist channel across all spans
        snrsw = (Pchsw)/(Pasesw*np.ones(numpch) + GnliEq13sw*Rs*1e9)
        Popt = PchdBm[np.argmax(snrsw)]  
        # ===================== find Popt ==========================
        Popt = PchdBm[np.argmax(snrsw)]                                                   
        Gwdm = (1e-3*10**(Popt/10)*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
        Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))*numspans
        Pase = NF[yearind]*h*f*(db2lin(alpha[yearind]*Ls) - 1)*Rs*1e9*numspans
        Pch = 1e-3*10**(Popt/10) 
        snr = (Pch/(Pase + Gnli*Rs*1e9)) - db2lin(trxaging[yearind] + oxcaging[yearind])
        snr = ( snr**(-1) + (db2lin(TRxb2b))**(-1) )**(-1)
        #snr = snr + np.random.normal(0,db2lin(sd),numpoints)
        sdnorm = sd[yearind]
        return lin2db(snr) + np.random.normal(0,sdnorm,numpoints)

# define the FEC thresholds - all correspond to BER of 2e-2 (2% FEC)
FT2 = 0.24 
FT4 = 3.25 
FT16 = 12.72
FT64 = 18.43
FT128 = 22.35

# %%

def initreach(linklen):
    gnSNR = np.empty([numyears,1])
    for i in range(numyears):
        gnSNR[i] = fmsnr(linklen, LspansA, numlam[i], NF[i], alpha[i], i)
    # fixed margin case
    if gnSNR[-1] - fmT > FT128:
        MF = 128
    elif gnSNR[-1] - fmT > FT64:
        MF = 64
    elif gnSNR[-1] - fmT > FT16:
        MF = 16
    elif gnSNR[-1] - fmT > FT4:
        MF = 4
    elif gnSNR[-1] - fmT > FT2:
        MF = 2
    else:
        print("not able to establish a link")

    # GP case   
    if gnSNR[0] - fmTGI > FT128:
        MFG = 128
    elif gnSNR[0] - fmTGI > FT64:
        MFG = 64
    elif gnSNR[0] - fmTGI > FT16:
        MFG = 16
    elif gnSNR[0] - fmTGI > FT4:
        MFG = 4
    elif gnSNR - fmTGI > FT2:
        MFG = 2
    else:
        print("not able to establish a link")
    return MF, MFG

mf = np.empty([numlens,1])
mfGi = np.empty([numlens,1])
for i in range(numlens):
    mf[i], mfGi[i] = initreach(linklen[i])

# %%

test = SNRgen(500,numlam[0],0)
    
def GPreach(linklen, numsig, yearind):
    
    truesnr = SNRgen(linklen,numlam[yearind],yearind)
    x = np.linspace(0,numpoints-1,numpoints)
    prmn, sigma = GPtrain(x,truesnr)
    prmn = np.mean(prmn)
    if (prmn - FT128)/sigma > numsig:
        FT = FT128
        mfG = 128
    elif (prmn - FT64)/sigma > numsig:
        FT = FT64
        mfG = 64
    elif (prmn - FT16)/sigma > numsig:
        FT = FT16
        mfG = 16
    elif (prmn - FT4)/sigma > numsig:
        FT = FT4
        mfG = 4
    elif (prmn - FT2)/sigma > numsig:
        FT = FT2
        mfG = 2
    else:
        print("not able to establish a link")
    
    if SNRnew(linklen,numlam[yearind],yearind) > FT:
        estbl = 1
        Um = prmn - numsig*sigma - FT
    else:
        estbl = 0
    return mfG, estbl, Um


modf = np.empty([numyears,numlens])
estbl = np.empty([numyears,numlens])
Um = np.empty([numyears,numlens])
for i in range(numyears):
    for j in range(numlens):
        modf[i][j], estbl[i][j], Um[i][j] = GPreach(linklen[j], 5, i)


# %%

modfT = np.transpose(modf)
UmT = np.transpose(Um)
estblT = np.transpose(estbl)

# %%

lenind = 30

plt.plot(years, UmT[lenind])
plt.xlabel("time (years)")
plt.ylabel("U margin (dB)")
plt.show()


mfpl = mf*np.ones(numyears)

plt.plot(years, modfT[lenind])
plt.plot(years, mfpl[lenind])
plt.xlabel("time (years)")
plt.ylabel("QAM order")
plt.show()












