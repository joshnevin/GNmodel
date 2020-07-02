################### imports ####################
import numpy as np
import matplotlib.pyplot as plt
import time 
from scipy import special
#from scipy.stats import iqr
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF
#from GHquad import GHquad
#from NFmodelGNPy import nf_model
from NFmodelGNPy import lin2db
from NFmodelGNPy import db2lin
#from GNf import GNmain
#import random
from dijkstra import dijkstra
import matplotlib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel as W
from sklearn.preprocessing import StandardScaler
import cProfile
from scipy.special import erfc

# %%
nodesN = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14']

graphN = {'1':{'2':2100,'3':3000,'8':4800},'2':{'1':2100,'3':1200,'4':1500},'3':{'1':3000,'2':1200,'6':3600},    
         '4':{'2':1500,'5':1200,'11':3900},'5':{'4':1200,'6':2400,'7':1200}, '6':{'3':3600,'5':2400,'10':2100,'14':3600},
         '7':{'5':1200,'8':1500,'10':2700}, '8':{'1':4800,'7':1500,'9':1500}, '9':{'8':1500,'10':1500,'12':600,'13':600},
         '10':{'6':2100,'7':2700,'9':1500}, '11':{'4':3900,'12':1200,'13':1500}, '12':{'9':600,'11':1200,'14':600},
         '13':{'9':600,'11':1500,'14':300}, '14':{'6':3600,'12':600,'13':300}
         } 
graphnormN = {'1':{'2':2100,'3':3000,'8':4800},'2':{'1':2100,'3':1200,'4':1500},'3':{'1':3000,'2':1200,'6':3600},    
         '4':{'2':1500,'5':1200,'11':3900},'5':{'4':1200,'6':2400,'7':1200}, '6':{'3':3600,'5':2400,'10':2100,'14':3600},
         '7':{'5':1200,'8':1500,'10':2700}, '8':{'1':4800,'7':1500,'9':1500}, '9':{'8':1500,'10':1500,'12':600,'13':600},
         '10':{'6':2100,'7':2700,'9':1500}, '11':{'4':3900,'12':1200,'13':1500}, '12':{'9':600,'11':1200,'14':600},
         '13':{'9':600,'11':1500,'14':300}, '14':{'6':3600,'12':600,'13':300}
         }        
edgesN = {'1':{'2':0,'3':1,'8':2},'2':{'1':3,'3':4,'4':5},'3':{'1':6,'2':7,'6':8},    
         '4':{'2':9,'5':10,'11':11},'5':{'4':12,'6':13,'7':14}, '6':{'3':15,'5':16,'10':17,'14':18},
         '7':{'5':19,'8':20,'10':21}, '8':{'1':22,'7':23,'9':24}, '9':{'8':25,'10':26,'12':27,'13':28},
         '10':{'6':29,'7':30,'9':31}, '11':{'4':32,'12':33,'13':34}, '12':{'9':35,'11':36,'14':37},
         '13':{'9':38,'11':39,'14':40}, '14':{'6':41,'12':42,'13':43}
         }
numedgesN = 44
LspansN = 100

nodesD = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14']

graphD = {'1':{'2':400,'3':160,'4':160},'2':{'1':400,'4':400,'5':240},'3':{'1':160,'4':160,'6':320},    
         '4':{'1':160,'2':400,'3':160,'5':320,'7':240,'10':400},'5':{'2':240,'4':320,'10':480,'11':320}, '6':{'3':320,'7':80,'8':80},
         '7':{'4':240,'6':80,'9':80}, '8':{'6':80,'9':80}, '9':{'7':80,'8':80,'10':240},
         '10':{'4':400,'5':480,'9':240,'11':320,'12':240}, '11':{'5':320,'10':320,'12':240,'14':240}, '12':{'10':240,'11':240,'13':80},
         '13':{'12':80,'14':160}, '14':{'11':240,'13':160}
         } 
graphnormD = {'1':{'2':400,'3':160,'4':160},'2':{'1':400,'4':400,'5':240},'3':{'1':160,'4':160,'6':320},    
         '4':{'1':160,'2':400,'3':160,'5':320,'7':240,'10':400},'5':{'2':240,'4':320,'10':480,'11':320}, '6':{'3':320,'7':80,'8':80},
         '7':{'4':240,'6':80,'9':80}, '8':{'6':80,'9':80}, '9':{'7':80,'8':80,'10':240},
         '10':{'4':400,'5':480,'9':240,'11':320,'12':240}, '11':{'5':320,'10':320,'12':240,'14':240}, '12':{'10':240,'11':240,'13':80},
         '13':{'12':80,'14':160}, '14':{'11':240,'13':160}
         } 
edgesD = {'1':{'2':0,'3':1,'4':2},'2':{'1':3,'4':4,'5':5},'3':{'1':6,'4':7,'6':8},    
         '4':{'1':9,'2':10,'3':11,'5':12,'7':13,'10':14},'5':{'2':15,'4':16,'10':17,'11':18}, '6':{'3':19,'7':20,'8':21},
         '7':{'4':22,'6':23,'9':24}, '8':{'6':25,'9':26}, '9':{'7':27,'8':28,'10':29},
         '10':{'4':30,'5':31,'9':32,'11':33,'12':34}, '11':{'5':35,'10':36,'12':37,'14':38}, '12':{'10':39,'11':40,'13':41},
         '13':{'12':42,'14':43}, '14':{'11':44,'13':45}
         } 
numedgesD = 46
LspansD = 80

nodesAL = ['1','2','3','4','5','6','7','8','9','10','11']

graphAL = {'1':{'4':1200,'5':1600},'2':{'3':1100,'7':300},'3':{'2':1100,'8':300},    
         '4':{'1':1200,'5':1500,'9':500},'5':{'1':1600,'4':1500,'6':900}, '6':{'5':900,'7':700,'11':1000},
         '7':{'2':300,'6':700,'10':1100}, '8':{'3':300,'10':900}, '9':{'4':500,'11':2200},
         '10':{'7':1100,'8':900,'11':1100}, '11':{'6':1000,'9':2200,'10':1100}
         } 
graphnormAL = {'1':{'4':1200,'5':1600},'2':{'3':1100,'7':300},'3':{'2':1100,'8':300},    
         '4':{'1':1200,'5':1500,'9':500},'5':{'1':1600,'4':1500,'6':900}, '6':{'5':900,'7':700,'11':1000},
         '7':{'2':300,'6':700,'10':1100}, '8':{'3':300,'10':900}, '9':{'4':500,'11':2200},
         '10':{'7':1100,'8':900,'11':1100}, '11':{'6':1000,'9':2200,'10':1100}
         } 
edgesAL = {'1':{'4':0,'5':1},'2':{'3':2,'7':3},'3':{'2':4,'8':5},    
         '4':{'1':6,'5':7,'9':8},'5':{'1':9,'4':10,'6':11}, '6':{'5':12,'7':13,'11':14},
         '7':{'2':15,'6':16,'10':17}, '8':{'3':18,'10':19}, '9':{'4':20,'11':21},
         '10':{'7':22,'8':23,'11':24}, '11':{'6':25,'9':26,'10':27}
         } 
numedgesAL = 28
LspansAL = 100

graphA = graphN
if graphA == graphN:
    graphnormA = graphnormN
    numedgesA = numedgesN
    nodesA = nodesN
    edgesA = edgesN
    LspansA = LspansN
elif graphA == graphD:
    graphnormA = graphnormD
    numedgesA = numedgesD
    nodesA = nodesD
    edgesA = edgesD
    LspansA = LspansD
elif graphA == graphAL:
    graphnormA = graphnormAL
    numedgesA = numedgesAL
    nodesA = nodesAL
    edgesA = edgesAL
    LspansA = LspansAL    

def getedgelen(graph,numedges):
    edgelens = np.empty([numedges,1])
    count = 0
    for key in graph:
        for key2 in graph.get(key):
            #print(graph.get(key).get(key2))
            edgelens[count] = graph.get(key).get(key2)
            count = count + 1
    return edgelens 
edgelensA = getedgelen(graphA, numedgesA)

# %%

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

OSNRmeasBW = 12.478 # OSNR measurement BW [GHz]
Rs = 32 # Nyquist channel spacing [GHz]
testlen = 1000.0     # all ageing effects modelled using values in: Faster return of investment in WDM networks when elastic transponders dynamically fit ageing of link margins, Pesic et al.
years = np.linspace(0,10,21) # define how many years are in the lifetime of the network and the resolution 
numyears = np.size(years)
sd = np.linspace(0.04, 0.08, np.size(years)) # added SNR uncertainty SD - assumed to double over lifetime
numlam = np.linspace(30, 150, np.size(years)) # define the channel loading over the network lifetime 
NF = np.linspace(4.5,5.5,np.size(years)) # define the NF ageing of the amplifiers 
alpha = 0.2 + 0.00163669*years # define the fibre ageing due to splice losses over time 
trxaging = ((1 + 0.05*years)*2).reshape(np.size(years),1)*(OSNRmeasBW/Rs) # define TRx ageing 
oxcaging = ((0.03 + 0.007*years)*2).reshape(np.size(years),1)*(OSNRmeasBW/Rs) # define filter ageing, assuming two filters per link, one at Tx and one at Rx
# =============================================================================

# find the worst-case margin required                         
fmD = sd[-1]*5 # D margin is defined as 5xEoL SNR uncertainty SD that is added
fmDGI = sd[0]*5 
#fmT = trxaging[-1] + oxcaging[-1] + fmD # total static margin: amplifier ageing, NL loading and fibre ageing included in GN
#fmTGI = trxaging[0] + oxcaging[0] + fmDGI # total static margin: use the BoL values for GP initial planning case



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
        #snr = (Pch/(Pase + Gnli*Rs*1e9)) 
        snr = (Pch/(Pase + Gnli*Rs*1e9)) - trxaging[yearind] - oxcaging[yearind]
        snr = ( snr**(-1) + (db2lin(TRxb2b))**(-1) )**(-1)
        return lin2db(snr) #- trxaging[yearind] - oxcaging[yearind]
    
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
        snr = (Pch/(Pase + Gnli*Rs*1e9)) - trxaging[yearind] - oxcaging[yearind]
        snr = ( snr**(-1) + (db2lin(TRxb2b))**(-1) )**(-1)
        #snr = snr + np.random.normal(0,db2lin(sd),numpoints)
        sdnorm = sd[yearind]
        return lin2db(snr) + np.random.normal(0,sdnorm,1) #- trxaging[yearind] - oxcaging[yearind]
    
test1 = fmsnr(1200, 100, numlam[-1], NF[-1], alpha[-1], -1)
test2 = SNRnew(1200,numlam[-1], -1)
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
        snr = (Pch/(Pase + Gnli*Rs*1e9)) - trxaging[yearind] - oxcaging[yearind]
        snr = ( snr**(-1) + (db2lin(TRxb2b))**(-1) )**(-1)
        #snr = snr + np.random.normal(0,db2lin(sd),numpoints)
        sdnorm = sd[yearind]
        return lin2db(snr) + np.random.normal(0,sdnorm,numpoints) 
test3 = SNRgen(4800,numlam[-1], -1)
# define the FEC thresholds - all correspond to BER of 2e-2 (2% FEC) - given by MATLAB bertool 
FT2 = 3.243 
FT4 = 6.254 
FT8 = 10.697
FT16 = 12.707
FT32 = 16.579
FT64 = 18.432
FT128 = 22.185

# %% sigma check
# =============================================================================
# sigma = np.empty([numyears,1])
# for i in range(numyears):
#     truesnr = SNRgen(3000,numlam[i],i)
#     x = np.linspace(0,numpoints-1,numpoints)
#     prmn, sigma[i] = GPtrain(x,truesnr)
# plt.plot(years, sigma, label = 'GP')
# plt.plot(years, sd)
# plt.legend()
# plt.xlabel("time (years)")
# plt.ylabel("$\sigma$ (dB)")
# plt.savefig('YGPsigmavsSD.pdf', dpi=200,bbox_inches='tight')
# plt.show()
# =============================================================================
# %%

def initreach(linklen):
    
    gnSNRF = fmsnr(linklen, LspansA, numlam[-1], NF[-1], alpha[-1], -1)
    gnSNRG = fmsnr(linklen, LspansA, numlam[0], NF[0], alpha[0], 0)
    # fixed margin case
    if gnSNRF - fmD > FT128:
        MF = 128
        UmF = gnSNRF - fmD - FT128
    elif gnSNRF - fmD > FT64:
        MF = 64
        UmF = gnSNRF - fmD - FT64
    elif gnSNRF - fmD > FT32:
        MF = 32
        UmF = gnSNRF - fmD - FT32
    elif gnSNRF - fmD > FT16:
        MF = 16
        UmF = gnSNRF - fmD - FT16
    elif gnSNRF - fmD > FT8:
        MF = 8
        UmF = gnSNRF - fmD - FT8
    elif gnSNRF - fmD > FT4:
        MF = 4
        UmF = gnSNRF - fmD - FT4
    elif gnSNRF - fmD > FT2:
        MF = 2
        UmF = gnSNRF - fmD - FT2
    else:
        print("not able to establish a link")

    # GP case   
    if gnSNRG - fmDGI > FT128:
        MFG = 128
        UmG = gnSNRG - fmDGI - FT128
    elif gnSNRG - fmDGI > FT64:
        MFG = 64
        UmG = gnSNRG - fmDGI - FT64
    elif gnSNRG - fmDGI > FT32:
        MFG = 32
        UmG = gnSNRG - fmDGI - FT32
    elif gnSNRG - fmDGI > FT16:
        MFG = 16
        UmG = gnSNRG - fmDGI - FT16
    elif gnSNRG - fmDGI > FT8:
        MFG = 8
        UmG = gnSNRG - fmDGI - FT8
    elif gnSNRG - fmDGI > FT4:
        MFG = 4
        UmG = gnSNRG - fmDGI - FT4
    elif gnSNRG - fmDGI > FT2:
        MFG = 2
        UmG = gnSNRG - fmDGI - FT2
    else:
        print("not able to establish a link")
    return MF, MFG, UmF, UmG

mf = np.empty([numedgesA,1])
mfGi = np.empty([numedgesA,1])
UmF = np.empty([numedgesA,1])
UmGi = np.empty([numedgesA,1])
for i in range(numedgesA):
    mf[i], mfGi[i], UmF[i], UmGi[i] = initreach(edgelensA[i])

# %%

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
    elif (prmn - FT32)/sigma > numsig:
        FT = FT32
        mfG = 32
    elif (prmn - FT16)/sigma > numsig:
        FT = FT16
        mfG = 16
    elif (prmn - FT8)/sigma > numsig:
        FT = FT8
        mfG = 8
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

modf = np.empty([numedgesA,numyears])
estbl = np.empty([numedgesA,numyears])
Um = np.empty([numedgesA,numyears])
for i in range(numedgesA):
    for j in range(numyears):
        modf[i][j], estbl[i][j], Um[i][j] = GPreach(edgelensA[i], 5, j)
    #for j in range(numlens):
        #modf[i][j], estbl[i][j], Um[i][j] = GPreach(linklen[j], 5, i)

# =============================================================================
# modfT = np.transpose(modf)
# UmT = np.transpose(Um)
# estblT = np.transpose(estbl)
# =============================================================================
# %% throughput calculation 
if graphA == graphN:
    np.savetxt('modfN.csv', modf, delimiter=',') 
    np.savetxt('UmN.csv', Um, delimiter=',') 
    np.savetxt('mfN.csv', mf, delimiter=',') 
    np.savetxt('UmFN.csv', UmF, delimiter=',') 
elif graphA == graphAL:
    np.savetxt('modfAL.csv', modf, delimiter=',') 
    np.savetxt('UmAL.csv', Um, delimiter=',') 
    np.savetxt('mfAL.csv', mf, delimiter=',') 
    np.savetxt('UmFAL.csv', UmF, delimiter=',')     
if graphA == graphD:
    np.savetxt('modfD.csv', modf, delimiter=',') 
    np.savetxt('UmD.csv', Um, delimiter=',') 
    np.savetxt('mfD.csv', mf, delimiter=',') 
    np.savetxt('UmFD.csv', UmF, delimiter=',') 

Rs = 32
totmodgp = np.sum(modf,axis=0).reshape(numyears,1)
thrptgp = np.log2(totmodgp)*2*Rs
totUmgp = np.sum(Um,axis=0).reshape(numyears,1)
totthrptgp = np.multiply(thrptgp,numlam.reshape(numyears,1))/1e3


mfpl = mf*np.ones(numyears)
UmFpl = UmF*np.ones(numyears)
totmodfm = np.sum(mfpl,axis=0).reshape(numyears,1)
thrptfm = np.log2(totmodfm)*2*Rs
totthrptfm = np.multiply(thrptfm,numlam.reshape(numyears,1))/1e3
totUmfm = np.sum(UmFpl,axis=0).reshape(numyears,1)

totthrptdiff = ((totthrptgp - totthrptfm)/totthrptfm)*100
# record changes to the MF between QoT-based planning and network switch-on
gpmfadjust = mfGi - np.transpose(modf)[0].reshape(numedgesA,1)
    
# %% plotting 
if graphA == graphN:
    suffix = 'N'
elif graphA == graphAL:
    suffix = 'AL'
elif graphA == graphD:
    suffix = 'D'
    
plt.plot(years, thrptgp, label = 'GP')
plt.plot(years, thrptfm, label = 'FM')
plt.xlabel("time (years)")
plt.ylabel("total throughput per ch. (Gb/s)")
plt.legend()
plt.savefig('Ytotalthrptperch' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.show()

plt.plot(years, totthrptgp, label = 'GP')
plt.plot(years, totthrptfm, label = 'FM')
plt.xlabel("time (years)")
plt.ylabel("total throughput (Tb/s)")
plt.legend()
plt.savefig('Ytotalthrpt' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.show()

plt.plot(years, totthrptdiff, label = 'GP - FM')
plt.xlabel("time (years)")
plt.ylabel("total throughput gain (%)")
plt.legend()
plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.show()


plt.plot(years, totUmgp, label = 'GP')
plt.plot(years, totUmfm, label = 'FM')
plt.xlabel("time (years)")
plt.ylabel("total U margin (dB)")
plt.legend()
plt.savefig('YtotalU' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.show()

# %%
linkind = 2

plt.plot(years, Um[linkind], label = "GP")
plt.plot(years, UmFpl[linkind], label = "FM")
plt.xlabel("time (years)")
plt.ylabel("U margin (dB)")
plt.legend()
plt.savefig('YonelinkU' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.show()

plt.plot(years, modf[linkind], label = "GP")
plt.plot(years, mfpl[linkind], label = "FM")
plt.xlabel("time (years)")
plt.ylabel("QAM order")
plt.legend()
plt.savefig('YonelinkMF' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
plt.show()

# %% heteroscedastic data generation
hetdatagen = True
if hetdatagen:
    def datatshet(edgelen, Lspans, numlam, NF,sd, alpha, yearind):
                
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
            #Pun = GNmain(Lspans, 1, numlam, 101, 201, alpha, Disp, PchdBm, NF, NLco,False,numpoints)[0] 
            #Popt = PchdBm[np.argmax(Pun)]  
            
            Gwdm = (1e-3*10**(Popt/10)*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
            Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))*numspans
            Pase = NF*h*f*(db2lin(alpha*Lspans) - 1)*Rs*1e9*numspans
            Pch = 1e-3*10**(Popt/10) 
            snr = (Pch/(Pase + Gnli*Rs*1e9)) - db2lin(trxagingh[yearind] + oxcagingh[yearind]) # subtract static ageing effects
            snr = ( snr**(-1) + (db2lin(TRxb2b))**(-1) )**(-1) # add TRx B2B noise 
            #snr = snr + np.random.normal(0,db2lin(sd),numpoints)
            sdnorm = sd # noise on each link is assumed to be proportional to the link length 
            return lin2db(snr) + np.random.normal(0,sdnorm,1) 
        
    def hetsave(edgelens, numedges,Lspans, yearind):    
            linkSNR = np.empty([numedges,1])
            for i in range(numedges):
                linkSNR[i] = datatshet(edgelens[i],Lspans,numlamh[yearind], NFh[yearind], sdh[yearind], alphah[yearind], yearind)
            TRxSNR = 26 # add TRx noise of 26dB B2B 
            linkSNR = lin2db( 1/(  1/(db2lin(linkSNR)) + 1/(db2lin(TRxSNR))  ))
            linkSNR = linkSNR.reshape(numedges)
    # =============================================================================
    #         if graphA == graphN:
    #             np.savetxt('tsSNRN' + str(int(yearind)) + '.csv', linkSNR, delimiter=',') 
    #             linkPopt = linkPopt.reshape(1,1)
    #             np.savetxt('tsPoptN' + str(int(yearind)) + '.csv', linkPopt, delimiter=',') 
    #         elif graphA == graphD:
    #             np.savetxt('tsSNRD' + str(int(yearind)) + '.csv', linkSNR, delimiter=',') 
    #             linkPopt = linkPopt.reshape(1,1)
    #             np.savetxt('tsPoptD' + str(int(yearind)) + '.csv', linkPopt, delimiter=',') 
    #         elif graphA == graphAL:
    #             np.savetxt('tsSNRAL' + str(int(yearind)) + '.csv', linkSNR, delimiter=',')
    #             linkPopt = linkPopt.reshape(1,1)
    #             np.savetxt('tsPoptAL' + str(int(yearind)) + '.csv', linkPopt, delimiter=',') 
    # =============================================================================
            return linkSNR
    numphet = 5  
    numyrsh = 200
    yearsh = np.linspace(0,10,numyrsh)
    numlamh = np.linspace(30,150,numyrsh,dtype=int)
    NFh = np.linspace(4.5,5.5,numyrsh)
    sdh = np.linspace(0.04,0.08,numyrsh)
    alphah = 0.2 + 0.00163669*yearsh
    hetdata = np.empty([numyrsh,numedgesA])
    trxagingh = ((1 + 0.05*yearsh)*2).reshape(np.size(yearsh),1) 
    oxcagingh = ((0.03 + 0.007*yearsh)*2).reshape(np.size(yearsh),1)
    #linkPopt = np.empty([numyears,1])
    for i in range(numyrsh):
        hetdata[i] = hetsave(edgelensA, numedgesA, LspansA, i)
    hetdata = np.transpose(hetdata)
    if graphA == graphN:
        np.savetxt('hetdataN.csv', hetdata, delimiter=',') 
    elif graphA == graphD:
        np.savetxt('hetdataD.csv', hetdata, delimiter=',') 
    elif graphA == graphAL:
        np.savetxt('hetdataAL.csv', hetdata, delimiter=',') 









