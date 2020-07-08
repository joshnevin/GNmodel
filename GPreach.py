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

nyqch = False

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
# =============================================================================
# else:
#     graphN = {'1':{'2':2100,'3':3000,'8':4000},'2':{'1':2100,'3':1200,'4':1500},'3':{'1':3000,'2':1200,'6':3600},    
#              '4':{'2':1500,'5':1200,'11':3900},'5':{'4':1200,'6':2400,'7':1200}, '6':{'3':3600,'5':2400,'10':2100,'14':3600},
#              '7':{'5':1200,'8':1500,'10':2700}, '8':{'1':4800,'7':1500,'9':1500}, '9':{'8':1500,'10':1500,'12':600,'13':600},
#              '10':{'6':2100,'7':2700,'9':1500}, '11':{'4':3900,'12':1200,'13':1500}, '12':{'9':600,'11':1200,'14':600},
#              '13':{'9':600,'11':1500,'14':300}, '14':{'6':3600,'12':600,'13':300}
#              } 
#     graphnormN = {'1':{'2':2100,'3':3000,'8':4000},'2':{'1':2100,'3':1200,'4':1500},'3':{'1':3000,'2':1200,'6':3600},    
#              '4':{'2':1500,'5':1200,'11':3900},'5':{'4':1200,'6':2400,'7':1200}, '6':{'3':3600,'5':2400,'10':2100,'14':3600},
#              '7':{'5':1200,'8':1500,'10':2700}, '8':{'1':4800,'7':1500,'9':1500}, '9':{'8':1500,'10':1500,'12':600,'13':600},
#              '10':{'6':2100,'7':2700,'9':1500}, '11':{'4':3900,'12':1200,'13':1500}, '12':{'9':600,'11':1200,'14':600},
#              '13':{'9':600,'11':1500,'14':300}, '14':{'6':3600,'12':600,'13':300}
# =============================================================================
             #}  
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
Disp = 16.7
linklen = np.linspace(100, 5000, 50)
numlens = len(linklen)

def marginsnrtest(edgelen, Lspans, numlam, NF, alpha):
    lam = 1550 # operating wavelength centre [nm]
    f = 299792458/(lam*1e-9) # operating frequency [Hz]
    c = 299792.458 # speed of light in vacuum [nm/ps] -> needed for calculation of beta2
    Rs = 32 # symbol rate [GBaud]
    h = 6.63*1e-34  # Planck's constant [Js]
    if nyqch:
        NchNy = numlam
        BWNy = (NchNy*Rs)/1e3 # keep Nyquist spacing for non-saturated channel
    else:
        NchRS = numlam
        BchRS = 41.6
        Df = 50
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
    if nyqch:
        Gwdmsw = (Pchsw*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
        Gnlisw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))
    else:
        Gwdmsw = Pchsw/(BchRS*1e9)
        Gnlisw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))  
    G = al*Ls
    NFl = 10**(NF/10) 
    Gl = 10**(G/10) 
    Pasesw = NFl*h*f*(Gl - 1)*Rs*1e9 # [W] the ASE noise power in one Nyquist channel across all spans
    snrsw = (Pchsw)/(Pasesw*np.ones(numpch) + Gnlisw*Rs*1e9)
    Popt = PchdBm[np.argmax(snrsw)]                                                   
    if nyqch:
        Gwdm = (1e-3*10**(Popt/10)*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
        Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))*Numspans
    else:
        Gwdm = (1e-3*10**(Popt/10))/(BchRS*1e9)
        Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))*Numspans
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
years = np.linspace(0,10,61) # define how many years are in the lifetime of the network and the resolution 
numyears = np.size(years)
sd = np.linspace(0.04, 0.08, np.size(years)) # added SNR uncertainty SD - assumed to double over lifetime
#numlam = np.linspace(30, 150, np.size(years)) # Nyquist: define the channel loading over the network lifetime 
numlam = np.linspace(20, 100, np.size(years)) 
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


# %% heteroscedastic data generation
hetdatagen = False
if hetdatagen:
    def datatshet(edgelen, Lspans, numlam, NF,sd, alpha, yearind, nyqch):
            Ls = Lspans
            D = Disp
            gam = NLco
            lam = 1550 # operating wavelength centre [nm]
            f = 299792458/(lam*1e-9) # operating frequency [Hz]
            c = 299792.458 # speed of light in vacuum [nm/ps] -> needed for calculation of beta2
            Rs = 32 # symbol rate [GBaud]
            h = 6.63*1e-34  # Planck's constant [Js]
            if nyqch:
                NchNy = numlam
                BWNy = (NchNy*Rs)/1e3 
            else:
                NchRS = numlam
                Df = 50 # 50 GHz grid 
                BchRS = 41.6 # RS from GN model paper - raised cosine + roll-off of 0.3 
            allin = np.log((10**(alpha/10)))/2 # fibre loss [1/km] -> weird definition, due to exponential decay of electric field instead of power, which is standard 
            beta2 = (D*(lam**2))/(2*np.pi*c) # dispersion coefficient at given wavelength [ps^2/km]
            Leff = (1 - np.exp(-2*allin*Ls ))/(2*allin)  # effective length [km]      
            Leffa = 1/(2*allin)  # the asymptotic effective length [km]  
            numspans = int(edgelen/Lspans)
            
            numpch = len(PchdBm)
            Pchsw = 1e-3*10**(PchdBm/10)  # ^ [W]
            if nyqch:
                Gwdmsw = (Pchsw*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
                Gnlisw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))
            else:
                Gwdmsw = Pchsw/(BchRS*1e9)
                Gnlisw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))
            G = alpha*Ls
            NFl = 10**(NF/10) 
            Gl = 10**(G/10) 
            Pasesw = NFl*h*f*(Gl - 1)*Rs*1e9 # [W] the ASE noise power in one Nyquist channel across all spans
            snrsw = (Pchsw)/(Pasesw*np.ones(numpch) + Gnlisw*Rs*1e9)
            Popt = PchdBm[np.argmax(snrsw)]     
            #Pun = GNmain(Lspans, 1, numlam, 101, 201, alpha, Disp, PchdBm, NF, NLco,False,numpoints)[0] 
            #Popt = PchdBm[np.argmax(Pun)]  
            
            if nyqch:
                Gwdm = (1e-3*10**(Popt/10)*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
                Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))*numspans
            else:
                Gwdm = (1e-3*10**(Popt/10))/(BchRS*1e9)
                Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))*numspans
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
                linkSNR[i] = datatshet(edgelens[i],Lspans,numlamh[yearind], NFh[yearind], sdh[yearind], alphah[yearind], yearind, False)
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
    numyrsh = 101
    yearsh = np.linspace(0,10,numyrsh)
    numlamh = np.linspace(20,100,numyrsh,dtype=int)
    NFh = np.linspace(4.5,5.5,numyrsh)
    sdh = np.linspace(0.04,0.08,numyrsh)
    #sdh = 0.04 + 0.0004*yearsh**2
    #sdh = 0.04 + (0.04/10**0.5)*yearsh**0.5
    #sdh = 0.04 + (0.36/10**0.5)*yearsh**0.5
    #sdh = 0.04 + 8e-4*yearsh**2
    alphah = 0.2 + 0.00163669*yearsh
    hetdata = np.empty([numyrsh,numedgesA])
    trxagingh = ((1 + 0.05*yearsh)*2).reshape(np.size(yearsh),1) 
    oxcagingh = ((0.03 + 0.007*yearsh)*2).reshape(np.size(yearsh),1)
    #linkPopt = np.empty([numyears,1])

    plt.plot(yearsh, sdh, label = 'linear')
    #plt.plot(yearsh, sdh2, label = 'quadratic')
    #plt.plot(yearsh, sdh3, label = 'square root')
    plt.xlabel("time (years)")
    plt.ylabel("$\sigma$(dB)")
    plt.legend()
    plt.savefig('sigmavstime.pdf', dpi=200,bbox_inches='tight')
    plt.show()
 
# =============================================================================
#     for i in range(numyrsh):
#         hetdata[i] = hetsave(edgelensA, numedgesA, LspansA, i)
#     hetdata = np.transpose(hetdata)
#     if graphA == graphN:
#         np.savetxt('hetdataN.csv', hetdata, delimiter=',') 
#         np.savetxt('hetsdvarN.csv', sdh, delimiter=',') 
#     elif graphA == graphD:
#         np.savetxt('hetdataD.csv', hetdata, delimiter=',') 
#         np.savetxt('hetsdvarD.csv', sdh, delimiter=',') 
#     elif graphA == graphAL:
#         np.savetxt('hetdataAL.csv', hetdata, delimiter=',')
#         np.savetxt('hetsdvarAL.csv', sdh, delimiter=',') 
# =============================================================================
    testlens = [500,1000,2000,3000]
    
    hetdata = np.empty([len(testlens),numyrsh])
    #hetdata1000 = np.empty([numyrsh,1])
    #hetdata2000 = np.empty([numyrsh,1])
    #hetdata3000 = np.empty([numyrsh,1])
    for i in range(len(testlens)):
        for j in range(numyrsh):
            hetdata[i][j] = datatshet(testlens[i], 100, numlamh[j], NFh[j], sdh[j], alphah[j], j, False)
    
    np.savetxt('hetdata.csv', hetdata, delimiter=',')
# %%

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


def fmsnr(edgelen, Lspans, numlam, NF, alpha, yearind, nyqch):
        Ls = Lspans
        D = Disp
        gam = NLco
        lam = 1550 # operating wavelength centre [nm]
        f = 299792458/(lam*1e-9) # operating frequency [Hz]
        c = 299792.458 # speed of light in vacuum [nm/ps] -> needed for calculation of beta2
        Rs = 32 # symbol rate [GBaud]
        h = 6.63*1e-34  # Planck's constant [Js]
        if nyqch:
            NchNy = numlam
            BWNy = (NchNy*Rs)/1e3 
        else:
            NchRS = numlam
            Df = 50 # 50 GHz grid 
            BchRS = 41.6 # RS from GN model paper - raised cosine + roll-off of 0.3 
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
        if nyqch:
            Gwdmsw = (Pchsw*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
            Gnlisw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))
        else:
            Gwdmsw = Pchsw/(BchRS*1e9)
            Gnlisw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))
        G = alpha*Ls
        NFl = 10**(NF/10) 
        Gl = 10**(G/10) 
        Pasesw = NFl*h*f*(Gl - 1)*Rs*1e9 # [W] the ASE noise power in one Nyquist channel across all spans
        #snrsw = (Pchsw)/(Pasesw*np.ones(numpch) + GnliEq13sw*Rs*1e9)
        snrsw = (Pchsw)/(Pasesw*np.ones(numpch) + Gnlisw*Rs*1e9)
        Popt = PchdBm[np.argmax(snrsw)]  
        # ======================= find Popt ===============================
        if nyqch:
            Gwdm = (1e-3*10**(Popt/10)*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
            Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))*numspans
        else:
            Gwdm = (1e-3*10**(Popt/10))/(BchRS*1e9)
            Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))*numspans
        Pase = NF*h*f*(db2lin(alpha*Lspans) - 1)*Rs*1e9*numspans
        Pch = 1e-3*10**(Popt/10) 
        #snr = (Pch/(Pase + Gnli*Rs*1e9)) 
        snr = (Pch/(Pase + Gnli*Rs*1e9)) - trxaging[yearind] - oxcaging[yearind]
        snr = ( snr**(-1) + (db2lin(TRxb2b))**(-1) )**(-1)
        return lin2db(snr) #- trxaging[yearind] - oxcaging[yearind]
  
def SNRnew(edgelen,numlam, yearind, nyqch):  # function for generating a new SNR value to test if uncertainty is dealt with
        Ls = LspansA
        D = Disp          # rather than the worst-case number 
        gam = NLco
        lam = 1550 # operating wavelength centre [nm]
        f = 299792458/(lam*1e-9) # operating frequency [Hz]
        c = 299792.458 # speed of light in vacuum [nm/ps] -> needed for calculation of beta2
        Rs = 32 # symbol rate [GBaud]
        h = 6.63*1e-34  # Planck's constant [Js]
        if nyqch:
            NchNy = numlam
            BWNy = (NchNy*Rs)/1e3 
        else:
            NchRS = numlam
            Df = 50 # 50 GHz grid 
            BchRS = 41.6 # RS from GN model paper - raised cosine + roll-off of 0.3 
        allin = np.log((10**(alpha[yearind]/10)))/2 # fibre loss [1/km] -> weird definition, due to exponential decay of electric field instead of power, which is standard 
        beta2 = (D*(lam**2))/(2*np.pi*c) # dispersion coefficient at given wavelength [ps^2/km]
        Leff = (1 - np.exp(-2*allin*Ls ))/(2*allin)  # effective length [km]      
        Leffa = 1/(2*allin)  # the asymptotic effective length [km]  
        numspans = int(edgelen/Ls)
        # ===================== find Popt ==========================
        numpch = len(PchdBm)
        Pchsw = 1e-3*10**(PchdBm/10)  # ^ [W]
        if nyqch:
            Gwdmsw = (Pchsw*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
            Gnlisw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))
        else:
            Gwdmsw = Pchsw/(BchRS*1e9)
            Gnlisw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))
        G = alpha[yearind]*Ls
        NFl = 10**(NF[yearind]/10) 
        Gl = 10**(G/10) 
        Pasesw = NFl*h*f*(Gl - 1)*Rs*1e9 # [W] the ASE noise power in one Nyquist channel across all spans
        snrsw = (Pchsw)/(Pasesw*np.ones(numpch) + Gnlisw*Rs*1e9)
        Popt = PchdBm[np.argmax(snrsw)]  
        # ===================== find Popt ==========================
        if nyqch:
            Gwdm = (1e-3*10**(Popt/10)*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
            Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))*numspans
        else:
            Gwdm = (1e-3*10**(Popt/10))/(BchRS*1e9)
            Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))*numspans                                                
        Pase = NF[yearind]*h*f*(db2lin(alpha[yearind]*Ls) - 1)*Rs*1e9*numspans
        Pch = 1e-3*10**(Popt/10) 
        snr = (Pch/(Pase + Gnli*Rs*1e9)) - trxaging[yearind] - oxcaging[yearind]
        snr = ( snr**(-1) + (db2lin(TRxb2b))**(-1) )**(-1)
        #snr = snr + np.random.normal(0,db2lin(sd),numpoints)
        sdnorm = sd[yearind]
        return lin2db(snr) + np.random.normal(0,sdnorm,1) #- trxaging[yearind] - oxcaging[yearind]
    
test1 = fmsnr(4800, 100, numlam[-1], NF[-1], alpha[-1], -1, False)
test2 = SNRnew(4800,numlam[-1], -1, False)
# =============================================================================
snrtest = np.linspace(0,20,100)
capsh = 41.6*np.log2(1 + db2lin(snrtest))
#cap2 = 32*np.log2(2)*np.ones(100)
cap4 = 32*np.log2(4)*np.ones(100)
#cap8 = 32*np.log2(8)*np.ones(100)
cap16 = 32*np.log2(16)*np.ones(100)
# cap32 = 32*np.log2(32)*np.ones(100)
cap64 = 32*np.log2(64)*np.ones(100)
# cap128 = 32*np.log2(128)*np.ones(100)
plt.plot(snrtest, capsh, label = "Shannon")
#plt.plot(snrtest, cap2, label = "BPSK")
plt.plot(snrtest, cap4, label = "QPSK")
#plt.plot(snrtest, cap8, label = "8QAM")
plt.plot(snrtest, cap16, label = "16QAM")
# plt.plot(snrtest, cap32, label = "32QAM")
plt.plot(snrtest, cap64, label = "64QAM")
# plt.plot(snrtest, cap128, label = "128QAM")
plt.xlabel("SNR (dB)")
plt.ylabel("Gb/s")
plt.legend(ncol = 2)
plt.savefig('shannontest.pdf', dpi=200,bbox_inches='tight')
plt.show()
# =============================================================================

# %%

def SNRgen(edgelen,numlam, yearind, nyqch):  # function for generating a new SNR value to test if uncertainty is dealt with
        Ls = LspansA
        D = Disp          # rather than the worst-case number 
        gam = NLco
        lam = 1550 # operating wavelength centre [nm]
        f = 299792458/(lam*1e-9) # operating frequency [Hz]
        c = 299792.458 # speed of light in vacuum [nm/ps] -> needed for calculation of beta2
        Rs = 32 # symbol rate [GBaud]
        h = 6.63*1e-34  # Planck's constant [Js]
        if nyqch:
            NchNy = numlam
            BWNy = (NchNy*Rs)/1e3 
        else:
            NchRS = numlam
            Df = 50 # 50 GHz grid 
            BchRS = 41.6 # RS from GN model paper - raised cosine + roll-off of 0.3 
        allin = np.log((10**(alpha[yearind]/10)))/2 # fibre loss [1/km] -> weird definition, due to exponential decay of electric field instead of power, which is standard 
        beta2 = (D*(lam**2))/(2*np.pi*c) # dispersion coefficient at given wavelength [ps^2/km]
        Leff = (1 - np.exp(-2*allin*Ls ))/(2*allin)  # effective length [km]      
        Leffa = 1/(2*allin)  # the asymptotic effective length [km]  
        numspans = int(edgelen/Ls)
        # ===================== find Popt ==========================
        numpch = len(PchdBm)
        Pchsw = 1e-3*10**(PchdBm/10)  # ^ [W]
        if nyqch:
            Gwdmsw = (Pchsw*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
            Gnlisw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))
        else:
            Gwdmsw = Pchsw/(BchRS*1e9)
            Gnlisw = 1e24*(8/27)*(gam**2)*(Gwdmsw**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))
        G = alpha[yearind]*Ls
        NFl = 10**(NF[yearind]/10) 
        Gl = 10**(G/10) 
        Pasesw = NFl*h*f*(Gl - 1)*Rs*1e9 # [W] the ASE noise power in one Nyquist channel across all spans
        snrsw = (Pchsw)/(Pasesw*np.ones(numpch) + Gnlisw*Rs*1e9)
        Popt = PchdBm[np.argmax(snrsw)]  
        # ===================== find Popt ==========================
        if nyqch:
            Gwdm = (1e-3*10**(Popt/10)*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
            Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))*numspans
        else:
            Gwdm = (1e-3*10**(Popt/10))/(BchRS*1e9)
            Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))*numspans                                                                             
        Pase = NF[yearind]*h*f*(db2lin(alpha[yearind]*Ls) - 1)*Rs*1e9*numspans
        Pch = 1e-3*10**(Popt/10) 
        snr = (Pch/(Pase + Gnli*Rs*1e9)) - trxaging[yearind] - oxcaging[yearind]
        snr = ( snr**(-1) + (db2lin(TRxb2b))**(-1) )**(-1)
        #snr = snr + np.random.normal(0,db2lin(sd),numpoints)
        sdnorm = sd[yearind]
        return lin2db(snr) + np.random.normal(0,sdnorm,numpoints) 
test3 = SNRgen(4800,numlam[-1], -1, False)
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
#

def initreach(linklen):
    
    gnSNRF = fmsnr(linklen, LspansA, numlam[-1], NF[-1], alpha[-1], -1, False)
    gnSNRG = fmsnr(linklen, LspansA, numlam[0], NF[0], alpha[0], 0, False)
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
    
def GPreach(linklen, numsig, yearind, nyqch):
    truesnr = SNRgen(linklen,numlam[yearind],yearind, nyqch)
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
    if nyqch:
        C = 32*np.log2(1 + db2lin(prmn))
    else:
        C = 41.6*np.log2(1 + db2lin(prmn))
    if SNRnew(linklen,numlam[yearind],yearind, nyqch) > FT:
        estbl = 1
        Um = prmn - numsig*sigma - FT
    else:
        estbl = 0
    return mfG, estbl, Um, C
      
def rtmreach(linklen, numsig, yearind, nyqch): # RTM = real-time + (D) margin
    truesnr = SNRgen(linklen,numlam[yearind],yearind, nyqch)
    meansnr = np.mean(truesnr)
    if meansnr - fmD > FT128:
        FT = FT128
        mfG = 128
    elif meansnr - fmD > FT64:
        FT = FT64
        mfG = 64
    elif meansnr - fmD > FT32:
        FT = FT32
        mfG = 32
    elif meansnr - fmD > FT16:
        FT = FT16
        mfG = 16
    elif meansnr - fmD > FT8:
        FT = FT8
        mfG = 8
    elif meansnr - fmD > FT4:
        FT = FT4
        mfG = 4
    elif meansnr - fmD > FT2:
        FT = FT2
        mfG = 2
    else:
        print("not able to establish a link")
    if nyqch:
        C = 32*np.log2(1 + db2lin(meansnr))
    else:
        C = 41.6*np.log2(1 + db2lin(meansnr))
    if SNRnew(linklen,numlam[yearind],yearind, nyqch) > FT:
        estbl = 1
        Um = meansnr - fmD - FT
    else:
        estbl = 0
    return mfG, estbl, Um, C
 
modf = np.empty([numedgesA,numyears])
estbl = np.empty([numedgesA,numyears])
Um = np.empty([numedgesA,numyears])
shcp = np.empty([numedgesA,numyears])
modfrt = np.empty([numedgesA,numyears])
estblrt = np.empty([numedgesA,numyears])
Umrt = np.empty([numedgesA,numyears])
shcprt = np.empty([numedgesA,numyears])
start = time.time()
for i in range(numedgesA):
    for j in range(numyears):
        modf[i][j], estbl[i][j], Um[i][j], shcp[i][j] = GPreach(edgelensA[i], 5, j, nyqch)
        modfrt[i][j], estblrt[i][j], Umrt[i][j], shcprt[i][j] = rtmreach(edgelensA[i], 5, j, nyqch)
end = time.time()

print("GP algorithm took" + str(end-start))
    #for j in range(numlens):
        #modf[i][j], estbl[i][j], Um[i][j] = GPreach(linklen[j], 5, i)

# =============================================================================
# modfT = np.transpose(modf)
# UmT = np.transpose(Um)
# estblT = np.transpose(estbl)
# =============================================================================
# %% throughput calculation 
if graphA == graphN:
    np.savetxt('modfhrN.csv', modf, delimiter=',') 
    np.savetxt('UmhrN.csv', Um, delimiter=',') 
    np.savetxt('mfhrN.csv', mf, delimiter=',') 
    np.savetxt('UmFhrN.csv', UmF, delimiter=',') 
    np.savetxt('shcphrN.csv', shcp, delimiter=',') 
    np.savetxt('mfGihrN.csv', mfGi, delimiter=',') 
    np.savetxt('modfhrrtN.csv', modfrt, delimiter=',') 
    np.savetxt('UmhrrtN.csv', Umrt, delimiter=',') 
    np.savetxt('shcphrrtN.csv', shcprt, delimiter=',') 
elif graphA == graphAL:
    np.savetxt('modfAL.csv', modf, delimiter=',') 
    np.savetxt('UmAL.csv', Um, delimiter=',') 
    np.savetxt('mfAL.csv', mf, delimiter=',') 
    np.savetxt('UmFAL.csv', UmF, delimiter=',')   
    np.savetxt('shcpAL.csv', shcp, delimiter=',') 
    np.savetxt('mfGiAL.csv', mfGi, delimiter=',') 
if graphA == graphD:
    np.savetxt('modfD.csv', modf, delimiter=',') 
    np.savetxt('UmD.csv', Um, delimiter=',') 
    np.savetxt('mfD.csv', mf, delimiter=',') 
    np.savetxt('UmFD.csv', UmF, delimiter=',') 
    np.savetxt('shcpD.csv', shcp, delimiter=',') 
    np.savetxt('mfGiD.csv', mfGi, delimiter=',') 
# %%
# =============================================================================
# recoverdata = True
# if recoverdata: 
#     if graphA == graphN:
#         modfN = np.genfromtxt(open("modfN.csv", "r"), delimiter=",", dtype =float)
#         UmN = np.genfromtxt(open("UmN.csv", "r"), delimiter=",", dtype =float)
#         mfN = np.genfromtxt(open("mfN.csv", "r"), delimiter=",", dtype =float)
#         UmFN = np.genfromtxt(open("UmFN.csv", "r"), delimiter=",", dtype =float)
#         shcpN = np.genfromtxt(open("shcpN.csv", "r"), delimiter=",", dtype =float)
#     if graphA == graphD:
#         modfD = np.genfromtxt(open("modfD.csv", "r"), delimiter=",", dtype =float)
#         UmD = np.genfromtxt(open("UmD.csv", "r"), delimiter=",", dtype =float)
#         mfD = np.genfromtxt(open("mfD.csv", "r"), delimiter=",", dtype =float)
#         UmFD = np.genfromtxt(open("UmFD.csv", "r"), delimiter=",", dtype =float)
#         shcpD = np.genfromtxt(open("shcpD.csv", "r"), delimiter=",", dtype =float)
#     if graphA == graphAL:
#         modfAL = np.genfromtxt(open("modfAL.csv", "r"), delimiter=",", dtype =float)
#         UmAL = np.genfromtxt(open("UmAL.csv", "r"), delimiter=",", dtype =float)
#         mfAL = np.genfromtxt(open("mfAL.csv", "r"), delimiter=",", dtype =float)
#         UmAL = np.genfromtxt(open("UmFAL.csv", "r"), delimiter=",", dtype =float)
#         shcpAL = np.genfromtxt(open("shcpAL.csv", "r"), delimiter=",", dtype =float)
# =============================================================================    

modfN = np.genfromtxt(open("modfN.csv", "r"), delimiter=",", dtype =float)
UmN = np.genfromtxt(open("UmN.csv", "r"), delimiter=",", dtype =float)
mfN = np.genfromtxt(open("mfN.csv", "r"), delimiter=",", dtype =float)
UmFN = np.genfromtxt(open("UmFN.csv", "r"), delimiter=",", dtype =float)
shcpN = np.genfromtxt(open("shcpN.csv", "r"), delimiter=",", dtype =float)
modfD = np.genfromtxt(open("modfD.csv", "r"), delimiter=",", dtype =float)
UmD = np.genfromtxt(open("UmD.csv", "r"), delimiter=",", dtype =float)
mfD = np.genfromtxt(open("mfD.csv", "r"), delimiter=",", dtype =float)
UmFD = np.genfromtxt(open("UmFD.csv", "r"), delimiter=",", dtype =float)
shcpD = np.genfromtxt(open("shcpD.csv", "r"), delimiter=",", dtype =float)
modfAL = np.genfromtxt(open("modfAL.csv", "r"), delimiter=",", dtype =float)
UmAL = np.genfromtxt(open("UmAL.csv", "r"), delimiter=",", dtype =float)
mfAL = np.genfromtxt(open("mfAL.csv", "r"), delimiter=",", dtype =float)
UmFAL = np.genfromtxt(open("UmFAL.csv", "r"), delimiter=",", dtype =float)
shcpAL = np.genfromtxt(open("shcpAL.csv", "r"), delimiter=",", dtype =float)

def dataproc(modf, Um, mf, UmF, shcp, numedgesA):

    Rs = 32
    FECOH = 0.2
    thrptgp = np.sum(np.log2(modf),axis=0).reshape(numyears,1)*(1-FECOH)*2*Rs/1e3 # 15% FEC overhead + 5% OTU framing overhead 
    totUmgp = np.sum(Um,axis=0).reshape(numyears,1)
    totthrptgp = np.multiply(thrptgp,numlam.reshape(numyears,1))
    
    totshcp = np.multiply((np.sum(shcp,axis=0).reshape(numyears,1)*(1-FECOH)*2), numlam.reshape(numyears,1))/1e3  # 15% FEC overhead + 5% OTU framing overhead 
    
    mfpl = mf.reshape(numedgesA,1)*np.ones(numyears)
    UmFpl = UmF.reshape(numedgesA,1)*np.ones(numyears)
    thrptfm = np.sum(np.log2(mfpl),axis=0).reshape(numyears,1)*2*Rs*(1-FECOH)/1e3
    totthrptfm = np.multiply(thrptfm,numlam.reshape(numyears,1))
    totUmfm = np.sum(UmFpl,axis=0).reshape(numyears,1)
    
    totthrptdiff = ((totthrptgp - totthrptfm)/totthrptfm)*100
    totthrptdiffsh = ((totshcp - totthrptfm)/totthrptfm)*100
    # record changes to the MF between QoT-based planning and network switch-on
    #gpmfadjust = mfGi - np.transpose(modf)[0].reshape(numedgesA,1)
    
    return thrptgp, thrptfm, totthrptgp, totthrptfm, totthrptdiff, totthrptdiffsh, totUmgp, totUmfm, totshcp, Um, UmFpl, modf, mfpl

#thrptgpN, thrptfmN, totthrptgpN, totthrptfmN, totthrptdiffN, totthrptdiffshN, totUmgpN, totUmfmN, totshcpN, UmN, UmFplN, modfN, mfplN = dataproc(modfN, UmN, mfN, UmFN, shcpN, numedgesN)
#thrptgpAL, thrptfmAL, totthrptgpAL, totthrptfmAL, totthrptdiffAL, totthrptdiffshAL, totUmgpAL, totUmfmAL, totshcpAL, UmAL, UmFplAL, modfAL, mfplAL = dataproc(modfAL, UmAL, mfAL, UmFAL, shcpAL, numedgesAL)
#thrptgpD, thrptfmD, totthrptgpD, totthrptfmD, totthrptdiffD, totthrptdiffshD, totUmgpD, totUmfmD, totshcpD, UmD, UmFplD, modfD, mfplD = dataproc(modfD, UmD, mfD, UmFD, shcpD, numedgesD)

thrptgpN, thrptfmN, totthrptgpN, totthrptfmN, totthrptdiffN, totthrptdiffshN, totUmgpN, totUmfmN, totshcpN, UmN, UmFplN, modfN, mfplN = dataproc(modf, Um, mf, UmF, shcp, numedgesN)
thrptrtN, thrptfmN, totthrptrtN, totthrptfmN, totthrptdiffrtN, totthrptdiffrtshN, totUmrtN, totUmfmN, totshcprtN, UmrtN, UmFplN, modfrtN, mfplN = dataproc(modfrt, Umrt, mfN, UmFN, shcprt, numedgesN)

totthrptintN = np.sum(np.multiply(totthrptdiffN,years.reshape(numyears,1)))
#totthrptintAL = np.sum(np.multiply(totthrptdiffAL,years.reshape(numyears,1)))
#totthrptintD = np.sum(np.multiply(totthrptdiffD,years.reshape(numyears,1)))
# %% plotting for JOCN 2020 
JOCNplot = False
if JOCNplot:
    font = { 'family' : 'sans-serif',
                'weight' : 'normal',
                'size'   : 15}
    matplotlib.rc('font', **font)
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ln2 = ax1.plot(years, totthrptdiffN,'-', color = 'b',label = 'GP N')
    ln3 = ax1.plot(years, totthrptdiffAL,'-', color = 'g',label = 'GP Al')
    ln5 = ax1.plot(years, totthrptdiffD,'-', color = 'r',label = 'GP D')
    
    ln1 = ax2.plot(years, totthrptdiffshN,'--', color = 'b', label = 'Sh. N')
    ln4 = ax2.plot(years, totthrptdiffshAL,'--', color = 'g',label = 'Sh. Al')
    ln6 = ax2.plot(years, totthrptdiffshD,'--', color = 'r',label = 'Sh. D')
    
    ax1.set_xlabel("time (years)")
    ax1.set_ylabel("GP throughput gain (%)")
    ax2.set_ylabel("Shannon throughput gain (%)")
    
    ax1.set_xlim([years[0], years[-1]])
    ax1.set_ylim([totthrptdiffN[-1]-4, totthrptdiffN[0]+4])
    ax2.set_ylim([totthrptdiffshD[-1]-4, totthrptdiffshN[0]+4])
    
    lns = ln1+ln2+ln3+ln4+ln5+ln6
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
    #plt.axis([years[0],years[-1],1.0,8.0])
    #plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
    plt.savefig('JOCNtotalthrptdiff.pdf', dpi=200,bbox_inches='tight')
    plt.show()
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ln6 = ax1.plot(years, totthrptgpN,'--', color = 'b',label = 'GP N')
    ln7 = ax1.plot(years, totthrptfmN,'-', color = 'b',label = 'FM N')
    ln1 = ax1.plot(years, totshcpN,'-.', color = 'b',label = 'Sh. N')
    
    ln8 = ax1.plot(years, totthrptgpAL,'--', color = 'r',label = 'GP AL')
    ln9 = ax1.plot(years, totthrptfmAL,'-', color = 'r',label = 'FM AL')
    ln3 = ax1.plot(years, totshcpAL,'-.', color = 'r',label = 'Sh. AL')
    
    ln4 = ax2.plot(years, totthrptgpD,'--', color = 'g',label = 'GP D')
    ln5 = ax2.plot(years, totthrptfmD,'-', color = 'g',label = 'FM D')
    ln2 = ax2.plot(years, totshcpD,'-.', color = 'g',label = 'Sh. D')
    
    ax1.set_xlabel("time (years)")
    ax1.set_ylabel("total throughput N & Al (Tb/s)")
    ax2.set_ylabel("total throughput D (Tb/s)")
    
    ax1.set_xlim([years[0], years[-1]])
    ax1.set_ylim([totthrptfmAL[0] - 10, totshcpN[-1] + 10])
    ax2.set_ylim([totthrptfmAL[0] - 10, totshcpD[-1] + 10])
    
    lns = ln1+ln2+ln3+ln4+ln5+ln6+ln7+ln8+ln9
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
    #plt.axis([years[0],years[-1],1.0,8.0])
    #plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
    plt.savefig('JOCNtotalthrpt.pdf', dpi=200,bbox_inches='tight')
    plt.show()
    
    linkind1 = 2
    linkind2 = 1
    
    imodfN = [int(i) for i in modfN[linkind1]]
    imfplN = [int(i) for i in mfplN[linkind1]]
    y2lb = ['1','2']
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    ln1 = ax1.plot(years, UmN[linkind1],'--',color = 'b',label = 'GP U')
    ln2 = ax1.plot(years, UmFplN[linkind1],'--',color = 'r', label = "FM U")
    
    ln3 = ax2.plot(years, imodfN,'-',color = 'b',label = 'GP SE')
    ln4 = ax2.plot(years, imfplN,'-',color = 'r',label = 'FM SE')
    
    ax1.set_xlabel("time (years)")
    ax1.set_ylabel("U margin (dB)")
    ax2.set_ylabel("spectral efficiency (bits/sym)")
    
    ax1.set_xlim([years[0], years[-1]])
    ax1.set_ylim([-0.2, 3.2])
    ax2.set_ylim([1.9, 4.1])
    
    ax2.set_yticks([2,4])
    ax2.set_yticklabels(y2lb)
    lns = ln1+ln2+ln3+ln4
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
    #plt.axis([years[0],years[-1],1.0,8.0])
    #plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
    plt.savefig('JOCNUmargin.pdf', dpi=200,bbox_inches='tight')
    plt.show()

    
    fig, ax1 = plt.subplots()
    #ax2 = ax1.twinx()
    
    ln1 = ax1.plot(years, totUmgpN,'-', color = 'b',label = 'GP N')
    ln5 = ax1.plot(years, totUmgpAL,'-', color = 'g',label = 'GP Al')
    ln3 = ax1.plot(years, totUmgpD,'-', color = 'r',label = 'GP D')
    
    ln2 = ax1.plot(years, totUmfmN,'--', color = 'b', label = 'FM N')
    ln6 = ax1.plot(years, totUmfmAL,'--', color = 'g',label = 'FM Al')
    ln4 = ax1.plot(years, totUmfmD,'--', color = 'r',label = 'FM D')
    
    ax1.set_xlabel("time (years)")
    ax1.set_ylabel("total U margin (dB)")
    #ax2.set_ylabel("Shannon throughput gain (%)")
    
    ax1.set_xlim([years[0], years[-1]])
    ax1.set_ylim([31, 115])
    #ax2.set_ylim([totthrptdiffshD[-1]-4, totthrptdiffshN[0]+4])
    
    lns = ln1+ln2+ln3+ln4+ln5+ln6
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=2,ncol=3, prop={'size': 10})
    #plt.axis([years[0],years[-1],1.0,8.0])
    #plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
    plt.savefig('JOCNtotalUmargin.pdf', dpi=200,bbox_inches='tight')
    plt.show()

# %% 
if graphA == graphN:
    suffix = 'N'
elif graphA == graphAL:
    suffix = 'AL'
elif graphA == graphD:
    suffix = 'D'
    
# =============================================================================
# plt.plot(years, thrptgp, label = 'GP')
# plt.plot(years, thrptfm, label = 'FM')
# plt.xlabel("time (years)")
# plt.ylabel("total throughput per ch. (Tb/s)")
# plt.legend()
# #plt.axis([years[0],years[-1],1.0,8.0])
# plt.savefig('Ytotalthrptperch' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
# plt.show()
# =============================================================================

# =============================================================================
# plt.plot(years, totshcpN, label = 'Shannon')
# plt.plot(years, totthrptgpN, label = 'GP')
# plt.plot(years, totthrptfmN, label = 'FM')
# plt.xlabel("time (years)")
# plt.ylabel("total throughput (Tb/s)")
# plt.legend()
# plt.axis([years[0],years[-1],totthrptfmN[0],totshcpN[-1]])
# plt.savefig('Ytotalthrpt' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
# plt.show()
# =============================================================================

# =============================================================================
# plt.plot(years, totthrptdiff, label = 'GP - FM')
# plt.xlabel("time (years)")
# plt.ylabel("total throughput gain (%)")
# plt.legend()
# plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
# plt.show()
# 
# plt.plot(years, totthrptdiffsh, label = 'Shannon - FM')
# plt.xlabel("time (years)")
# plt.ylabel("total throughput gain (%)")
# plt.legend()
# plt.savefig('Ytotalthrptdiffsh' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
# plt.show()
# =============================================================================

# =============================================================================
# plt.plot(years, totUmgp, label = 'GP')
# plt.plot(years, totUmfm, label = 'FM')
# plt.xlabel("time (years)")
# plt.ylabel("total U margin (dB)")
# plt.legend()
# #plt.savefig('YtotalU' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
# plt.show()
# =============================================================================

# 



# %%
linkind = 2

# =============================================================================
# plt.plot(years, UmN[linkind], label = "GP")
# plt.plot(years, UmrtN[linkind], label = "RTM")
# plt.plot(years, UmFplN[linkind], label = "FM")
# 
# plt.xlabel("time (years)")
# plt.ylabel("U margin (dB)")
# plt.legend()
# # #plt.savefig('YonelinkU' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
# plt.savefig('benefitofGP.pdf', dpi=200,bbox_inches='tight')
# plt.show()
# =============================================================================

# =============================================================================
# plt.plot(years, modfN[linkind], label = "GP")
# plt.plot(years, mfplN[linkind], label = "FM")
# plt.xlabel("time (years)")
# plt.ylabel("QAM order")
# plt.legend()
# #plt.savefig('YonelinkMF' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
# plt.show()
# =============================================================================

linkind1 = 40

imodfN = [int(i) for i in modfN[linkind1]]
imodfrtN = [int(i) for i in modfrtN[linkind1]]

imfplN = [int(i) for i in mfplN[linkind1]]
y2lb = ['5','6']

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ln5 = ax1.plot(years, UmN[linkind1],'--',color = 'b',label = 'GP U')
ln4 = ax1.plot(years, UmrtN[linkind1],'--',color = 'g',label = 'RMT U')
ln6 = ax1.plot(years, UmFplN[linkind1],'--',color = 'r', label = "FM U")

ln1 = ax2.plot(years, imodfrtN,'-',color = 'g',label = 'RTM SE')
ln2 = ax2.plot(years, imodfN,'-',color = 'b',label = 'GP SE')
ln3 = ax2.plot(years, imfplN,'-',color = 'r',label = 'FM SE')

ax1.set_xlabel("time (years)")
ax1.set_ylabel("U margin (dB)")
ax2.set_ylabel("spectral efficiency (bits/sym)")

ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([-1, 4])
#ax2.set_ylim([1.9, 4.1])

ax2.set_yticks([32,64])
ax2.set_yticklabels(y2lb)
lns = ln1+ln2+ln3+ln4+ln5+ln6
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('UmarginGPbenefit.pdf', dpi=200,bbox_inches='tight')
plt.show()

# %%

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
    
ln1 = ax1.plot(years, totthrptdiffN,'-', color = 'b',label = 'GP N')
ln2 = ax1.plot(years, totthrptdiffrtN,'-', color = 'g',label = 'RTM N')

    
ln3 = ax2.plot(years, totthrptdiffshN,'--', color = 'b', label = 'Sh. N')
ln4 = ax2.plot(years, totthrptdiffrtshN,'--', color = 'b', label = 'Sh. N')

    
ax1.set_xlabel("time (years)")
ax1.set_ylabel("GP & RTM throughput gain (%)")
ax2.set_ylabel("Shannon throughput gain (%)")
    
ax1.set_xlim([years[0], years[-1]])
#ax1.set_ylim([totthrptdiffN[-1]-4, totthrptdiffN[0]+4])
#ax2.set_ylim([totthrptdiffshD[-1]-4, totthrptdiffshN[0]+4])
    
lns = ln1+ln2+ln3+ln4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0,ncol=2, prop={'size': 10})
#plt.axis([years[0],years[-1],1.0,8.0])
#plt.savefig('Ytotalthrptdiff' + str(suffix) + '.pdf', dpi=200,bbox_inches='tight')
#plt.savefig('JOCNtotalthrptdiff.pdf', dpi=200,bbox_inches='tight')
plt.savefig('thrptdiffGPbenefit.pdf', dpi=200,bbox_inches='tight')
plt.show()

