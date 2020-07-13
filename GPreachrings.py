
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

# section 1: find the shortest and second shortest paths for each node pair and start to fill network 

nodesT = ['1','2','3','4','5','6','7','8','9','10']

graphT = {'1':{'2':100,'10':200},'2':{'1':100,'3':400},'3':{'2':100,'4':200},    
         '4':{'3':100,'5':300},'5':{'4':100,'6':100},'6':{'5':100,'7':200}, '7':{'6':100,'8':200},
         '8':{'7':100,'9':100}, '9':{'8':200,'10':100}, '10':{'1':300,'9':100}
         }
edgesT = {'1':{'2':0,'10':1},'2':{'1':2,'3':3},'3':{'2':4,'4':5},    
         '4':{'3':6,'5':7},'5':{'4':8,'6':9},'6':{'5':10,'7':11}, '7':{'6':12,'8':13},
         '8':{'7':14,'9':15}, '9':{'8':16,'10':17}, '10':{'1':18,'9':19}
         }
numnodesT = 10
numedgesT = 20
LspansT = 100

graphA = graphT
if graphA == graphT:
    numnodesA = numnodesT
    numedgesA = numedgesT
    nodesA = nodesT
    LspansA = LspansT


def removekey(d, keysrc, keydes): # function for removing key from dict - used to remove blocked links 
    r = dict(d)                     # removes the link between nodes 'keysrc' and 'keydes'
    del r.get(keysrc)[keydes]
    return r    

def findroutes(nodes, secondpath):
    dis = []
    path = []
    numnodes = len(nodes)
    if graphA == graphT:
        for i in range(numnodes):
            for j in range(numnodes): 
                if i == j:
                    continue
                d, p = dijkstra({'1':{'2':100,'10':200},'2':{'1':100,'3':400},'3':{'2':100,'4':200},    
                                 '4':{'3':100,'5':300},'5':{'4':100,'6':100},'6':{'5':100,'7':200}, '7':{'6':100,'8':200},
                                 '8':{'7':100,'9':100}, '9':{'8':200,'10':100}, '10':{'1':300,'9':100}
                                 } , nodes[i], nodes[j])
                dis.append(d)
                path.append(p)
                if secondpath:
                    shgraph = removekey({'1':{'2':100,'10':200},'2':{'1':100,'3':400},'3':{'2':100,'4':200},    
                                         '4':{'3':100,'5':300},'5':{'4':100,'6':100},'6':{'5':100,'7':200}, '7':{'6':100,'8':200},
                                         '8':{'7':100,'9':100}, '9':{'8':200,'10':100}, '10':{'1':300,'9':100}
                                         }, p[0],p[1])
                    d2, p2 = dijkstra(shgraph , nodes[i], nodes[j])
                    dis.append(d2)
                    path.append(p2)
    return dis, path
        
pthdists, pths = findroutes(nodesT,True)             

# %% section 2: find the number of wavelengths on each link in the topology 
     
def getlinklen(shpath,graph,edges):  # takes nodes traversed as input and returns the lengths of each link and the edge indices 
    linklen = np.empty([len(shpath)-1,1])
    link = []
    for i in range(len(shpath)-1):
        linklen[i] = float((graph.get(shpath[i])).get(shpath[i+1]))
        link.append((edges.get(shpath[i])).get(shpath[i+1]))
    return linklen, link                

edgeinds = []
numlamlk = np.zeros([numedgesT,1])
for i in range(len(pths)):
    edgeinds.append(getlinklen(pths[i], graphT, edgesT)[1])  # transparent network: only need total distance for each path 
    numlamlk[edgeinds[i]] = numlamlk[edgeinds[i]] + 1
    #print(edgeinds[i])

# %% section 3: ageing effects and margin calculation 

PchdBm = np.linspace(-6,6,500)  # 500 datapoints for higher resolution of Pch
TRxb2b = 26 # fron UCL paper: On the limits of digital back-propagation in the presence of transceiver noise, Lidia Galdino et al.
numpoints = 100

alpha = 0.2
NLco = 1.27
Disp = 16.7

OSNRmeasBW = 12.478 # OSNR measurement BW [GHz]
Rs = 32 # Symbol rate [Gbd]
testlen = 1000.0     # all ageing effects modelled using values in: Faster return of investment in WDM networks when elastic transponders dynamically fit ageing of link margins, Pesic et al.
years = np.linspace(0,10,21) # define how many years are in the lifetime of the network and the resolution 
numyears = np.size(years)
sd = np.linspace(0.04, 0.08, np.size(years)) # added SNR uncertainty SD - assumed to double over lifetime

NF = np.linspace(4.5,5.5,np.size(years)) # define the NF ageing of the amplifiers 
alpha = 0.2 + 0.00163669*years # define the fibre ageing due to splice losses over time 
trxaging = ((1 + 0.05*years)*2).reshape(np.size(years),1)*(OSNRmeasBW/Rs) # define TRx ageing 
oxcaging = ((0.03 + 0.007*years)*2).reshape(np.size(years),1)*(OSNRmeasBW/Rs) # define filter ageing, assuming two filters per link, one at Tx and one at Rx

# find the worst-case margin required                         
fmD = sd[-1]*5 # static D margin is defined as 5xEoL SNR uncertainty SD that is added
fmDGI = sd[0]*5 # inital D margin used for GP approach (before LP establishment)

# %% section 4: find the SNR over each path, accounting for the varying number of wavelengths on each link 

def SNRgen(pathind, yearind, nyqch):  # function for generating a new SNR value to test if uncertainty is dealt with
        Ls = LspansA
        D = Disp          # rather than the worst-case number 
        gam = NLco
        lam = 1550 # operating wavelength centre [nm]
        f = 299792458/(lam*1e-9) # operating frequency [Hz]
        c = 299792.458 # speed of light in vacuum [nm/ps] -> needed for calculation of beta2
        Rs = 32 # symbol rate [GBaud]
        h = 6.63*1e-34  # Planck's constant [Js]
        edgelen = pthdists[pathind]
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
        # ===================== find Popt for one span ==========================
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
        # =======================================================================
        numspans = int(edgelen/Ls)
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

testsnrgen = SNRgen(4, 0, False)










   