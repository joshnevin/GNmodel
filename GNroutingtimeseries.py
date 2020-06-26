
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

datagen = False
GPtraining = False
numpoints = 100


alpha = 0.2
NLco = 1.27
NchRS = 101
Disp = 16.7


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
# choose 'active' topology 
graphA = graphAL
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

# %% GNPy test function 
PchdBm = np.linspace(-6,6,500)  # 500 datapoints for higher resolution of Pch
TRxb2b = 26 # fron UCL paper: On the limits of digital back-propagation in the presence of transceiver noise, Lidia Galdino et al.
# =============================================================================
# def GNtest(edgelen, Lspans, numlam, NF, alpha):
#     #################### equipment characteristics ####################
#     lam = 1550 # operating wavelength centre [nm]
#     f = 299792458/(lam*1e-9) # operating frequency [Hz]
#     c = 299792.458 # speed of light in vacuum [nm/ps] -> needed for calculation of beta2
#     Rs = 32 # symbol rate [GBaud]
#     h = 6.63*1e-34  # Planck's constant [Js]
#     NchNy = 157 # 157 Nyquist channels = full C-band
#     BWNy = (NchNy*Rs)/1e3 # keep Nyquist spacing for non-saturated channel
#     #BWNy = (157*Rs)/1e3 # full 5THz BW
#     Ns = int(edgelen/Lspans)
#     al = alpha
#     D = Disp
#     gam = NLco
#     Ls = Lspans
#     allin = np.log((10**(al/10)))/2 # fibre loss [1/km] -> weird definition, due to exponential decay of electric field instead of power, which is standard 
#     #gam = 1.27 # fibre nonlinearity coefficient [1/W*km]
#     beta2 = (D*(lam**2))/(2*np.pi*c) # dispersion coefficient at given wavelength [ps^2/km]
#     Leff = (1 - np.exp(-2*allin*Ls ))/(2*allin)  # effective length [km]      
#     Leffa = 1/(2*allin)  # the asymptotic effective length [km]
#     LF = 1 # filtering effects coefficient, takes values 0 < LF <= 1 
#     Pch = 1e-3*10**(PchdBm/10)  # ^ [W]
#     Gwdm = (Pch*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
#     
#     ## equation 13, Gnli(0) for single-span Nyquist-WDM 
#     GnliEq13 = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))
#     numpch = len(PchdBm)
#     # ASE noise bit 
#     G = al*Ls
#     NFl = 10**(NF/10) 
#     Gl = 10**(G/10) 
#     OSNRmeasBW = 12.478*1e9 # OSNR measurement BW [Hz] - note that 0.1nm BW is used in GNPy measurements of OSNR
#     Pasech = NFl*h*f*(Gl - 1)*Rs*1e9*Ns # [W] the ASE noise power in one Nyquist channel across all spans
#     Pasech = NFl*h*f*(Gl - 1)*OSNRmeasBW*Ns
#     # SNR calc + plotting
#     SNRanalytical = 10*np.log10((LF*Pch)/(Pasech*np.ones(numpch) + GnliEq13*Ns*Rs*1e9))
#     SNRanalytical = 10*np.log10((LF*Pch)/(Pasech*np.ones(numpch) + GnliEq13*Ns*OSNRmeasBW))
#     Popt = PchdBm[np.argmax(SNRanalytical)]
#     
#     Pun = GNmain(Lspans, 1, numlam, 101, 201, alpha, Disp, PchdBm, NF, NLco,False,numpoints)[0] 
#     Poptgnmain = PchdBm[np.argmax(Pun)]
#     
#     return SNRanalytical, Popt, Poptgnmain
# 
# testGNsnr, testGNPopt, testgnmainpopt = GNtest(300.0, 100.0, 157, 5.0, 0.2)
# testGNPysnr = np.genfromtxt(open("SNRGNPy.csv", "r"), delimiter=",", dtype =float)
# plt.plot(PchdBm, testGNsnr, label="me")
# #plt.plot(PchdBm, testGNPysnr, label="GNPy")
# #plt.plot(PchdBm, testGNmainsnr, label="GNmain")
# plt.legend()
# plt.xlabel("Pch (dBm)")
# plt.ylabel("SNR (dB)")
# plt.savefig('ZGNPycomparison.pdf', dpi=200,bbox_inches='tight')
# plt.show()
# =============================================================================

def marginsnrtest(edgelen, Lspans, numlam, NF, alpha):
    lam = 1550 # operating wavelength centre [nm]
    f = 299792458/(lam*1e-9) # operating frequency [Hz]
    c = 299792.458 # speed of light in vacuum [nm/ps] -> needed for calculation of beta2
    Rs = 32 # symbol rate [GBaud]
    h = 6.63*1e-34  # Planck's constant [Js]
    NchNy = 157
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

testlen = 4800.0     # all ageing effects modelled using values in: Faster return of investment in WDM networks when elastic transponders dynamically fit ageing of link margins, Pesic et al.
years = np.linspace(0,10,21) # define how many years are in the lifetime of the network and the resolution 
numyears = np.size(years)
sd = np.linspace(0.04, 0.08, np.size(years)) # added SNR uncertainty SD - assumed to double over lifetime
numlam = np.linspace(30, 150, np.size(years)) # define the channel loading over the network lifetime 
NF = np.linspace(4.5,5.5,np.size(years)) # define the NF ageing of the amplifiers 
alpha = 0.2 + 0.00163669*years # define the fibre ageing due to splice losses over time 
trxaging = ((1 + 0.05*years)*2).reshape(np.size(years),1) # define TRx ageing 
oxcaging = ((0.03 + 0.007*years)*2).reshape(np.size(years),1) # define filter ageing, assuming two filters per link, one at Tx and one at Rx
# =============================================================================
# testsnrnli = np.empty([np.size(years),1])
# nliampmargin = np.empty([np.size(years),1])
# testsnrjustfibre = np.empty([np.size(years),1])
# for i in range(np.size(years)):
#     testsnrnli[i] = marginsnrtest(testlen, LspansA, numlam[i], NF[i], alpha[i])
# testfinalsnr = ( testsnrnli - (trxaging + oxcaging)) 
# plt.plot(years, testsnrnli, label = 'SNR with NF, NLI + FA')
# plt.plot(years, testfinalsnr, label = 'SNR - TRx + filtering')
# plt.xlabel("years")
# plt.ylabel("SNR (dB)")
# plt.legend()
# plt.savefig('marginvarsnr.pdf', dpi=200,bbox_inches='tight')
# plt.show()
# =============================================================================
# find the worst-case margin required
fmD = np.empty([numedgesA,1])
for i in range(numedgesA):
    fmD[i] = sd[-1]*(edgelensA[i]/1000.0)*5 # D margin is defined as 5xEoL SNR uncertainty SD that is added
fmT = trxaging[-1] + oxcaging[-1] + fmD # total static margin: amplifier ageing, NL loading and fibre ageing included in GN
# %%
if datagen:
    def datats(edgelen, Lspans, numlam, NF,sd, alpha, yearind):
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
        snr = (Pch/(Pase + Gnli*Rs*1e9)) - db2lin(trxaging[yearind] + oxcaging[yearind]) # subtract static ageing effects
        snr = ( snr**(-1) + (db2lin(TRxb2b))**(-1) )**(-1) # add TRx B2B noise 
        #snr = snr + np.random.normal(0,db2lin(sd),numpoints)
        sdnorm = sd*(edgelen/1000.0) # noise on each link is assumed to be proportional to the link length 
        return lin2db(snr) + np.random.normal(0,sdnorm,numpoints), Popt 
            #return lin2db( ( ((Pch)/(Pase*np.ones(numpoints) + Gnli*Rs*1e9)  )**(-1)  +  np.random.normal(db2lin(26), db2lin(TRxsd), numpoints)**(-1) )**(-1) ) , Popt, Poptind
        
    def savedatgauss(edgelens, numedges,Lspans, yearind):    
        linkSNR = np.empty([numedges,numpoints])
        for i in range(numedges):
            linkSNR[i], linkPopt = datats(edgelens[i],Lspans,numlam[yearind], NF[yearind], sd[yearind], alpha[yearind], yearind)
        TRxSNR = 26 # add TRx noise of 26dB B2B 
        linkSNR = lin2db( 1/(  1/(db2lin(linkSNR)) + 1/(db2lin(TRxSNR))  ))
        if graphA == graphN:
            np.savetxt('tsSNRN' + str(int(yearind)) + '.csv', linkSNR, delimiter=',') 
            linkPopt = linkPopt.reshape(1,1)
            np.savetxt('tsPoptN' + str(int(yearind)) + '.csv', linkPopt, delimiter=',') 
        elif graphA == graphD:
            np.savetxt('tsSNRD' + str(int(yearind)) + '.csv', linkSNR, delimiter=',') 
            linkPopt = linkPopt.reshape(1,1)
            np.savetxt('tsPoptD' + str(int(yearind)) + '.csv', linkPopt, delimiter=',') 
        elif graphA == graphAL:
            np.savetxt('tsSNRAL' + str(int(yearind)) + '.csv', linkSNR, delimiter=',')
            linkPopt = linkPopt.reshape(1,1)
            np.savetxt('tsPoptAL' + str(int(yearind)) + '.csv', linkPopt, delimiter=',') 
        return linkSNR, linkPopt
    
    linkSNR = np.empty([numyears,numedgesA,numpoints])
    linkPopt = np.empty([numyears,1])
    for i in range(numyears):
        linkSNR[i], linkPopt[i] = savedatgauss(edgelensA, numedgesA, LspansA, i)
if datagen == False:
    def importdat(Lspans, numedges, yearind):
            if graphA == graphN:
                linkSNR = np.genfromtxt(open('tsSNRN' + str(int(yearind)) + '.csv', "r"), delimiter=",", dtype =float)
                linkPopt = np.genfromtxt(open('tsPoptN' + str(int(yearind)) + '.csv', "r"), delimiter=",", dtype =float)
            elif graphA == graphD:
                linkSNR = np.genfromtxt(open('tsSNRD' + str(int(yearind)) + '.csv', "r"), delimiter=",", dtype =float)
                linkPopt = np.genfromtxt(open('tsPoptD' + str(int(yearind)) + '.csv', "r"), delimiter=",", dtype =float)
            elif graphA == graphAL:
                linkSNR = np.genfromtxt(open('tsSNRAL' + str(int(yearind)) + '.csv', "r"), delimiter=",", dtype =float)
                linkPopt = np.genfromtxt(open('tsPoptAL' + str(int(yearind)) + '.csv', "r"), delimiter=",", dtype =float)
            return linkSNR,  linkPopt
    linkSNR = np.empty([numyears,numedgesA,numpoints])
    linkPopt = np.empty([numyears,1])
    for i in range(numyears):
        linkSNR[i], linkPopt[i] = importdat(LspansA,numedgesA, i)
# %% data generation for long-term heteroscedastic GP
hetdatagen = False
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
            sdnorm = sd*(edgelen/1000.0) # noise on each link is assumed to be proportional to the link length 
            return lin2db(snr) + np.random.normal(0,sdnorm,1), Popt 
        
    def hetsave(edgelens, numedges,Lspans, yearind):    
            linkSNR = np.empty([numedges,1])
            for i in range(numedges):
                linkSNR[i], linkPopt = datatshet(edgelens[i],Lspans,numlamh[yearind], NFh[yearind], sdh[yearind], alphah[yearind], yearind)
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
            return linkSNR, linkPopt
    numphet = 5  
    numyrsh = 100
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
        hetdata[i], _ = hetsave(edgelensA, numedgesA, LspansA, i)
    hetdata = np.transpose(hetdata)
    if graphA == graphN:
        np.savetxt('hetdataN.csv', hetdata, delimiter=',') 
    elif graphA == graphD:
        np.savetxt('hetdataD.csv', hetdata, delimiter=',') 
    elif graphA == graphAL:
        np.savetxt('hetdataAL.csv', hetdata, delimiter=',') 
# %% plotting Popt vs time
# =============================================================================
# x = np.linspace(0,numpoints-1,numpoints)
# plt.plot(x,linkSNR[-1][0],'+')
# plt.show()
# 
# plt.plot(years,linkPopt)
# plt.xlabel("years")
# plt.ylabel("Popt (dBm)")
# plt.savefig('yearspopt.pdf', dpi=200,bbox_inches='tight')
# plt.show()
# =============================================================================
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

if GPtraining:
    x = np.linspace(0,numpoints-1,numpoints)
    prmn = np.empty([numyears,numedgesA])
    sigma = np.empty([numyears,numedgesA])
    for i in range(numyears):
        y = linkSNR[i]
        prmnt = np.empty([np.size(y,0),np.size(y,1)])
        sigmat = np.empty([np.size(y,0),1])
        start = time.time()
        for j in range(numedgesA):
            #print("link " + str(j))
            prmns, sigmat[j] = GPtrain(x,y[j])
            prmnt[j] = prmns.reshape(numpoints)
        end = time.time()
        print("GP training took " + str(end-start))   
        if graphA == graphN:   
            np.savetxt('prmntsN' + str(i) + '.csv', prmnt, delimiter=',') 
            np.savetxt('sigtsN' + str(i) + '.csv', sigmat, delimiter=',')
        elif graphA == graphD:
            np.savetxt('prmntsD' + str(i) + '.csv', prmnt, delimiter=',') 
            np.savetxt('sigtsD' + str(i) + '.csv', sigmat, delimiter=',')
        elif graphA == graphAL:
            np.savetxt('prmntsAL' + str(i) + '.csv', prmnt, delimiter=',') 
            np.savetxt('sigtsAL' + str(i) + '.csv', sigmat, delimiter=',')
        for j in range(numedgesA):
            prmn[i][j] = np.mean(prmnt[j]) # average over the predictive mean to smooth out any variation - doing this might be dodgy
        sigma[i] = sigmat.reshape(numedgesA)
# %% import trained GP models
if GPtraining == False:  
    sigma =  np.empty([numyears,numedgesA])
    prmn = np.empty([numyears,numedgesA])
    for i in range(numyears):
        if graphA == graphN:
            prmnt = np.genfromtxt(open('prmntsN' + str(i) + '.csv', "r"), delimiter=",", dtype =float)
            sigmat = np.genfromtxt(open('sigtsN' + str(i) + '.csv', "r"), delimiter=",", dtype =float)
        elif graphA == graphD:
            prmnt = np.genfromtxt(open('prmntsD' + str(i) + '.csv', "r"), delimiter=",", dtype =float)
            sigmat = np.genfromtxt(open('sigtsD' + str(i) + '.csv', "r"), delimiter=",", dtype =float)
        elif graphA == graphAL:
            prmnt = np.genfromtxt(open('prmntsAL' + str(i) + '.csv', "r"), delimiter=",", dtype =float)
            sigmat = np.genfromtxt(open('sigtsAL' + str(i) + '.csv', "r"), delimiter=",", dtype =float)
        for j in range(numedgesA):
            prmn[i][j] = np.mean(prmnt[j])
        sigma[i] = sigmat 




# %% 
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
    
def fmdatagen(edgelens,Lspans, yearind):    # this generates a GN model SNR value for each year, but currently only using
    fmSNR = np.empty([np.size(edgelens),1]) # the EoL values - the margin for loading, NF ageing and fibre ageing is included here
    for i in range(np.size(edgelens)):
        fmSNR[i] = fmsnr(edgelens[i],Lspans, numlam[yearind], NF[yearind], alpha[yearind], yearind)
    return fmSNR
fmSNR = np.empty([numyears, numedgesA, 1])
for i in range(numyears):
    fmSNR[i] = fmdatagen(edgelensA,LspansA,i)
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
# %%    
def getlinklen(shpath,graph,edges):  # function used in routing 
        linklen = np.empty([len(shpath)-1,1])
        link = []
        for i in range(len(shpath)-1):
            linklen[i] = float((graph.get(shpath[i])).get(shpath[i+1]))
            link.append((edges.get(shpath[i])).get(shpath[i+1]))
        return linklen, link   
def requestgen(graph): # function used in routing
            src = random.choice(list(graph.keys()))
            des = random.choice(list(graph.keys()))
            while des == src:
                des = random.choice(list(graph.keys()))
            return src, des
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
        sdnorm = sd[yearind]*(edgelen/1000.0)
        return lin2db(snr) + np.random.normal(0,sdnorm,1)
# define the FEC thresholds - all correspond to BER of 2e-2 (2% FEC)
FT2 = 0.24 
FT4 = 3.25 
FT16 = 12.72
FT64 = 18.43
FT128 = 22.35
# %% fixed margin routing algorithm - no second shortest path routing applied 
def fmrta(graph, edges, Rsource, Rdest, showres,nodes,numedges,edgelens,fmSNR, Lspans, yearind):
    dis = []
    path = []
    numnodes = np.size(nodes)
    if graphA == graphN:
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
    elif graphA == graphD:
        for i in range(numnodes):    
            for j in range(numnodes): 
                d, p = dijkstra({'1':{'2':400,'3':160,'4':160},'2':{'1':400,'4':400,'5':240},'3':{'1':160,'4':160,'6':320},    
                '4':{'1':160,'2':400,'3':160,'5':320,'7':240,'10':400},'5':{'2':240,'4':320,'10':480,'11':320}, '6':{'3':320,'7':80,'8':80},
                '7':{'4':240,'6':80,'9':80}, '8':{'6':80,'9':80}, '9':{'7':80,'8':80,'10':240},
                '10':{'4':400,'5':480,'9':240,'11':320,'12':240}, '11':{'5':320,'10':320,'12':240,'14':240}, '12':{'10':240,'11':240,'13':80},
                '13':{'12':80,'14':160}, '14':{'11':240,'13':160}
                } , nodes[i], nodes[j])
                if i == j:
                    continue  # don't include lightpaths of length 0
                else:
                    dis.append(d)
                    path.append(p)
    elif graphA == graphAL:
        for i in range(numnodes):    
            for j in range(numnodes): 
                d, p = dijkstra({'1':{'4':1200,'5':1600},'2':{'3':1100,'7':300},'3':{'2':1100,'8':300},    
                '4':{'1':1200,'5':1500,'9':500},'5':{'1':1600,'4':1500,'6':900}, '6':{'5':900,'7':700,'11':1000},
                '7':{'2':300,'6':700,'10':1100}, '8':{'3':300,'10':900}, '9':{'4':500,'11':2200},
                '10':{'7':1100,'8':900,'11':1100}, '11':{'6':1000,'9':2200,'10':1100}
                }  , nodes[i], nodes[j])
                if i == j:
                    continue  # don't include lightpaths of length 0
                else:
                    dis.append(d)
                    path.append(p)
    pathdists = []
    links = []                                                
    for i in range(np.size(path)):
        pathdists.append(getlinklen(path[i],graphnormA,edges)[0])
        links.append(getlinklen(path[i],graph,edges)[1])
    estlam = np.zeros([numedges,int(numlam[yearind])]) # 0 for empty, 1 for occupied
    reqlams = 0
    conten = 0
    failures = 0 # post-reach determination failures
    noreach = 0 # cases in which algorithm unable to assign any modulation 
    ct128 = 0
    ct64 = 0
    ct16 = 0
    ct4 = 0
    ct2 = 0
    Um = 0 # unallocated margin
    numreq = np.size(Rsource)
    randdist = []
    for i in range(numreq):
        # update for online learning 
        #  choose random source and destination nodes 
    
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
        edgesuc2 = 0
        FT = np.zeros(np.size(linkSNR[0][randedges],0))
        #print(np.size(linkSNR[randedges],0))
        #print(np.size(randedges))
        for j in range(np.size(linkSNR[0][randedges],0)):
            
            if fmSNR[-1][randedges][j] - fmT[randedges][j] > FT128: # use the worst-case fixed margin SNR estimate
                FT[j] = FT128
                ct128 = ct128 + 1
                edgesuc = edgesuc + 1
            elif fmSNR[-1][randedges][j] - fmT[randedges][j] > FT64:
                FT[j] = FT64
                ct64 = ct64 + 1
                edgesuc = edgesuc + 1
            elif fmSNR[-1][randedges][j] - fmT[randedges][j] > FT16:
                FT[j] = FT16
                ct16 = ct16 + 1
                edgesuc = edgesuc + 1
            elif fmSNR[-1][randedges][j] - fmT[randedges][j] > FT4:
                FT[j] = FT4
                ct4 = ct4 + 1
                edgesuc = edgesuc + 1
            elif fmSNR[-1][randedges][j] - fmT[randedges][j] > FT2:
                FT[j] = FT2
                ct2 = ct2 + 1
                edgesuc = edgesuc + 1
            else:
                break    
        if edgesuc == np.size(linkSNR[0][randedges],0):
            # generate new SNR value here
            for w in range(np.size(randedges)):
                # retrieve estlam row corresponding to randegdes[w] index, find number of 1s, pass to SNRnew()
                estSNR = SNRnew(edgelens[randedges[w]], np.count_nonzero(estlam[randedges[w]])+1, yearind)
                if estSNR > FT[w]:
                    # link successfully established
                    Um = Um + fmSNR[-1][randedges[w]] - fmT[randedges[w]] - FT[w]                                       
                    edgesuc2 = edgesuc2 + 1
            if edgesuc2 == np.size(randedges):
                # path successfully established
                reqlams = reqlams + 1
                for l in range(len(randedges)):
                    estlam[randedges[l]][lamslot[l]] = 1
            else: 
                # link not established
                failures = failures + 1
        else: 
            noreach = noreach + 1
    ava = (reqlams/numreq)*100 
    tottime = (sum(randdist)*1e3*1.468)/299792458
    if showres:
        print("Normal availability = " + str(ava) + "%") 
        print("Normal total traversal time = " + str('%.2f' % tottime) + "s")
        print("Normal number of failures = " + str(failures))
    return ava, estlam, reqlams, conten,ct128, ct64, ct16, ct4,ct2, failures, noreach, Um
# =============================================================================
# testsrc = []
# testdes = []
# for _ in range(2000):
#     rsctest, rdstest = requestgen(graphA)
#     testsrc.append(rsctest)
#     testdes.append(rdstest)   
# test1, _, _, _, _, _, _, _, _, test10, _, _ = fmrta(graphA, edgesA, testsrc, testdes, False,nodesA,numedgesA,edgelensA,fmSNR, LspansA, -1)
# =============================================================================
# %%
def removekey(d, keysrc, keydes): # function for removing key from dict - used to remove blocked links 
    r = dict(d)
    del r.get(keysrc)[keydes]
    return r 
def fmrta2(graph, edges, Rsource, Rdest, showres,nodes,numedges,edgelens,fmSNR, Lspans, yearind):
    dis = []
    path = []
    numnodes = len(nodes)
    if graphA == graphN:
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
    elif graphA == graphD:
        for i in range(numnodes):    
            for j in range(numnodes): 
                d, p = dijkstra({'1':{'2':400,'3':160,'4':160},'2':{'1':400,'4':400,'5':240},'3':{'1':160,'4':160,'6':320},    
                '4':{'1':160,'2':400,'3':160,'5':320,'7':240,'10':400},'5':{'2':240,'4':320,'10':480,'11':320}, '6':{'3':320,'7':80,'8':80},
                '7':{'4':240,'6':80,'9':80}, '8':{'6':80,'9':80}, '9':{'7':80,'8':80,'10':240},
                '10':{'4':400,'5':480,'9':240,'11':320,'12':240}, '11':{'5':320,'10':320,'12':240,'14':240}, '12':{'10':240,'11':240,'13':80},
                '13':{'12':80,'14':160}, '14':{'11':240,'13':160}
                } , nodes[i], nodes[j])
                if i == j:
                    continue  # don't include lightpaths of length 0
                else:
                    dis.append(d)
                    path.append(p)
    elif graphA == graphAL:
        for i in range(numnodes):    
            for j in range(numnodes): 
                d, p = dijkstra({'1':{'4':1200,'5':1600},'2':{'3':1100,'7':300},'3':{'2':1100,'8':300},    
                '4':{'1':1200,'5':1500,'9':500},'5':{'1':1600,'4':1500,'6':900}, '6':{'5':900,'7':700,'11':1000},
                '7':{'2':300,'6':700,'10':1100}, '8':{'3':300,'10':900}, '9':{'4':500,'11':2200},
                '10':{'7':1100,'8':900,'11':1100}, '11':{'6':1000,'9':2200,'10':1100}
                }  , nodes[i], nodes[j])
                if i == j:
                    continue  # don't include lightpaths of length 0
                else:
                    dis.append(d)
                    path.append(p)
    pathdists = []
    links = []                                                
    for i in range(len(path)):
        pathdists.append(getlinklen(path[i],graphnormA,edges)[0])
        links.append(getlinklen(path[i],graphA,edges)[1])
    estlam = np.zeros([numedges,int(numlam[yearind])]) # 0 for empty, 1 for occupied
    reqlams = 0
    conten = 0
    failures = 0 # post-reach determination failures
    noreach = 0 # cases in which algorithm unable to assign any modulation 
    ct128 = 0
    ct64 = 0
    ct16 = 0
    ct4 = 0
    ct2 = 0
    Um = 0 # unallocated margin
    numreq = len(Rsource)
    for i in range(numreq):
        # update for online learning 
        #  choose random source and destination nodes 
    
        # find corresponding path index
        def srcdestcheck(path,src,dest):
            if path[0] == src and path[-1] == dest:
                return True  
        
        randpathind = [j for j in range(len(path)) if  srcdestcheck(path[j], Rsource[i], Rdest[i])][0]
        #print("selected request index: " + str(randpathind))
        randedges = links[randpathind]  # selected edges for request 
        shptcon = False
        lamconten = False
        testlamslot = [np.where(estlam[randedges[k]]==0) for k in range(len(randedges))]
        for g in range(np.size(randedges)):
            if np.size(testlamslot[g]) == 0:  # if randedges[g] is blocked 
                if shptcon:
                    break
                target = randedges[g]             
                for y in nodesA: # find source and destination nodes for link randedges[g]
                    if target in [list(edgesA.get(y).values())][0]:
                        srcnode = y
                        desnode = list(edgesA.get(y).keys())[list(edgesA.get(y).values()).index(target)]
                        break 
                if graphA == graphN:
                    graphA2 = removekey({'1':{'2':2100,'3':3000,'8':4800},'2':{'1':2100,'3':1200,'4':1500},'3':{'1':3000,'2':1200,'6':3600},    
                     '4':{'2':1500,'5':1200,'11':3900},'5':{'4':1200,'6':2400,'7':1200}, '6':{'3':3600,'5':2400,'10':2100,'14':3600},
                     '7':{'5':1200,'8':1500,'10':2700}, '8':{'1':4800,'7':1500,'9':1500}, '9':{'8':1500,'10':1500,'12':600,'13':600},
                     '10':{'6':2100,'7':2700,'9':1500}, '11':{'4':3900,'12':1200,'13':1500}, '12':{'9':600,'11':1200,'14':600},
                     '13':{'9':600,'11':1500,'14':300}, '14':{'6':3600,'12':600,'13':300}
                     }, srcnode, desnode) 
                elif graphA == graphD:
                    graphA2 = removekey({'1':{'2':400,'3':160,'4':160},'2':{'1':400,'4':400,'5':240},'3':{'1':160,'4':160,'6':320},    
                     '4':{'1':160,'2':400,'3':160,'5':320,'7':240,'10':400},'5':{'2':240,'4':320,'10':480,'11':320}, '6':{'3':320,'7':80,'8':80},
                     '7':{'4':240,'6':80,'9':80}, '8':{'6':80,'9':80}, '9':{'7':80,'8':80,'10':240},
                     '10':{'4':400,'5':480,'9':240,'11':320,'12':240}, '11':{'5':320,'10':320,'12':240,'14':240}, '12':{'10':240,'11':240,'13':80},
                     '13':{'12':80,'14':160}, '14':{'11':240,'13':160}
                     }, srcnode, desnode)  # remove this link from the graph and find shortest path with Dijkstra        
                elif graphA == graphAL:
                    graphA2 = removekey({'1':{'4':1200,'5':1600},'2':{'3':1100,'7':300},'3':{'2':1100,'8':300},    
                    '4':{'1':1200,'5':1500,'9':500},'5':{'1':1600,'4':1500,'6':900}, '6':{'5':900,'7':700,'11':1000},
                    '7':{'2':300,'6':700,'10':1100}, '8':{'3':300,'10':900}, '9':{'4':500,'11':2200},
                    '10':{'7':1100,'8':900,'11':1100}, '11':{'6':1000,'9':2200,'10':1100}
                    }, srcnode, desnode)  # remove this link from the graph and find shortest path with Dijkstra 
                dist2, path2 = dijkstra(graphA2, Rsource[i], Rdest[i])
                link2 = (getlinklen(path2,graphA,edges)[1]) # converts nodes traversed to edges traversed = 'new randedges'
                shptcon = True
        if shptcon:
            testlamslot2 = [np.where(estlam[link2[p]]==0) for p in range(np.size(link2))]
            for h in range(np.size(link2)):
                if np.size(testlamslot2[h]) == 0: # second-shortest also blocked    
                    lamconten = True
                else: # if second-shortest not blocked, replace randedges with link2
                    randedges = link2
        if lamconten:
            conten = conten + 1
            continue # deny request
        lamslot = [np.where(estlam[randedges[k]]==0)[0][0] for k in range(np.size(randedges))] # first available wavelength slot for this edge
        # need to check SNR for all the edges in the path
        edgesuc = 0
        edgesuc2 = 0
        FT = np.zeros(np.size(linkSNR[-1][randedges],0))
        #print(np.size(linkSNR[randedges],0))
        #print(np.size(randedges))
        for j in range(np.size(linkSNR[-1][randedges],0)):
            if fmSNR[-1][randedges][j] - fmT[randedges][j] > FT128:
                FT[j] = FT128
                ct128 = ct128 + 1
                edgesuc = edgesuc + 1
            elif fmSNR[-1][randedges][j] - fmT[randedges][j] > FT64:
                FT[j] = FT64
                ct64 = ct64 + 1
                edgesuc = edgesuc + 1
            elif fmSNR[-1][randedges][j] - fmT[randedges][j] > FT16:
                FT[j] = FT16
                ct16 = ct16 + 1
                edgesuc = edgesuc + 1
            elif fmSNR[-1][randedges][j] - fmT[randedges][j] > FT4:
                FT[j] = FT4
                ct4 = ct4 + 1
                edgesuc = edgesuc + 1
            elif fmSNR[-1][randedges][j] - fmT[randedges][j] > FT2:
                FT[j] = FT2
                ct2 = ct2 + 1
                edgesuc = edgesuc + 1
            else:
                break   
        if edgesuc == np.size(linkSNR[-1][randedges],0):
            # generate new SNR value here
            for w in range(np.size(randedges)):
                # retrieve estlam row corresponding to randegdes[w] index, find number of 1s, pass to SNRnew()
                estSNR = SNRnew(edgelens[randedges[w]], np.count_nonzero(estlam[randedges[w]])+1, yearind)
                if estSNR > FT[w]:
                    # link successfully established
                    Um = Um + fmSNR[-1][randedges[w]] - fmT[randedges][j] - FT[w]                                       
                    edgesuc2 = edgesuc2 + 1
            if edgesuc2 == np.size(randedges):
                # path successfully established
                reqlams = reqlams + 1
                for l in range(len(randedges)):
                    estlam[randedges[l]][lamslot[l]] = 1
            else: 
                # link not established
                failures = failures + 1
        else: 
            noreach = noreach + 1   
    ava = (reqlams/numreq)*100 
    return ava, estlam, reqlams, conten,ct128, ct64, ct16, ct4,ct2, failures, noreach, Um
#test21, test22, test23, test24, test25, test26, test27, test28, test29, test210, test211, test212 = fmrta2(graphA, edgesA, testsrc, testdes, False,nodesA,numedgesA,edgelensA,fmSNR, LspansA, 0)
# %%
#cProfile.run('fmrta2(graphA, edgesA, testsrc, testdes, False,nodesA,numedgesA,edgelensA,fmSNR, LspansA, 0)')
# %%
def varrtap(graph,edges,Rsource,Rdest,showres,numsig,nodes,numedges,edgelens,Lspans,yearind):
    dis = []
    path = []
    numnodes = np.size(nodes)
    if graphA == graphN:
        for i in range(numnodes):    
            for j in range(numnodes): 
                d, p = dijkstra({'1':{'2':sigma[yearind][0],'3':sigma[yearind][1],'8':sigma[yearind][2]},'2':{'1':sigma[yearind][3],'3':sigma[yearind][4],'4':sigma[yearind][5]},'3':{'1':sigma[yearind][6],'2':sigma[yearind][7],'6':sigma[yearind][8]},    
                 '4':{'2':sigma[yearind][9],'5':sigma[yearind][10],'11':sigma[yearind][11]},'5':{'4':sigma[yearind][12],'6':sigma[yearind][13],'7':sigma[yearind][14]}, '6':{'3':sigma[yearind][15],'5':sigma[yearind][16],'10':sigma[yearind][17],'14':sigma[yearind][18]},
                 '7':{'5':sigma[yearind][19],'8':sigma[yearind][20],'10':sigma[yearind][21]}, '8':{'1':sigma[yearind][22],'7':sigma[yearind][23],'9':sigma[yearind][24]}, '9':{'8':sigma[yearind][25],'10':sigma[yearind][26],'12':sigma[yearind][27],'13':sigma[yearind][28]},
                 '10':{'6':sigma[yearind][29],'7':sigma[yearind][30],'9':sigma[yearind][31]}, '11':{'4':sigma[yearind][32],'12':sigma[yearind][33],'13':sigma[yearind][34]}, '12':{'9':sigma[yearind][35],'11':sigma[yearind][36],'14':sigma[yearind][37]},
                 '13':{'9':sigma[yearind][38],'11':sigma[yearind][39],'14':sigma[yearind][40]}, '14':{'6':sigma[yearind][41],'12':sigma[yearind][42],'13':sigma[yearind][43]}
                 }    , nodes[i], nodes[j])
                if i == j:
                    continue  # don't include lightpaths of length 0
                else:
                    dis.append(d)
                    path.append(p)
        graphvar = {'1':{'2':sigma[yearind][0],'3':sigma[yearind][1],'8':sigma[yearind][2]},'2':{'1':sigma[yearind][3],'3':sigma[yearind][4],'4':sigma[yearind][5]},'3':{'1':sigma[yearind][6],'2':sigma[yearind][7],'6':sigma[yearind][8]},    
                 '4':{'2':sigma[yearind][9],'5':sigma[yearind][10],'11':sigma[yearind][11]},'5':{'4':sigma[yearind][12],'6':sigma[yearind][13],'7':sigma[yearind][14]}, '6':{'3':sigma[yearind][15],'5':sigma[yearind][16],'10':sigma[yearind][17],'14':sigma[yearind][18]},
                 '7':{'5':sigma[yearind][19],'8':sigma[yearind][20],'10':sigma[yearind][21]}, '8':{'1':sigma[yearind][22],'7':sigma[yearind][23],'9':sigma[yearind][24]}, '9':{'8':sigma[yearind][25],'10':sigma[yearind][26],'12':sigma[yearind][27],'13':sigma[yearind][28]},
                 '10':{'6':sigma[yearind][29],'7':sigma[yearind][30],'9':sigma[yearind][31]}, '11':{'4':sigma[yearind][32],'12':sigma[yearind][33],'13':sigma[yearind][34]}, '12':{'9':sigma[yearind][35],'11':sigma[yearind][36],'14':sigma[yearind][37]},
                 '13':{'9':sigma[yearind][38],'11':sigma[yearind][39],'14':sigma[yearind][40]}, '14':{'6':sigma[yearind][41],'12':sigma[yearind][42],'13':sigma[yearind][43]}
                 }   
    elif graphA == graphD:
        for i in range(numnodes):    
            for j in range(numnodes): 
                d, p = dijkstra({'1':{'2':sigma[yearind][0],'3':sigma[yearind][1],'4':sigma[yearind][2]},'2':{'1':sigma[yearind][3],'4':sigma[yearind][4],'5':sigma[yearind][5]},'3':{'1':sigma[yearind][6],'4':sigma[yearind][7],'6':sigma[yearind][8]},    
                '4':{'1':sigma[yearind][9],'2':sigma[yearind][10],'3':sigma[yearind][11],'5':sigma[yearind][12],'7':sigma[yearind][13],'10':sigma[yearind][14]},'5':{'2':sigma[yearind][15],'4':sigma[yearind][16],'10':sigma[yearind][17],'11':sigma[yearind][18]}, '6':{'3':sigma[yearind][19],'7':sigma[yearind][20],'8':sigma[yearind][21]},
                '7':{'4':sigma[yearind][22],'6':sigma[yearind][23],'9':sigma[yearind][24]}, '8':{'6':sigma[yearind][25],'9':sigma[yearind][26]}, '9':{'7':sigma[yearind][27],'8':sigma[yearind][28],'10':sigma[yearind][29]},
                '10':{'4':sigma[yearind][30],'5':sigma[yearind][31],'9':sigma[yearind][32],'11':sigma[yearind][33],'12':sigma[yearind][34]}, '11':{'5':sigma[yearind][35],'10':sigma[yearind][36],'12':sigma[yearind][37],'14':sigma[yearind][38]}, '12':{'10':sigma[yearind][39],'11':sigma[yearind][40],'13':sigma[yearind][41]},
                '13':{'12':sigma[yearind][42],'14':sigma[yearind][43]}, '14':{'11':sigma[yearind][44],'13':sigma[yearind][45]}
                }  , nodes[i], nodes[j])
                if i == j:
                    continue  # don't include lightpaths of length 0
                else:
                    dis.append(d)
                    path.append(p)
        graphvar = {'1':{'2':sigma[yearind][0],'3':sigma[yearind][1],'4':sigma[yearind][2]},'2':{'1':sigma[yearind][3],'4':sigma[yearind][4],'5':sigma[yearind][5]},'3':{'1':sigma[yearind][6],'4':sigma[yearind][7],'6':sigma[yearind][8]},    
                '4':{'1':sigma[yearind][9],'2':sigma[yearind][10],'3':sigma[yearind][11],'5':sigma[yearind][12],'7':sigma[yearind][13],'10':sigma[yearind][14]},'5':{'2':sigma[yearind][15],'4':sigma[yearind][16],'10':sigma[yearind][17],'11':sigma[yearind][18]}, '6':{'3':sigma[yearind][19],'7':sigma[yearind][20],'8':sigma[yearind][21]},
                '7':{'4':sigma[yearind][22],'6':sigma[yearind][23],'9':sigma[yearind][24]}, '8':{'6':sigma[yearind][25],'9':sigma[yearind][26]}, '9':{'7':sigma[yearind][27],'8':sigma[yearind][28],'10':sigma[yearind][29]},
                '10':{'4':sigma[yearind][30],'5':sigma[yearind][31],'9':sigma[yearind][32],'11':sigma[yearind][33],'12':sigma[yearind][34]}, '11':{'5':sigma[yearind][35],'10':sigma[yearind][36],'12':sigma[yearind][37],'14':sigma[yearind][38]}, '12':{'10':sigma[yearind][39],'11':sigma[yearind][40],'13':sigma[yearind][41]},
                '13':{'12':sigma[yearind][42],'14':sigma[yearind][43]}, '14':{'11':sigma[yearind][44],'13':sigma[yearind][45]}
                }  
    elif graphA == graphAL:
        for i in range(numnodes):    
            for j in range(numnodes): 
                d, p = dijkstra({'1':{'4':sigma[yearind][0],'5':sigma[yearind][1]},'2':{'3':sigma[yearind][2],'7':sigma[yearind][3]},'3':{'2':sigma[yearind][4],'8':sigma[yearind][5]},    
                '4':{'1':sigma[yearind][6],'5':sigma[yearind][7],'9':sigma[yearind][8]},'5':{'1':sigma[yearind][9],'4':sigma[yearind][10],'6':sigma[yearind][11]}, '6':{'5':sigma[yearind][12],'7':sigma[yearind][13],'11':sigma[yearind][14]},
                '7':{'2':sigma[yearind][15],'6':sigma[yearind][16],'10':sigma[yearind][17]}, '8':{'3':sigma[yearind][18],'10':sigma[yearind][19]}, '9':{'4':sigma[yearind][20],'11':sigma[yearind][21]},
                '10':{'7':sigma[yearind][22],'8':sigma[yearind][23],'11':sigma[yearind][24]}, '11':{'6':sigma[yearind][25],'9':sigma[yearind][26],'10':sigma[yearind][27]}
                }  , nodes[i], nodes[j])
                if i == j:
                    continue  # don't include lightpaths of length 0
                else:
                    dis.append(d)
                    path.append(p)
        graphvar = {'1':{'4':sigma[yearind][0],'5':sigma[yearind][1]},'2':{'3':sigma[yearind][2],'7':sigma[yearind][3]},'3':{'2':sigma[yearind][4],'8':sigma[yearind][5]},    
                '4':{'1':sigma[yearind][6],'5':sigma[yearind][7],'9':sigma[yearind][8]},'5':{'1':sigma[yearind][9],'4':sigma[yearind][10],'6':sigma[yearind][11]}, '6':{'5':sigma[yearind][12],'7':sigma[yearind][13],'11':sigma[yearind][14]},
                '7':{'2':sigma[yearind][15],'6':sigma[yearind][16],'10':sigma[yearind][17]}, '8':{'3':sigma[yearind][18],'10':sigma[yearind][19]}, '9':{'4':sigma[yearind][20],'11':sigma[yearind][21]},
                '10':{'7':sigma[yearind][22],'8':sigma[yearind][23],'11':sigma[yearind][24]}, '11':{'6':sigma[yearind][25],'9':sigma[yearind][26],'10':sigma[yearind][27]}
                } 
    pathdists = []
    links = []  
                                               
    for i in range(np.size(path)):
        pathdists.append(getlinklen(path[i],graphnormA,edges)[0])
        links.append(getlinklen(path[i],graphvar,edges)[1])
    
    estlam = np.zeros([numedges,int(numlam[yearind])]) # 0 for empty, 1 for occupied
    reqlams = 0
    conten = 0
    failures = 0
    ct128 = 0
    ct64 = 0
    ct16 =0 
    ct4 = 0
    ct2 = 0
    Um = 0
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
        lamslot = [np.where(estlam[randedges[k]]==0)[0][0] for k in range(np.size(randedges))] # first available wavelength slot for this edge
        # need to check SNR for all the edges in the path
        edgesuc = 0
        edgesuc2 = 0
        FT = np.zeros(np.size(linkSNR[0][randedges],0))
        
        for j in range(np.size(linkSNR[0][randedges],0)): # for each edge in the path
            #if linkSNR[randedges][j][prmnopt[randedges[j]]] > FT:
        
            if (prmn[yearind][randedges][j] - FT128)/sigma[yearind][randedges[j]] > numsig:
                FT[j] = FT128
                ct128 = ct128 + 1
                edgesuc = edgesuc + 1
            elif (prmn[yearind][randedges][j] - FT64)/sigma[yearind][randedges[j]] > numsig:
                FT[j] = FT64
                ct64 = ct64 + 1
                edgesuc = edgesuc + 1
            elif (prmn[yearind][randedges][j] - FT16)/sigma[yearind][randedges[j]] > numsig:
                FT[j] = FT16
                ct16 = ct16 + 1
                edgesuc = edgesuc + 1
            elif (prmn[yearind][randedges][j] - FT4)/sigma[yearind][randedges[j]] > numsig:
                FT[j] = FT4
                ct4 = ct4 + 1
                edgesuc = edgesuc + 1
            elif (prmn[yearind][randedges][j] - FT2)/sigma[yearind][randedges[j]] > numsig:
                FT[j] = FT2
                ct2 = ct2 + 1
                edgesuc = edgesuc + 1
            else:
                break   
        
        if edgesuc == np.size(linkSNR[0][randedges],0):
            # generate new SNR value here
            for w in range(np.size(randedges)):
                #estSNR = SNRnew(edgelens[randedges[w]], linkpert[randedges[w]], linkPch[randedges[w]], Lspans, np.count_nonzero(estlam[randedges[w]])+1 )
                estSNR = SNRnew(edgelens[randedges[w]], np.count_nonzero(estlam[randedges[w]])+1, yearind)
                if estSNR > FT[w]:
                    # link successfully established
                    Um = Um + (prmn[yearind][randedges][w]- FT[w]) - (numsig*sigma[yearind][randedges[w]])  
                    edgesuc2 = edgesuc2 + 1
            if edgesuc2 == np.size(randedges):
                # path successfully established
                reqlams = reqlams + 1
                for l in range(len(randedges)):
                    estlam[randedges[l]][lamslot[l]] = 1
            else: 
                # link not established
                failures = failures + 1
    ava = (reqlams/numreq)*100 
    tottime = ((sum(randdist)*1e3*1.468)/299792458)[0]
    if showres:
        print("Variance-aided availability = " + str(ava) + "%") 
        print("Variance-aided total traversal time = " + str('%.2f' % tottime) + "s")
    return ava, estlam, reqlams, tottime,conten, conf, ct128, ct64, ct16, ct4,ct2, failures, Um
# =============================================================================
testdes = []
testsrc = []
for _ in range(2000):
    rsctest, rdstest = requestgen(graphA)
    testsrc.append(rsctest)
    testdes.append(rdstest)                                                                                                                        # graph,edges,Rsource,Rdest,showres,numsig,nodes,numedges,edgelens,Lspans
#test1, test2, test3, test4, test5, test6, test7, test8, test9, test10, test11, test12, test13 = varrtap(graphA, edgesA, testsrc, testdes, False,5.0,nodesA,numedgesA,edgelensA, LspansA, 0)
#test14 = test13/(test7 + test8 + test9 + test10 + test11)
# =============================================================================
# %%
def varrtap2(edges,Rsource,Rdest,showres,numsig,nodes,numedges,edgelens,Lspans,yearind):
    dis = []
    path = []
    numnodes = np.size(nodes)
    if graphA == graphN:
        for i in range(numnodes):    
            for j in range(numnodes): 
                d, p = dijkstra({'1':{'2':sigma[yearind][0],'3':sigma[yearind][1],'8':sigma[yearind][2]},'2':{'1':sigma[yearind][3],'3':sigma[yearind][4],'4':sigma[yearind][5]},'3':{'1':sigma[yearind][6],'2':sigma[yearind][7],'6':sigma[yearind][8]},    
                 '4':{'2':sigma[yearind][9],'5':sigma[yearind][10],'11':sigma[yearind][11]},'5':{'4':sigma[yearind][12],'6':sigma[yearind][13],'7':sigma[yearind][14]}, '6':{'3':sigma[yearind][15],'5':sigma[yearind][16],'10':sigma[yearind][17],'14':sigma[yearind][18]},
                 '7':{'5':sigma[yearind][19],'8':sigma[yearind][20],'10':sigma[yearind][21]}, '8':{'1':sigma[yearind][22],'7':sigma[yearind][23],'9':sigma[yearind][24]}, '9':{'8':sigma[yearind][25],'10':sigma[yearind][26],'12':sigma[yearind][27],'13':sigma[yearind][28]},
                 '10':{'6':sigma[yearind][29],'7':sigma[yearind][30],'9':sigma[yearind][31]}, '11':{'4':sigma[yearind][32],'12':sigma[yearind][33],'13':sigma[yearind][34]}, '12':{'9':sigma[yearind][35],'11':sigma[yearind][36],'14':sigma[yearind][37]},
                 '13':{'9':sigma[yearind][38],'11':sigma[yearind][39],'14':sigma[yearind][40]}, '14':{'6':sigma[yearind][41],'12':sigma[yearind][42],'13':sigma[yearind][43]}
                 }, nodes[i], nodes[j])
                if i == j:
                    continue  # don't include lightpaths of length 0
                else:
                    dis.append(d)
                    path.append(p)
        graphvar = {'1':{'2':sigma[yearind][0],'3':sigma[yearind][1],'8':sigma[yearind][2]},'2':{'1':sigma[yearind][3],'3':sigma[yearind][4],'4':sigma[yearind][5]},'3':{'1':sigma[yearind][6],'2':sigma[yearind][7],'6':sigma[yearind][8]},    
                 '4':{'2':sigma[yearind][9],'5':sigma[yearind][10],'11':sigma[yearind][11]},'5':{'4':sigma[yearind][12],'6':sigma[yearind][13],'7':sigma[yearind][14]}, '6':{'3':sigma[yearind][15],'5':sigma[yearind][16],'10':sigma[yearind][17],'14':sigma[yearind][18]},
                 '7':{'5':sigma[yearind][19],'8':sigma[yearind][20],'10':sigma[yearind][21]}, '8':{'1':sigma[yearind][22],'7':sigma[yearind][23],'9':sigma[yearind][24]}, '9':{'8':sigma[yearind][25],'10':sigma[yearind][26],'12':sigma[yearind][27],'13':sigma[yearind][28]},
                 '10':{'6':sigma[yearind][29],'7':sigma[yearind][30],'9':sigma[yearind][31]}, '11':{'4':sigma[yearind][32],'12':sigma[yearind][33],'13':sigma[yearind][34]}, '12':{'9':sigma[yearind][35],'11':sigma[yearind][36],'14':sigma[yearind][37]},
                 '13':{'9':sigma[yearind][38],'11':sigma[yearind][39],'14':sigma[yearind][40]}, '14':{'6':sigma[yearind][41],'12':sigma[yearind][42],'13':sigma[yearind][43]}
                 }  
    elif graphA == graphD:
        for i in range(numnodes):    
            for j in range(numnodes): 
                d, p = dijkstra({'1':{'2':sigma[yearind][0],'3':sigma[yearind][1],'4':sigma[yearind][2]},'2':{'1':sigma[yearind][3],'4':sigma[yearind][4],'5':sigma[yearind][5]},'3':{'1':sigma[yearind][6],'4':sigma[yearind][7],'6':sigma[yearind][8]},    
                '4':{'1':sigma[yearind][9],'2':sigma[yearind][10],'3':sigma[yearind][11],'5':sigma[yearind][12],'7':sigma[yearind][13],'10':sigma[yearind][14]},'5':{'2':sigma[yearind][15],'4':sigma[yearind][16],'10':sigma[yearind][17],'11':sigma[yearind][18]}, '6':{'3':sigma[yearind][19],'7':sigma[yearind][20],'8':sigma[yearind][21]},
                '7':{'4':sigma[yearind][22],'6':sigma[yearind][23],'9':sigma[yearind][24]}, '8':{'6':sigma[yearind][25],'9':sigma[yearind][26]}, '9':{'7':sigma[yearind][27],'8':sigma[yearind][28],'10':sigma[yearind][29]},
                '10':{'4':sigma[yearind][30],'5':sigma[yearind][31],'9':sigma[yearind][32],'11':sigma[yearind][33],'12':sigma[yearind][34]}, '11':{'5':sigma[yearind][35],'10':sigma[yearind][36],'12':sigma[yearind][37],'14':sigma[yearind][38]}, '12':{'10':sigma[yearind][39],'11':sigma[yearind][40],'13':sigma[yearind][41]},
                '13':{'12':sigma[yearind][42],'14':sigma[yearind][43]}, '14':{'11':sigma[yearind][44],'13':sigma[yearind][45]}
                }  , nodes[i], nodes[j])
                if i == j:
                    continue  # don't include lightpaths of length 0
                else:
                    dis.append(d)
                    path.append(p)
        graphvar = {'1':{'2':sigma[yearind][0],'3':sigma[yearind][1],'4':sigma[yearind][2]},'2':{'1':sigma[yearind][3],'4':sigma[yearind][4],'5':sigma[yearind][5]},'3':{'1':sigma[yearind][6],'4':sigma[yearind][7],'6':sigma[yearind][8]},    
                '4':{'1':sigma[yearind][9],'2':sigma[yearind][10],'3':sigma[yearind][11],'5':sigma[yearind][12],'7':sigma[yearind][13],'10':sigma[yearind][14]},'5':{'2':sigma[yearind][15],'4':sigma[yearind][16],'10':sigma[yearind][17],'11':sigma[yearind][18]}, '6':{'3':sigma[yearind][19],'7':sigma[yearind][20],'8':sigma[yearind][21]},
                '7':{'4':sigma[yearind][22],'6':sigma[yearind][23],'9':sigma[yearind][24]}, '8':{'6':sigma[yearind][25],'9':sigma[yearind][26]}, '9':{'7':sigma[yearind][27],'8':sigma[yearind][28],'10':sigma[yearind][29]},
                '10':{'4':sigma[yearind][30],'5':sigma[yearind][31],'9':sigma[yearind][32],'11':sigma[yearind][33],'12':sigma[yearind][34]}, '11':{'5':sigma[yearind][35],'10':sigma[yearind][36],'12':sigma[yearind][37],'14':sigma[yearind][38]}, '12':{'10':sigma[yearind][39],'11':sigma[yearind][40],'13':sigma[yearind][41]},
                '13':{'12':sigma[yearind][42],'14':sigma[yearind][43]}, '14':{'11':sigma[yearind][44],'13':sigma[yearind][45]}
                }  
    elif graphA == graphAL:
        for i in range(numnodes):    
            for j in range(numnodes): 
                d, p = dijkstra({'1':{'4':sigma[yearind][0],'5':sigma[yearind][1]},'2':{'3':sigma[yearind][2],'7':sigma[yearind][3]},'3':{'2':sigma[yearind][4],'8':sigma[yearind][5]},    
                '4':{'1':sigma[yearind][6],'5':sigma[yearind][7],'9':sigma[yearind][8]},'5':{'1':sigma[yearind][9],'4':sigma[yearind][10],'6':sigma[yearind][11]}, '6':{'5':sigma[yearind][12],'7':sigma[yearind][13],'11':sigma[yearind][14]},
                '7':{'2':sigma[yearind][15],'6':sigma[yearind][16],'10':sigma[yearind][17]}, '8':{'3':sigma[yearind][18],'10':sigma[yearind][19]}, '9':{'4':sigma[yearind][20],'11':sigma[yearind][21]},
                '10':{'7':sigma[yearind][22],'8':sigma[yearind][23],'11':sigma[yearind][24]}, '11':{'6':sigma[yearind][25],'9':sigma[yearind][26],'10':sigma[yearind][27]}
                }  , nodes[i], nodes[j])
                if i == j:
                    continue  # don't include lightpaths of length 0
                else:
                    dis.append(d)
                    path.append(p)
        graphvar = {'1':{'4':sigma[yearind][0],'5':sigma[yearind][1]},'2':{'3':sigma[yearind][2],'7':sigma[yearind][3]},'3':{'2':sigma[yearind][4],'8':sigma[yearind][5]},    
                '4':{'1':sigma[yearind][6],'5':sigma[yearind][7],'9':sigma[yearind][8]},'5':{'1':sigma[yearind][9],'4':sigma[yearind][10],'6':sigma[yearind][11]}, '6':{'5':sigma[yearind][12],'7':sigma[yearind][13],'11':sigma[yearind][14]},
                '7':{'2':sigma[yearind][15],'6':sigma[yearind][16],'10':sigma[yearind][17]}, '8':{'3':sigma[yearind][18],'10':sigma[yearind][19]}, '9':{'4':sigma[yearind][20],'11':sigma[yearind][21]},
                '10':{'7':sigma[yearind][22],'8':sigma[yearind][23],'11':sigma[yearind][24]}, '11':{'6':sigma[yearind][25],'9':sigma[yearind][26],'10':sigma[yearind][27]}
                }
    pathdists = []
    links = []                                                
    for i in range(np.size(path)):
        pathdists.append(getlinklen(path[i],graphnormA,edges)[0])
        links.append(getlinklen(path[i],graphvar,edges)[1])
    estlam = np.zeros([numedges,int(numlam[yearind])]) # 0 for empty, 1 for occupied
    reqlams = 0
    conten = 0
    failures = 0
    ct128 = 0
    ct64 = 0
    ct16 =0 
    ct4 = 0
    ct2 = 0
    Um = 0
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
        shptcon = False
        testlamslot = [np.where(estlam[randedges[k]]==0) for k in range(np.size(randedges))]
        for g in range(np.size(randedges)):
            if np.size(testlamslot[g]) == 0:
                if shptcon:
                    break
                target = randedges[g]             
                for y in nodesA: # find source and destination nodes for link randedges[g]
                    if target in [list(edgesA.get(y).values())][0]:
                        srcnode = y
                        desnode = list(edgesA.get(y).keys())[list(edgesA.get(y).values()).index(target)]
                        break 
                if graphA == graphN:
                    graphA2 = removekey({'1':{'2':sigma[yearind][0],'3':sigma[yearind][1],'8':sigma[yearind][2]},'2':{'1':sigma[yearind][3],'3':sigma[yearind][4],'4':sigma[yearind][5]},'3':{'1':sigma[yearind][6],'2':sigma[yearind][7],'6':sigma[yearind][8]},    
                 '4':{'2':sigma[yearind][9],'5':sigma[yearind][10],'11':sigma[yearind][11]},'5':{'4':sigma[yearind][12],'6':sigma[yearind][13],'7':sigma[yearind][14]}, '6':{'3':sigma[yearind][15],'5':sigma[yearind][16],'10':sigma[yearind][17],'14':sigma[yearind][18]},
                 '7':{'5':sigma[yearind][19],'8':sigma[yearind][20],'10':sigma[yearind][21]}, '8':{'1':sigma[yearind][22],'7':sigma[yearind][23],'9':sigma[yearind][24]}, '9':{'8':sigma[yearind][25],'10':sigma[yearind][26],'12':sigma[yearind][27],'13':sigma[yearind][28]},
                 '10':{'6':sigma[yearind][29],'7':sigma[yearind][30],'9':sigma[yearind][31]}, '11':{'4':sigma[yearind][32],'12':sigma[yearind][33],'13':sigma[yearind][34]}, '12':{'9':sigma[yearind][35],'11':sigma[yearind][36],'14':sigma[yearind][37]},
                 '13':{'9':sigma[yearind][38],'11':sigma[yearind][39],'14':sigma[yearind][40]}, '14':{'6':sigma[yearind][41],'12':sigma[yearind][42],'13':sigma[yearind][43]}
                 }, srcnode, desnode) 
                elif graphA == graphD:
                    graphA2 = removekey({'1':{'2':sigma[yearind][0],'3':sigma[yearind][1],'4':sigma[yearind][2]},'2':{'1':sigma[yearind][3],'4':sigma[yearind][4],'5':sigma[yearind][5]},'3':{'1':sigma[yearind][6],'4':sigma[yearind][7],'6':sigma[yearind][8]},    
                '4':{'1':sigma[yearind][9],'2':sigma[yearind][10],'3':sigma[yearind][11],'5':sigma[yearind][12],'7':sigma[yearind][13],'10':sigma[yearind][14]},'5':{'2':sigma[yearind][15],'4':sigma[yearind][16],'10':sigma[yearind][17],'11':sigma[yearind][18]}, '6':{'3':sigma[yearind][19],'7':sigma[yearind][20],'8':sigma[yearind][21]},
                '7':{'4':sigma[yearind][22],'6':sigma[yearind][23],'9':sigma[yearind][24]}, '8':{'6':sigma[yearind][25],'9':sigma[yearind][26]}, '9':{'7':sigma[yearind][27],'8':sigma[yearind][28],'10':sigma[yearind][29]},
                '10':{'4':sigma[yearind][30],'5':sigma[yearind][31],'9':sigma[yearind][32],'11':sigma[yearind][33],'12':sigma[yearind][34]}, '11':{'5':sigma[yearind][35],'10':sigma[yearind][36],'12':sigma[yearind][37],'14':sigma[yearind][38]}, '12':{'10':sigma[yearind][39],'11':sigma[yearind][40],'13':sigma[yearind][41]},
                '13':{'12':sigma[yearind][42],'14':sigma[yearind][43]}, '14':{'11':sigma[yearind][44],'13':sigma[yearind][45]}
                } , srcnode, desnode)  # remove this link from the graph and find shortest path with Dijkstra        
                elif graphA == graphAL:
                    graphA2 = removekey({'1':{'4':sigma[yearind][0],'5':sigma[yearind][1]},'2':{'3':sigma[yearind][2],'7':sigma[yearind][3]},'3':{'2':sigma[yearind][4],'8':sigma[yearind][5]},    
                '4':{'1':sigma[yearind][6],'5':sigma[yearind][7],'9':sigma[yearind][8]},'5':{'1':sigma[yearind][9],'4':sigma[yearind][10],'6':sigma[yearind][11]}, '6':{'5':sigma[yearind][12],'7':sigma[yearind][13],'11':sigma[yearind][14]},
                '7':{'2':sigma[yearind][15],'6':sigma[yearind][16],'10':sigma[yearind][17]}, '8':{'3':sigma[yearind][18],'10':sigma[yearind][19]}, '9':{'4':sigma[yearind][20],'11':sigma[yearind][21]},
                '10':{'7':sigma[yearind][22],'8':sigma[yearind][23],'11':sigma[yearind][24]}, '11':{'6':sigma[yearind][25],'9':sigma[yearind][26],'10':sigma[yearind][27]}
                }, srcnode, desnode)  # remove this link from the graph and find shortest path with Dijkstra 
                dist2, path2 = dijkstra(graphA2, Rsource[i], Rdest[i])
                link2 = (getlinklen(path2,graphA,edges)[1]) # converts nodes traversed to edges traversed = 'new randedges'
                shptcon = True
        if shptcon:
            testlamslot2 = [np.where(estlam[link2[p]]==0) for p in range(np.size(link2))]
            for h in range(np.size(link2)):
                if np.size(testlamslot2[h]) == 0: # second-shortest also blocked    
                    lamconten = True
                else: # if second-shortest not blocked, replace randedges with link2
                    randedges = link2
        if lamconten:
            conten = conten + 1
            continue
        lamslot = [np.where(estlam[randedges[k]]==0)[0][0] for k in range(np.size(randedges))] # first available wavelength slot for this edge
        # need to check SNR for all the edges in the path
        edgesuc = 0
        edgesuc2 = 0
        FT = np.zeros(np.size(linkSNR[0][randedges],0))
        for j in range(np.size(linkSNR[0][randedges],0)): # for each edge in the path
            #if linkSNR[randedges][j][prmnopt[randedges[j]]] > FT:
        
            if (prmn[yearind][randedges][j] - FT128)/sigma[yearind][randedges[j]] > numsig:
                FT[j] = FT128
                ct128 = ct128 + 1
                edgesuc = edgesuc + 1
            elif (prmn[yearind][randedges][j] - FT64)/sigma[yearind][randedges[j]] > numsig:   
                FT[j] = FT64
                ct64 = ct64 + 1
                edgesuc = edgesuc + 1
            elif (prmn[yearind][randedges][j] - FT16)/sigma[yearind][randedges[j]] > numsig:
                FT[j] = FT16
                ct16 = ct16 + 1
                edgesuc = edgesuc + 1
            elif (prmn[yearind][randedges][j] - FT4)/sigma[yearind][randedges[j]] > numsig:
                FT[j] = FT4
                ct4 = ct4 + 1
                edgesuc = edgesuc + 1
            elif (prmn[yearind][randedges][j] - FT2)/sigma[yearind][randedges[j]] > numsig:
                FT[j] = FT2
                ct2 = ct2 + 1
                edgesuc = edgesuc + 1
            else:
                break   
        if edgesuc == np.size(linkSNR[0][randedges],0):
            # generate new SNR value here
            for w in range(np.size(randedges)):
                #estSNR = SNRnew(edgelens[randedges[w]], linkpert[randedges[w]], linkPch[randedges[w]], Lspans, np.count_nonzero(estlam[randedges[w]])+1 )
                estSNR = SNRnew(edgelens[randedges[w]], np.count_nonzero(estlam[randedges[w]])+1, yearind)
                if estSNR > FT[w]:
                    # link successfully established
                    Um = Um + (prmn[yearind][randedges][w]- FT[w]) - (numsig*sigma[yearind][randedges[w]])  
                    edgesuc2 = edgesuc2 + 1
            if edgesuc2 == np.size(randedges):
                # path successfully established
                reqlams = reqlams + 1
                for l in range(len(randedges)):
                    estlam[randedges[l]][lamslot[l]] = 1
            else: 
                # link not established
                failures = failures + 1
    ava = (reqlams/numreq)*100 
    tottime = ((sum(randdist)*1e3*1.468)/299792458)[0]
    if showres:
        print("Variance-aided availability = " + str(ava) + "%") 
        print("Variance-aided total traversal time = " + str('%.2f' % tottime) + "s")
    return ava, estlam, reqlams, tottime,conten, conf, ct128, ct64, ct16, ct4,ct2, failures, Um
#test1, test2, test3, test4, test5, test6, test7, test8, test9, test10, test11, test12, test13 = varrtap2(edgesA, testsrc, testdes, False,5.0,nodesA,numedgesA,edgelensA, LspansA, 0)
#test14 = test13/(test7 + test8 + test9 + test10 + test11)
# %%
def gprout(edges,Rsource,Rdest,showres,numsig,nodes,numedges,edgelens,Lspans,yearind):
    dis = []
    path = []
    numnodes = np.size(nodes)
    if graphA == graphN:
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
        graphvar = {'1':{'2':2100,'3':3000,'8':4800},'2':{'1':2100,'3':1200,'4':1500},'3':{'1':3000,'2':1200,'6':3600},    
                 '4':{'2':1500,'5':1200,'11':3900},'5':{'4':1200,'6':2400,'7':1200}, '6':{'3':3600,'5':2400,'10':2100,'14':3600},
                 '7':{'5':1200,'8':1500,'10':2700}, '8':{'1':4800,'7':1500,'9':1500}, '9':{'8':1500,'10':1500,'12':600,'13':600},
                 '10':{'6':2100,'7':2700,'9':1500}, '11':{'4':3900,'12':1200,'13':1500}, '12':{'9':600,'11':1200,'14':600},
                 '13':{'9':600,'11':1500,'14':300}, '14':{'6':3600,'12':600,'13':300}
                 }  
    elif graphA == graphD:
        for i in range(numnodes):    
            for j in range(numnodes): 
                d, p = dijkstra({'1':{'2':400,'3':160,'4':160},'2':{'1':400,'4':400,'5':240},'3':{'1':160,'4':160,'6':320},    
                '4':{'1':160,'2':400,'3':160,'5':320,'7':240,'10':400},'5':{'2':240,'4':320,'10':480,'11':320}, '6':{'3':320,'7':80,'8':80},
                '7':{'4':240,'6':80,'9':80}, '8':{'6':80,'9':80}, '9':{'7':80,'8':80,'10':240},
                '10':{'4':400,'5':480,'9':240,'11':320,'12':240}, '11':{'5':320,'10':320,'12':240,'14':240}, '12':{'10':240,'11':240,'13':80},
                '13':{'12':80,'14':160}, '14':{'11':240,'13':160}
                }  , nodes[i], nodes[j])
                if i == j:
                    continue  # don't include lightpaths of length 0
                else:
                    dis.append(d)
                    path.append(p)
        graphvar = {'1':{'2':400,'3':160,'4':160},'2':{'1':400,'4':400,'5':240},'3':{'1':160,'4':160,'6':320},    
                '4':{'1':160,'2':400,'3':160,'5':320,'7':240,'10':400},'5':{'2':240,'4':320,'10':480,'11':320}, '6':{'3':320,'7':80,'8':80},
                '7':{'4':240,'6':80,'9':80}, '8':{'6':80,'9':80}, '9':{'7':80,'8':80,'10':240},
                '10':{'4':400,'5':480,'9':240,'11':320,'12':240}, '11':{'5':320,'10':320,'12':240,'14':240}, '12':{'10':240,'11':240,'13':80},
                '13':{'12':80,'14':160}, '14':{'11':240,'13':160}
                }  
    elif graphA == graphAL:
        for i in range(numnodes):    
            for j in range(numnodes): 
                d, p = dijkstra({'1':{'4':1200,'5':1600},'2':{'3':1100,'7':300},'3':{'2':1100,'8':300},    
                '4':{'1':1200,'5':1500,'9':500},'5':{'1':1600,'4':1500,'6':900}, '6':{'5':900,'7':700,'11':1000},
                '7':{'2':300,'6':700,'10':1100}, '8':{'3':300,'10':900}, '9':{'4':500,'11':2200},
                '10':{'7':1100,'8':900,'11':1100}, '11':{'6':1000,'9':2200,'10':1100}
                }  , nodes[i], nodes[j])
                if i == j:
                    continue  # don't include lightpaths of length 0
                else:
                    dis.append(d)
                    path.append(p)
        graphvar = {'1':{'4':1200,'5':1600},'2':{'3':1100,'7':300},'3':{'2':1100,'8':300},    
                '4':{'1':1200,'5':1500,'9':500},'5':{'1':1600,'4':1500,'6':900}, '6':{'5':900,'7':700,'11':1000},
                '7':{'2':300,'6':700,'10':1100}, '8':{'3':300,'10':900}, '9':{'4':500,'11':2200},
                '10':{'7':1100,'8':900,'11':1100}, '11':{'6':1000,'9':2200,'10':1100}
                }
    pathdists = []
    links = []                                                
    for i in range(np.size(path)):
        pathdists.append(getlinklen(path[i],graphnormA,edges)[0])
        links.append(getlinklen(path[i],graphvar,edges)[1])
    
    estlam = np.zeros([numedges,int(numlam[yearind])]) # 0 for empty, 1 for occupied
    reqlams = 0
    conten = 0
    failures = 0
    ct128 = 0
    ct64 = 0
    ct16 =0 
    ct4 = 0
    ct2 = 0
    Um = 0
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
        shptcon = False
        testlamslot = [np.where(estlam[randedges[k]]==0) for k in range(np.size(randedges))]
        for g in range(np.size(randedges)):
            if np.size(testlamslot[g]) == 0:
                if shptcon:
                    break
                target = randedges[g]             
                for y in nodesA: # find source and destination nodes for link randedges[g]
                    if target in [list(edgesA.get(y).values())][0]:
                        srcnode = y
                        desnode = list(edgesA.get(y).keys())[list(edgesA.get(y).values()).index(target)]
                        break 
                if graphA == graphN:
                    graphA2 = removekey({'1':{'2':2100,'3':3000,'8':4800},'2':{'1':2100,'3':1200,'4':1500},'3':{'1':3000,'2':1200,'6':3600},    
                 '4':{'2':1500,'5':1200,'11':3900},'5':{'4':1200,'6':2400,'7':1200}, '6':{'3':3600,'5':2400,'10':2100,'14':3600},
                 '7':{'5':1200,'8':1500,'10':2700}, '8':{'1':4800,'7':1500,'9':1500}, '9':{'8':1500,'10':1500,'12':600,'13':600},
                 '10':{'6':2100,'7':2700,'9':1500}, '11':{'4':3900,'12':1200,'13':1500}, '12':{'9':600,'11':1200,'14':600},
                 '13':{'9':600,'11':1500,'14':300}, '14':{'6':3600,'12':600,'13':300}
                 }, srcnode, desnode) 
                elif graphA == graphD:
                    graphA2 = removekey({'1':{'2':400,'3':160,'4':160},'2':{'1':400,'4':400,'5':240},'3':{'1':160,'4':160,'6':320},    
                '4':{'1':160,'2':400,'3':160,'5':320,'7':240,'10':400},'5':{'2':240,'4':320,'10':480,'11':320}, '6':{'3':320,'7':80,'8':80},
                '7':{'4':240,'6':80,'9':80}, '8':{'6':80,'9':80}, '9':{'7':80,'8':80,'10':240},
                '10':{'4':400,'5':480,'9':240,'11':320,'12':240}, '11':{'5':320,'10':320,'12':240,'14':240}, '12':{'10':240,'11':240,'13':80},
                '13':{'12':80,'14':160}, '14':{'11':240,'13':160}
                } , srcnode, desnode)  # remove this link from the graph and find shortest path with Dijkstra        
                elif graphA == graphAL:
                    graphA2 = removekey({'1':{'4':1200,'5':1600},'2':{'3':1100,'7':300},'3':{'2':1100,'8':300},    
                '4':{'1':1200,'5':1500,'9':500},'5':{'1':1600,'4':1500,'6':900}, '6':{'5':900,'7':700,'11':1000},
                '7':{'2':300,'6':700,'10':1100}, '8':{'3':300,'10':900}, '9':{'4':500,'11':2200},
                '10':{'7':1100,'8':900,'11':1100}, '11':{'6':1000,'9':2200,'10':1100}
                }, srcnode, desnode)  # remove this link from the graph and find shortest path with Dijkstra 
                dist2, path2 = dijkstra(graphA2, Rsource[i], Rdest[i])
                link2 = (getlinklen(path2,graphA,edges)[1]) # converts nodes traversed to edges traversed = 'new randedges'
                shptcon = True
        if shptcon:
            testlamslot2 = [np.where(estlam[link2[p]]==0) for p in range(np.size(link2))]
            for h in range(np.size(link2)):
                if np.size(testlamslot2[h]) == 0: # second-shortest also blocked    
                    lamconten = True
                else: # if second-shortest not blocked, replace randedges with link2
                    randedges = link2
        if lamconten:
            conten = conten + 1
            continue
        lamslot = [np.where(estlam[randedges[k]]==0)[0][0] for k in range(np.size(randedges))] # first available wavelength slot for this edge
        # need to check SNR for all the edges in the path
        edgesuc = 0
        edgesuc2 = 0
        FT = np.zeros(np.size(linkSNR[0][randedges],0))
        for j in range(np.size(linkSNR[0][randedges],0)): # for each edge in the path
            #if linkSNR[randedges][j][prmnopt[randedges[j]]] > FT:
        
            if (prmn[yearind][randedges][j] - FT128)/sigma[yearind][randedges[j]] > numsig:
                FT[j] = FT128
                ct128 = ct128 + 1
                edgesuc = edgesuc + 1
            elif (prmn[yearind][randedges][j] - FT64)/sigma[yearind][randedges[j]] > numsig:   
                FT[j] = FT64
                ct64 = ct64 + 1
                edgesuc = edgesuc + 1
            elif (prmn[yearind][randedges][j] - FT16)/sigma[yearind][randedges[j]] > numsig:
                FT[j] = FT16
                ct16 = ct16 + 1
                edgesuc = edgesuc + 1
            elif (prmn[yearind][randedges][j] - FT4)/sigma[yearind][randedges[j]] > numsig:
                FT[j] = FT4
                ct4 = ct4 + 1
                edgesuc = edgesuc + 1
            elif (prmn[yearind][randedges][j] - FT2)/sigma[yearind][randedges[j]] > numsig:
                FT[j] = FT2
                ct2 = ct2 + 1
                edgesuc = edgesuc + 1
            else:
                break   
        if edgesuc == np.size(linkSNR[0][randedges],0):
            # generate new SNR value here
            for w in range(np.size(randedges)):
                #estSNR = SNRnew(edgelens[randedges[w]], linkpert[randedges[w]], linkPch[randedges[w]], Lspans, np.count_nonzero(estlam[randedges[w]])+1 )
                estSNR = SNRnew(edgelens[randedges[w]], np.count_nonzero(estlam[randedges[w]])+1, yearind)
                if estSNR > FT[w]:
                    # link successfully established
                    Um = Um + (prmn[yearind][randedges][w]- FT[w]) - (numsig*sigma[yearind][randedges[w]])  
                    edgesuc2 = edgesuc2 + 1
            if edgesuc2 == np.size(randedges):
                # path successfully established
                reqlams = reqlams + 1
                for l in range(len(randedges)):
                    estlam[randedges[l]][lamslot[l]] = 1
            else: 
                # link not established
                failures = failures + 1
    ava = (reqlams/numreq)*100 
    tottime = ((sum(randdist)*1e3*1.468)/299792458)[0]
    if showres:
        print("Variance-aided availability = " + str(ava) + "%") 
        print("Variance-aided total traversal time = " + str('%.2f' % tottime) + "s")
    return ava, estlam, reqlams, tottime,conten, conf, ct128, ct64, ct16, ct4,ct2, failures, Um
# %%
# ========== this loop resets the network loading after numreqs, numtests times  ===========
def testrout(graph, edges, numtests,showres,numsig, numreq, yearind):

    avaf = np.empty([numtests,1])
    tottimef = np.empty([numtests,1])
    contenf = np.empty([numtests,1])
    ct128f = np.empty([numtests,1])
    ct64f = np.empty([numtests,1])
    ct16f = np.empty([numtests,1])
    ct4f = np.empty([numtests,1])
    ct2f = np.empty([numtests,1])
    failf = np.empty([numtests,1])
    noreachf = np.empty([numtests,1])
    Umf = np.empty([numtests,1])
    
    avavp = np.empty([numtests,1])
    tottimevp = np.empty([numtests,1])
    contenvp = np.empty([numtests,1])
    ct128vp = np.empty([numtests,1])
    ct64vp = np.empty([numtests,1])
    ct16vp = np.empty([numtests,1])
    ct4vp = np.empty([numtests,1])
    ct2vp = np.empty([numtests,1])
    failvp = np.empty([numtests,1])
    Umvp = np.empty([numtests,1])
    
    for i in range(numtests):
        rsrct = []
        rdest = []
        for _ in range(numreq):
            rsct, rdst = requestgen(graph)
            rsrct.append(rsct)
            rdest.append(rdst)
                                                                                                                                                                                                 
        #avavp[i], _, _, tottimevp[i], contenvp[i], conf , ct128vp[i], ct64vp[i],ct16vp[i], ct4vp[i],ct2vp[i], failvp[i], Umvp[i]   = varrtap2(edges,rsrct,rdest,showres,numsig,nodesA,numedgesA,edgelensA,LspansA,yearind)
        avavp[i], _, _, tottimevp[i], contenvp[i], conf , ct128vp[i], ct64vp[i],ct16vp[i], ct4vp[i],ct2vp[i], failvp[i], Umvp[i]   = gprout(edges,rsrct,rdest,showres,numsig,nodesA,numedgesA,edgelensA,LspansA,yearind)
        avaf[i], _, _, contenf[i], ct128f[i], ct64f[i],ct16f[i], ct4f[i],ct2f[i], failf[i], noreachf[i], Umf[i]   = fmrta2(graph,edges,rsrct,rdest,showres,nodesA,numedgesA,edgelensA,fmSNR,LspansA, yearind)
        #avaf2[i], _, _,  contenf2[i], ct128f2[i], ct64f2[i],ct16f2[i], ct4f2[i],ct2f2[i], failf2[i], noreachf2[i], Umf2[i]   = fmrta2(graph,edges,rsrct,rdest,showres,4.0,nodesA,numedgesA,edgelensA,fmSNR,LspansA)
        #avaf3[i], _, _,  contenf3[i], ct128f3[i], ct64f3[i],ct16f3[i], ct4f3[i],ct2f3[i], failf3[i], noreachf3[i], Umf3[i]   = fmrta2(graph,edges,rsrct,rdest,showres,2.0,nodesA,numedgesA,edgelensA,fmSNR,LspansA)
       
    avaavevp = np.mean(avavp)
    ttavevp = np.mean(tottimevp)
    wavconavevp = np.mean(contenvp/numreq)*100 # express as a %
    failavevp = np.mean(failvp/numreq)*100 # express as a %
    avaavef = np.mean(avaf)
    ttavef = np.mean(tottimef)
    wavconavef = np.mean(contenf/numreq)*100 # express as a %
    failavef = np.mean(failf/numreq)*100 # express as a %
    norchavef = np.mean(noreachf/numreq)*100
    
    ct128avevp = np.mean(ct128vp)
    ct64avevp = np.mean(ct64vp)
    ct16avevp = np.mean(ct16vp)
    ct4avevp = np.mean(ct4vp)
    ct2avevp = np.mean(ct2vp)
    ct128avef = np.mean(ct128f)
    ct64avef = np.mean(ct64f)
    ct16avef = np.mean(ct16f)
    ct4avef = np.mean(ct4f)
    ct2avef = np.mean(ct2f)
    
    Umnf = np.mean(Umf)/(ct128avef + ct64avef + ct16avef + ct4avef +  ct2avef)
    Umnvp = np.mean(Umvp)/(ct128avevp + ct64avevp + ct16avevp + ct4avevp +  ct2avevp)
    
    thrptvp = 2*(ct128avevp*7 + ct64avevp*6 + ct16avevp*4 + ct4avevp*2 + ct2avevp)/(ct128avevp + ct64avevp + ct16avevp + ct4avevp + ct2avevp)
    thrptf = 2*(ct128avef*7 + ct64avef*6 + ct16avef*4 + ct4avef*2 + ct2avef)/(ct128avef + ct64avef + ct16avef + ct4avef +  ct2avef)

    return avaavevp, wavconavevp, failavevp ,ttavevp, thrptvp, Umnvp,  avaavef, wavconavef, failavef ,ttavef, thrptf, norchavef, Umnf 

# =============================================================================
testava = False
if testava:
    #numreq = np.linspace(150,500,21,dtype=int)
    numreq = np.linspace(100,1500,21,dtype=int)
    numsig = 5.0
    nrs = np.size(numreq)
    avaavevp = np.empty([nrs,1])
    wavconavevp = np.empty([nrs,1])
    failavevp = np.empty([nrs,1])
    ttavevp = np.empty([nrs,1])
    thrptvp = np.empty([nrs,1]) 
    Umnvp = np.empty([nrs,1]) 
    avaavef = np.empty([nrs,1])
    wavconavef = np.empty([nrs,1])
    failavef = np.empty([nrs,1])
    ttavef = np.empty([nrs,1])
    thrptf = np.empty([nrs,1])
    norchavef = np.empty([nrs,1])
    Umnf = np.empty([nrs,1]) 
    start_time = time.time()
    for i in range(nrs):     
        avaavevp[i], wavconavevp[i], failavevp[i],ttavevp[i], thrptvp[i], Umnvp[i], avaavef[i], wavconavef[i], failavef[i] ,ttavef[i], thrptf[i], norchavef[i], Umnf[i] = testrout(graphA, edgesA, 10,False,numsig,numreq[i],0)
        print("completed iteration " + str(i))
    end_time = time.time()
    duration = time.time() - start_time

# print("Routing calculation duration: " + str(duration))
# =============================================================================


    plt.plot(numreq, avaavevp, label="GP")
    plt.plot(numreq, avaavef, label="FM")
    plt.legend()
    plt.xlabel("No. of requests")
    plt.ylabel("Availability (%)")
    #plt.savefig('zAvavsnreq.pdf', dpi=200,bbox_inches='tight')
    plt.show() 



# %%
timevar = True
if timevar:
    numreq = np.linspace(100,1500,21,dtype=int)
    numsig = 5.0
    nrs = np.size(numreq)
    avaavevp = np.empty([numyears,nrs])
    wavconavevp = np.empty([numyears,nrs])
    failavevp = np.empty([numyears,nrs])
    ttavevp = np.empty([numyears,nrs])
    thrptvp = np.empty([numyears,nrs]) 
    Umnvp = np.empty([numyears,nrs]) 
    avaavef = np.empty([numyears,nrs])
    wavconavef = np.empty([numyears,nrs])
    failavef = np.empty([numyears,nrs])
    ttavef = np.empty([numyears,nrs])
    thrptf = np.empty([numyears,nrs])
    norchavef = np.empty([numyears,nrs])
    Umnf = np.empty([numyears,nrs]) 
    start_time = time.time()
    for i in range(numyears):
        for j in range(nrs):     
            avaavevp[i][j], wavconavevp[i][j], failavevp[i][j],ttavevp[i][j], thrptvp[i][j], Umnvp[i][j], avaavef[i][j], wavconavef[i][j], failavef[i][j] ,ttavef[i][j], thrptf[i][j], norchavef[i][j], Umnf[i][j] = testrout(graphA, edgesA, 10,False,numsig,numreq[j], i)
            print("completed numreq iteration " + str(j))
        print("completed numyears iteration " + str(i))
    end_time = time.time()
    duration = time.time() - start_time


# %% plotting
    if graphA == graphN:
        np.savetxt('avaavevpN.csv', avaavevp, delimiter=',') 
        np.savetxt('avaavefN.csv', avaavef, delimiter=',') 
        np.savetxt('thrptvpN.csv', thrptvp, delimiter=',')
        np.savetxt('thrptfN.csv', thrptvp, delimiter=',')
        np.savetxt('UmnvpN.csv', Umnvp, delimiter=',')
        np.savetxt('UmnfN.csv', Umnf, delimiter=',')
    elif graphA == graphD:
        np.savetxt('avaavevpD.csv', avaavevp, delimiter=',') 
        np.savetxt('avaavefD.csv', avaavef, delimiter=',') 
        np.savetxt('thrptvpD.csv', thrptvp, delimiter=',')
        np.savetxt('thrptfD.csv', thrptvp, delimiter=',')
        np.savetxt('UmnvpD.csv', Umnvp, delimiter=',')
        np.savetxt('UmnfD.csv', Umnf, delimiter=',')
    elif graphA == graphAL:
        np.savetxt('avaavevpAL.csv', avaavevp, delimiter=',') 
        np.savetxt('avaavefAL.csv', avaavef, delimiter=',') 
        np.savetxt('thrptvpAL.csv', thrptvp, delimiter=',')
        np.savetxt('thrptfAL.csv', thrptvp, delimiter=',')
        np.savetxt('UmnvpAL.csv', Umnvp, delimiter=',')
        np.savetxt('UmnfAL.csv', Umnf, delimiter=',')
    
    
    font = { 'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 15}
    matplotlib.rc('font', **font)
    chyr = 0
    plt.plot(numreq, avaavevp[chyr], label="GP")
    plt.plot(numreq, avaavef[chyr], label="FM")
    plt.legend()
    plt.xlabel("No. of requests")
    plt.ylabel("Availability (%)")
    plt.savefig('zAvavsnreq.pdf', dpi=200,bbox_inches='tight')
    plt.show() 
        
    plt.plot(numreq, wavconavevp[chyr], label="GP")
    plt.plot(numreq, wavconavef[chyr], label="FM")
    plt.legend()
    plt.xlabel("No. of requests")
    plt.ylabel("Wavelength contention (%)")
    plt.savefig('zWlcvsnreq.pdf', dpi=200,bbox_inches='tight')
    plt.show() 
        
    plt.plot(numreq, failavevp[chyr], label="GP")
    plt.plot(numreq, failavef[chyr], label="FM")
    plt.legend()
    plt.xlabel("No. of requests")
    plt.ylabel("Failure post-reach est. (%)")
    plt.savefig('zfailvsnreq.pdf', dpi=200,bbox_inches='tight')
    plt.show() 
        
    plt.plot(numreq, thrptvp[chyr], label="GP")
    plt.plot(numreq, thrptf[chyr], label="FM")
    plt.legend()
    plt.xlabel("No. of requests")
    plt.ylabel("Spectral Efficiency (bits/sym)")
    plt.savefig('zThrptvsnreq.pdf', dpi=200,bbox_inches='tight')
    plt.show() 
        
    Rs = 32
    totthrptvp = thrptvp*avaavevp*Rs*1e-2
    totthrptf = thrptf*avaavef*Rs*1e-2
    
    plt.plot(numreq, totthrptvp[chyr], label="GP")
    plt.plot(numreq, totthrptf[chyr], label="FM")
    plt.legend()
    plt.xlabel("No. of requests")
    plt.ylabel("Average throughput (Gb/s)")
    plt.savefig('zTotalthrptvsnreq.pdf', dpi=200,bbox_inches='tight')
    plt.show() 
        
    plt.plot(numreq, Umnvp[chyr], label="GP")
    plt.plot(numreq, Umnf[chyr], label="FM")
    plt.legend()
    plt.xlabel("No. of requests")
    plt.ylabel("Unallocated margin (dB)")
    plt.savefig('zUmvsnreq.pdf', dpi=200,bbox_inches='tight')
    plt.show() 
        
    # %%
    
    blckind = np.empty([numyears,1],dtype=int)
    
    for i in range(numyears):
        blckind[i] = next(x for x, val in enumerate(avaavevp[i]) if val  < 100.0) - 1 # index of highest number of non-blocking requests
    pknumreq = numreq[blckind]
    pkthrptvp = np.empty([numyears,1])
    pkthrptf = np.empty([numyears,1])
    pkUmnvp = np.empty([numyears,1])
    pkUmnf = np.empty([numyears,1])
    for i in range(numyears):
        pkthrptvp[i] = totthrptvp[i][blckind[i][0]]
        pkthrptf[i] = totthrptf[i][blckind[i][0]]
        pkUmnvp[i] = Umnvp[i][blckind[i][0]]
        pkUmnf[i] = Umnf[i][blckind[i][0]]
    if graphA == graphN:    
        np.savetxt('pkthrptvpN.csv', pkthrptvp, delimiter=',') 
        np.savetxt('pkthrptfN.csv', pkthrptf, delimiter=',') 
        np.savetxt('pkUmnvpN.csv', pkUmnvp, delimiter=',')
        np.savetxt('pkUmnfN.csv', pkUmnf, delimiter=',')
        np.savetxt('pknumreqN.csv', pknumreq, delimiter=',')
    elif graphA == graphD:    
        np.savetxt('pkthrptvpD.csv', pkthrptvp, delimiter=',') 
        np.savetxt('pkthrptfD.csv', pkthrptf, delimiter=',') 
        np.savetxt('pkUmnvpD.csv', pkUmnvp, delimiter=',')
        np.savetxt('pkUmnfD.csv', pkUmnf, delimiter=',')
        np.savetxt('pknumreqD.csv', pknumreq, delimiter=',')
    elif graphA == graphAL:    
        np.savetxt('pkthrptvpAL.csv', pkthrptvp, delimiter=',') 
        np.savetxt('pkthrptfAL.csv', pkthrptf, delimiter=',') 
        np.savetxt('pkUmnvpAL.csv', pkUmnvp, delimiter=',')
        np.savetxt('pkUmnfAL.csv', pkUmnf, delimiter=',')
        np.savetxt('pknumreqAL.csv', pknumreq, delimiter=',')
   
    plt.plot(years, pkthrptvp, label="GP")
    plt.plot(years, pkthrptf, label="FM")
    plt.legend()
    plt.xlabel("Time (years)")
    plt.ylabel("Peak average throughput (Gb/s)")
    plt.savefig('zpkthrptvstime.pdf', dpi=200,bbox_inches='tight')
    plt.show() 
    
    plt.plot(years, pkUmnvp, label="GP")
    plt.plot(years, pkUmnf, label="FM")
    plt.legend()
    plt.xlabel("Time (years)")
    plt.ylabel("Peak unallocated margin (dB)")
    plt.savefig('zpkUmnvstime.pdf', dpi=200,bbox_inches='tight')
    plt.show() 
    
    plt.plot(years, pknumreq)
    #plt.legend()
    plt.xlabel("Time (years)")
    plt.ylabel("Blocking threshold (No. req.)")
    plt.savefig('zpkBTvstime.pdf', dpi=200,bbox_inches='tight')
    plt.show() 


