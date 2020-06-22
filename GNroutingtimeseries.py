
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

datagen = False
GPtraining = False
numpoints = 100

PchdBm = np.linspace(-10, 10, num = numpoints, dtype =float) 
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
# 
PchdBm = np.linspace(-10,10,numpoints)
#ripplepertmax = 0.1  # for fixed perturbation between spans 
#ripplepertmin = -0.1
numlam = 20 # initial expected number of wavelengths 

def marginsnrtest(edgelen, Lspans, numlam, NF, alpha):
    Ls = Lspans
    NchNy = numlam
    D = Disp
    gam = NLco
    lam = 1550 # operating wavelength centre [nm]
    f = 299792458/(lam*1e-9) # operating frequency [Hz]
    c = 299792.458 # speed of light in vacuum [nm/ps] -> needed for calculation of beta2
    Rs = 32 # symbol rate [GBaud]
    h = 6.63*1e-34  # Planck's constant [Js]
    #BWNy = (NchNy*Rs)/1e3 
    BWNy = (157*Rs)/1e3 # full 5THz BW
    allin = np.log((10**(alpha/10)))/2 # fibre loss [1/km] -> weird definition, due to exponential decay of electric field instead of power, which is standard 
    #gam = 1.27 # fibre nonlinearity coefficient [1/W*km]
    beta2 = (D*(lam**2))/(2*np.pi*c) # dispersion coefficient at given wavelength [ps^2/km]
    Leff = (1 - np.exp(-2*allin*Ls ))/(2*allin)  # effective length [km]      
    Leffa = 1/(2*allin)  # the asymptotic effective length [km]  
    #ripplepert = 0.1
    #ripplepert = np.random.uniform(0.1,0.2,numspans+1)
    numspans = int(edgelen/Lspans)
    Pun = GNmain(Lspans, 1, numlam, 101, 201, alpha, Disp, PchdBm, NF, NLco,False,numpoints)[0] 
    Popt = PchdBm[np.argmax(Pun)]                                                   
    Gwdm = (1e-3*10**(Popt/10)*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
    Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))*numspans
    Pase = NF*h*f*(db2lin(alpha*Lspans) - 1)*Rs*1e9*numspans
    Pch = 1e-3*10**(Popt/10) 
    snr = (  (Pch/(Pase + Gnli*Rs*1e9))**(-1) + db2lin(26)**(-1)  )**(-1)
    return lin2db(snr) 
def fmdatagentest(edgelens,Lspans, numlam, NF, alpha):
    marginSNR = np.empty([np.size(edgelens),1])
    for i in range(np.size(edgelens)):
        marginSNR[i] = marginsnrtest(edgelens[i],Lspans, numlam, NF, alpha)
    return marginSNR

# =============================================================================
# testlen = 4800.0
years = np.linspace(0,10,21)
numlam = np.linspace(30, 150, np.size(years))
NF = np.linspace(4.5,5.5,np.size(years))
alpha = 0.2 + 0.00163669*years
trxaging = ((1 + 0.05*years)*2).reshape(np.size(years),1)
oxcaging = ((0.03 + 0.007*years)*2).reshape(np.size(years),1)
# 
# testsnrnli = np.empty([np.size(years),1])
# nliampmargin = np.empty([np.size(years),1])
# testsnrjustfibre = np.empty([np.size(years),1])
# 
# for i in range(np.size(years)):
#     #testsnrnli[i] = fmdatagentest(edgelensA[testi], LspansA, numlam[i], NF[i])
#     testsnrnli[i] = marginsnrtest(testlen, LspansA, numlam[i], NF[i], alpha[i])
#     nliampmargin[i] = testsnrnli[0] - testsnrnli[i]
#     

# testsnrnli = testsnrnli -  1.03*np.ones([np.size(years),1])
# testfinalsnr = ( testsnrnli - (trxaging + oxcaging)) 
# 
# plt.plot(years, nliampmargin, label = 'NLI + NF + FA')
# plt.plot(years, trxaging, label = 'TRx aging')
# plt.plot(years, oxcaging, label = 'node filter aging')
# plt.legend()
# plt.xlabel("years")
# plt.ylabel("margin (dB)")
# plt.savefig('marginvaryearsFA.pdf', dpi=200,bbox_inches='tight')
# plt.show()
# 
# plt.plot(years, testsnrnli, label = 'SNR with NF, NLI + FA')
# plt.plot(years, testfinalsnr, label = 'SNR - TRx + filtering')
# plt.xlabel("years")
# plt.ylabel("SNR (dB)")
# plt.legend()
# plt.savefig('marginvarsnr.pdf', dpi=200,bbox_inches='tight')
# plt.show()
# =============================================================================

# %% find the worst-case margin required

#snrmar = fmdatagentest(edgelensA, LspansA, numlam[-1], NF[-1], alpha[-1])
#snrpl =  fmdatagentest(edgelensA, LspansA, numlam[0], NF[0], alpha[0])

#fmS = snrpl - snrmar + trxaging[-1] + oxcaging[-1]  # fixed worst-case S margin
fmD = np.empty([numedgesA,1])
for i in range(numedgesA):
    fmD[i] = 0.08*(edgelensA[i]/1000.0)*5
fmT = trxaging[-1] + oxcaging[-1] + fmD



# %%
numyears = np.size(years)
#sd = 0.04
sd = np.linspace(0.04, 0.08, np.size(years))
numlam = np.linspace(30, 150, np.size(years))
NF = np.linspace(4.5,5.5,np.size(years))
TRxb2b = 26  # TRx B2B SNR 

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
        #BWNy = (NchNy*Rs)/1e3 
        BWNy = (157*Rs)/1e3 # full 5THz BW
        allin = np.log((10**(alpha/10)))/2 # fibre loss [1/km] -> weird definition, due to exponential decay of electric field instead of power, which is standard 
        beta2 = (D*(lam**2))/(2*np.pi*c) # dispersion coefficient at given wavelength [ps^2/km]
        Leff = (1 - np.exp(-2*allin*Ls ))/(2*allin)  # effective length [km]      
        Leffa = 1/(2*allin)  # the asymptotic effective length [km]  
        numspans = int(edgelen/Lspans)
        Pun = GNmain(Lspans, 1, numlam, 101, 201, alpha, Disp, PchdBm, NF, NLco,False,numpoints)[0] 
        Popt = PchdBm[np.argmax(Pun)]                                                   
        Gwdm = (1e-3*10**(Popt/10)*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
        Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))*numspans
        Pase = NF*h*f*(db2lin(alpha*Lspans) - 1)*Rs*1e9*numspans
        Pch = 1e-3*10**(Popt/10) 
        snr = (Pch/(Pase + Gnli*Rs*1e9)) - db2lin(trxaging[yearind] + oxcaging[yearind])
        snr = ( snr**(-1) + (db2lin(TRxb2b))**(-1) )**(-1)
        #snr = snr + np.random.normal(0,db2lin(sd),numpoints)
        sdnorm = sd*(edgelen/1000.0)
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

# %%
x = np.linspace(0,numpoints-1,numpoints)
plt.plot(x,linkSNR[-1][0],'+')
plt.show()


plt.plot(years,linkPopt)
plt.xlabel("years")
plt.ylabel("Popt (dBm)")
plt.savefig('yearspopt.pdf', dpi=200,bbox_inches='tight')
plt.show()

# %%

def GPtrain(x,y):
    y = y.reshape(-1,1)
    x = x.reshape(-1,1)
    scaler = StandardScaler().fit(y)
    y = scaler.transform(y)
    #kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-3, 1e3)) + W(1.0, (1e-5, 1e5))
    kernel = C(0.03, (1e-3, 1e1)) * RBF(0.01, (1e-2, 1e2)) 
    #print("Initial kernel: %s" % kernel)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 20, normalize_y=False, alpha=np.var(y))
    #gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 9, normalize_y=False)
    gpr.fit(x, y)
    #print("Optimised kernel: %s" % gpr.kernel_)
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

if GPtraining:
    x = np.linspace(0,numpoints-1,numpoints)
    prmn = np.empty([numyears,numedgesA,1])
    sigma = np.empty([numyears,numedgesA,1])
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
        prmn[i] = prmnt
        sigma[i] = sigmat


# %% import trained GP models
if GPtraining == False:  
    #prmn = np.empty([numyears,numedgesA,numpoints])
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
            
        #prmn[i] = prmnt
        sigma[i] = sigmat #.reshape(numedgesA,1) 
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
        #BWNy = (NchNy*Rs)/1e3 
        BWNy = (157*Rs)/1e3 # full 5THz BW
        allin = np.log((10**(alpha/10)))/2 # fibre loss [1/km] -> weird definition, due to exponential decay of electric field instead of power, which is standard 
        beta2 = (D*(lam**2))/(2*np.pi*c) # dispersion coefficient at given wavelength [ps^2/km]
        Leff = (1 - np.exp(-2*allin*Ls ))/(2*allin)  # effective length [km]      
        Leffa = 1/(2*allin)  # the asymptotic effective length [km]  
        numspans = int(edgelen/Lspans)
        Pun = GNmain(Lspans, 1, numlam, 101, 201, alpha, Disp, PchdBm, NF, NLco,False,numpoints)[0] 
        Popt = PchdBm[np.argmax(Pun)]                                                   
        Gwdm = (1e-3*10**(Popt/10)*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
        Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))*numspans
        Pase = NF*h*f*(db2lin(alpha*Lspans) - 1)*Rs*1e9*numspans
        Pch = 1e-3*10**(Popt/10) 
        #snr = (Pch/(Pase + Gnli*Rs*1e9)) - db2lin(trxaging[yearind] + oxcaging[yearind])
        snr = (Pch/(Pase + Gnli*Rs*1e9))
        snr = ( snr**(-1) + (db2lin(TRxb2b))**(-1) )**(-1)
        return lin2db(snr) 
    
def fmdatagen(edgelens,Lspans, yearind):
    fmSNR = np.empty([np.size(edgelens),1])
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


def SNRnew(edgelen,numlam, yearind):
        Ls = LspansA
        NchNy = numlam
        D = Disp
        gam = NLco
        lam = 1550 # operating wavelength centre [nm]
        f = 299792458/(lam*1e-9) # operating frequency [Hz]
        c = 299792.458 # speed of light in vacuum [nm/ps] -> needed for calculation of beta2
        Rs = 32 # symbol rate [GBaud]
        h = 6.63*1e-34  # Planck's constant [Js]
        #BWNy = (NchNy*Rs)/1e3 
        BWNy = (157*Rs)/1e3 # full 5THz BW
        allin = np.log((10**(alpha[yearind]/10)))/2 # fibre loss [1/km] -> weird definition, due to exponential decay of electric field instead of power, which is standard 
        beta2 = (D*(lam**2))/(2*np.pi*c) # dispersion coefficient at given wavelength [ps^2/km]
        Leff = (1 - np.exp(-2*allin*Ls ))/(2*allin)  # effective length [km]      
        Leffa = 1/(2*allin)  # the asymptotic effective length [km]  
        numspans = int(edgelen/Ls)
        Pun = GNmain(Ls, 1, numlam, 101, 201, alpha[yearind], Disp, PchdBm, NF[yearind], NLco,False,numpoints)[0] 
        Popt = PchdBm[np.argmax(Pun)]                                                   
        Gwdm = (1e-3*10**(Popt/10)*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]
        Gnli = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))*numspans
        Pase = NF[yearind]*h*f*(db2lin(alpha[yearind]*Ls) - 1)*Rs*1e9*numspans
        Pch = 1e-3*10**(Popt/10) 
        snr = (Pch/(Pase + Gnli*Rs*1e9)) - db2lin(trxaging[yearind] + oxcaging[yearind])
        snr = ( snr**(-1) + (db2lin(TRxb2b))**(-1) )**(-1)
        #snr = snr + np.random.normal(0,db2lin(sd),numpoints)
        sdnorm = sd[yearind]*(edgelen/1000.0)
        return lin2db(snr) + np.random.normal(0,sdnorm,1)

test = SNRnew(edgelensA[0],40,0)

FT2 = 0.24
FT4 = 3.25 # all correspond to BER of 2e-2
FT16 = 12.72
FT64 = 18.43
FT128 = 22.35

     
# %% fixed margin routing algorithm 


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
    
    estlam = np.zeros([numedges,int(numlam[-1])]) # 0 for empty, 1 for occupied
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
                #print("request " + str(i) + " denied -- insufficient reach on edge " + str(randedges[j]))
                break   
        
        if edgesuc == np.size(linkSNR[0][randedges],0):
            # generate new SNR value here
            for w in range(np.size(randedges)):
                # retrieve estlam row corresponding to randegdes[w] index, find number of 1s, pass to SNRnew()
               #edgelen, Lspans, numlam, NF,sd, alpha, yearind
                
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
testsrc = []
testdes = []
for _ in range(2000):
    rsctest, rdstest = requestgen(graphA)
    testsrc.append(rsctest)
    testdes.append(rdstest)   
#test1, _, _, _, _, _, _, _, _, test10, _, _ = fmrta(graphA, edgesA, testsrc, testdes, False,nodesA,numedgesA,edgelensA,fmSNR, LspansA, -1)
# =============================================================================


# %%
def removekey(d, keysrc, keydes):
    r = dict(d)
    del r.get(keysrc)[keydes]
    return r 

def fmrta2(graph, edges, Rsource, Rdest, showres,nodes,numedges,edgelens,fmSNR, Lspans, yearind):
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
        links.append(getlinklen(path[i],graphA,edges)[1])
    estlam = np.zeros([numedges,int(numlam[-1])]) # 0 for empty, 1 for occupied
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
        shptcon = False
        lamconten = False
        testlamslot = [np.where(estlam[randedges[k]]==0) for k in range(np.size(randedges))]
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
                #print("request " + str(i) + " denied -- insufficient reach on edge " + str(randedges[j]))
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
 
test21, test22, test23, test24, test25, test26, test27, test28, test29, test210, test211, test212 = fmrta2(graphA, edgesA, testsrc, testdes, False,nodesA,numedgesA,edgelensA,fmSNR, LspansA, 0)

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
    
    estlam = np.zeros([numedges,int(numlam[-1])]) # 0 for empty, 1 for occupied
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
    testdes.append(rdstest)   
                                                                                                                       # graph,edges,Rsource,Rdest,showres,numsig,nodes,numedges,edgelens,Lspans
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
    
    estlam = np.zeros([numedges,int(numlam[-1])]) # 0 for empty, 1 for occupied
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
    
    avaf2 = np.empty([numtests,1])
    tottimef2 = np.empty([numtests,1])
    contenf2 = np.empty([numtests,1])
    ct128f2 = np.empty([numtests,1])
    ct64f2 = np.empty([numtests,1])
    ct16f2 = np.empty([numtests,1])
    ct4f2 = np.empty([numtests,1])
    ct2f2 = np.empty([numtests,1])
    failf2 = np.empty([numtests,1])
    noreachf2 = np.empty([numtests,1])
    Umf2 = np.empty([numtests,1])
    
    avaf3 = np.empty([numtests,1])
    tottimef3 = np.empty([numtests,1])
    contenf3 = np.empty([numtests,1])
    ct128f3 = np.empty([numtests,1])
    ct64f3 = np.empty([numtests,1])
    ct16f3 = np.empty([numtests,1])
    ct4f3 = np.empty([numtests,1])
    ct2f3 = np.empty([numtests,1])
    failf3 = np.empty([numtests,1])
    noreachf3 = np.empty([numtests,1])
    Umf3 = np.empty([numtests,1])
    
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
                                                                                                                                                                                                 
        avavp[i], _, _, tottimevp[i], contenvp[i], conf , ct128vp[i], ct64vp[i],ct16vp[i], ct4vp[i],ct2vp[i], failvp[i], Umvp[i]   = varrtap2(edges,rsrct,rdest,showres,numsig,nodesA,numedgesA,edgelensA,LspansA,yearind)
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
    avaavef2 = np.mean(avaf2)
    ttavef2 = np.mean(tottimef2)
    wavconavef2 = np.mean(contenf2/numreq)*100 # express as a %
    failavef2 = np.mean(failf2/numreq)*100 # express as a %
    avaavef3 = np.mean(avaf3)
    ttavef3 = np.mean(tottimef3)
    wavconavef3 = np.mean(contenf3/numreq)*100 # express as a %
    failavef3 = np.mean(failf3/numreq)*100 # express as a %
    
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
    ct128avef2 = np.mean(ct128f2)
    ct64avef2 = np.mean(ct64f2)
    ct16avef2 = np.mean(ct16f2)
    ct4avef2 = np.mean(ct4f2)
    ct2avef2 = np.mean(ct2f2)
    ct128avef3 = np.mean(ct128f3)
    ct64avef3 = np.mean(ct64f3)
    ct16avef3 = np.mean(ct16f3)
    ct4avef3 = np.mean(ct4f3)
    ct2avef3 = np.mean(ct2f3)
    
    Umnf = np.mean(Umf)/(ct128avef + ct64avef + ct16avef + ct4avef +  ct2avef)
    Umnf2 = np.mean(Umf2)/(ct128avef2 + ct64avef2 + ct16avef2 + ct4avef2 +  ct2avef2)
    Umnf3 = np.mean(Umf3)/(ct128avef3 + ct64avef3 + ct16avef3 + ct4avef3 +  ct2avef3)
    Umnvp = np.mean(Umvp)/(ct128avevp + ct64avevp + ct16avevp + ct4avevp +  ct2avevp)
    
    thrptvp = 2*(ct128avevp*7 + ct64avevp*6 + ct16avevp*4 + ct4avevp*2 + ct2avevp)/(ct128avevp + ct64avevp + ct16avevp + ct4avevp + ct2avevp)
    thrptf = 2*(ct128avef*7 + ct64avef*6 + ct16avef*4 + ct4avef*2 + ct2avef)/(ct128avef + ct64avef + ct16avef + ct4avef +  ct2avef)
    thrptf2 = 2*(ct128avef2*7 + ct64avef2*6 + ct16avef2*4 + ct4avef2*2 + ct2avef2)/(ct128avef2 + ct64avef2 + ct16avef2 + ct4avef2 + ct2avef2)
    thrptf3 = 2*(ct128avef3*7 + ct64avef3*6 + ct16avef3*4 + ct4avef3*2 + ct2avef3)/(ct128avef3 + ct64avef3 + ct16avef3 + ct4avef3 + ct2avef3) 
    
    return avaavef2, wavconavef2, failavef2 ,ttavef2, thrptf2, Umnf2, avaavef3, wavconavef3, failavef3 ,ttavef3, thrptf3, Umnf3, avaavevp, wavconavevp, failavevp ,ttavevp, thrptvp, Umnvp,  avaavef, wavconavef, failavef ,ttavef, thrptf, norchavef, Umnf 

reqvar = True
if reqvar:
    numreq = np.linspace(150,500,21,dtype=int)
    numsig = 0.5
    nrs = np.size(numreq)
    avaavef2 = np.empty([nrs,1])
    wavconavef2 = np.empty([nrs,1])
    failavef2 = np.empty([nrs,1])
    ttavef2 = np.empty([nrs,1])
    thrptf2 = np.empty([nrs,1]) 
    Umnf2 = np.empty([nrs,1]) 
    avaavef3 = np.empty([nrs,1])
    wavconavef3 = np.empty([nrs,1])
    failavef3 = np.empty([nrs,1])
    ttavef3 = np.empty([nrs,1])
    thrptf3 = np.empty([nrs,1]) 
    Umnf3 = np.empty([nrs,1]) 
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
        avaavef2[i], wavconavef2[i], failavef2[i] ,ttavef2[i], thrptf2[i], Umnf2[i], avaavef3[i], wavconavef3[i], failavef3[i] ,ttavef3[i], thrptf3[i], Umnf3[i],avaavevp[i], wavconavevp[i], failavevp[i],ttavevp[i], thrptvp[i], Umnvp[i], avaavef[i], wavconavef[i], failavef[i] ,ttavef[i], thrptf[i], norchavef[i], Umnf[i] = testrout(graphA, edgesA, 20,False,numsig,numreq[i],0)
    end_time = time.time()
    duration = time.time() - start_time
else:
    numreq = 200
    numsig = np.linspace(0.5,6.0,21)
    nrs = np.size(numsig)
    avaavef2 = np.empty([nrs,1])
    wavconavef2 = np.empty([nrs,1])
    failavef2 = np.empty([nrs,1])
    ttavef2 = np.empty([nrs,1])
    thrptf2 = np.empty([nrs,1]) 
    Umnf2 = np.empty([nrs,1]) 
    avaavef3 = np.empty([nrs,1])
    wavconavef3 = np.empty([nrs,1])
    failavef3 = np.empty([nrs,1])
    ttavef3 = np.empty([nrs,1])
    thrptf3 = np.empty([nrs,1]) 
    Umnf3 = np.empty([nrs,1]) 
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
        avaavef2[i], wavconavef2[i], failavef2[i] ,ttavef2[i], thrptf2[i], Umnf2[i], avaavef3[i], wavconavef3[i], failavef3[i] ,ttavef3[i], thrptf3[i], Umnf3[i],avaavevp[i], wavconavevp[i], failavevp[i],ttavevp[i], thrptvp[i], Umnvp[i], avaavef[i], wavconavef[i], failavef[i] ,ttavef[i], thrptf[i], norchavef[i], Umnf[i] = testrout(graphA, edgesA, 10,False,numsig[i],numreq, 0)
    end_time = time.time()
    duration = time.time() - start_time

print("Routing calculation duration: " + str(duration))

# %% plotting
font = { 'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 15}
matplotlib.rc('font', **font)

if reqvar:
    plt.plot(numreq, avaavevp, label="GP")
    plt.plot(numreq, avaavef, label="FM 6dB")
    plt.plot(numreq, avaavef2, label="FM 4dB")
    plt.plot(numreq, avaavef3, label="FM 2dB")
    plt.legend()
    plt.xlabel("No. of requests")
    plt.ylabel("Availability (%)")
    plt.savefig('zAvavsnreq.pdf', dpi=200,bbox_inches='tight')
    plt.show() 
    
    plt.plot(numreq, wavconavevp, label="GP")
    plt.plot(numreq, wavconavef, label="FM 6dB")
    plt.plot(numreq, wavconavef2, label="FM 4dB")
    plt.plot(numreq, wavconavef3, label="FM 2dB")
    plt.legend()
    plt.xlabel("No. of requests")
    plt.ylabel("Wavelength contention (%)")
    plt.savefig('zWlcvsnreq.pdf', dpi=200,bbox_inches='tight')
    plt.show() 
    
    plt.plot(numreq, failavevp, label="GP")
    plt.plot(numreq, failavef, label="FM 6dB")
    plt.plot(numreq, failavef2, label="FM 4dB")
    plt.plot(numreq, failavef3, label="FM 2dB")
    plt.legend()
    plt.xlabel("No. of requests")
    plt.ylabel("Failure post-reach est. (%)")
    plt.savefig('zfailvsnreq.pdf', dpi=200,bbox_inches='tight')
    plt.show() 
    
    plt.plot(numreq, thrptvp, label="GP")
    plt.plot(numreq, thrptf, label="FM 6dB")
    plt.plot(numreq, thrptf2, label="FM 4dB")
    plt.plot(numreq, thrptf3, label="FM 2dB")
    plt.legend()
    plt.xlabel("No. of requests")
    plt.ylabel("Spectral Efficiency (bits/sym)")
    plt.savefig('zThrptvsnreq.pdf', dpi=200,bbox_inches='tight')
    plt.show() 
    
    Rs = 32
    totthrptvp = thrptvp*avaavevp*Rs*1e-2
    totthrptf = thrptf*avaavef*Rs*1e-2
    totthrptf2 = thrptf2*avaavef2*Rs*1e-2
    totthrptf3 = thrptf3*avaavef3*Rs*1e-2
    
    plt.plot(numreq, totthrptvp, label="GP")
    plt.plot(numreq, totthrptf, label="FM 6dB")
    plt.plot(numreq, totthrptf2, label="FM 4dB")
    plt.plot(numreq, totthrptf3, label="FM 2dB")
    plt.legend()
    plt.xlabel("No. of requests")
    plt.ylabel("Average throughput (Gb/s)")
    plt.savefig('zTotalthrptvsnreq.pdf', dpi=200,bbox_inches='tight')
    plt.show() 
    
    plt.plot(numreq, Umnvp, label="GP")
    plt.plot(numreq, Umnf, label="FM 6dB")
    plt.plot(numreq, Umnf2, label="FM 4dB")
    plt.plot(numreq, Umnf3, label="FM 2dB")
    plt.legend()
    plt.xlabel("No. of requests")
    plt.ylabel("Unallocated margin (dB)")
    plt.savefig('zUmvsnreq.pdf', dpi=200,bbox_inches='tight')
    plt.show() 
    
else: 
    plt.plot(numsig, avaavevp, label="GP")
    plt.plot(numsig, avaavef, label="FM 6dB")
    plt.plot(numsig, avaavef2, label="FM 4dB")
    plt.plot(numsig, avaavef3, label="FM 2dB")
    plt.legend()
    plt.xlabel("$\sigma$")
    plt.ylabel("Availability (%)")
    plt.savefig('zAvavsnsig.pdf', dpi=200,bbox_inches='tight')
    plt.show() 
    
    plt.plot(numsig, wavconavevp, label="GP")
    plt.plot(numsig, wavconavef, label="FM 6dB")
    plt.plot(numsig, wavconavef2, label="FM 4dB")
    plt.plot(numsig, wavconavef3, label="FM 2dB")
    plt.legend()
    plt.xlabel("$\sigma$")
    plt.ylabel("Wavelength contention (%)")
    plt.savefig('zWlcvsnsig.pdf', dpi=200,bbox_inches='tight')
    plt.show() 
    
    plt.plot(numsig, failavevp, label="GP")
    plt.plot(numsig, failavef, label="FM 6dB")
    plt.plot(numsig, failavef2, label="FM 4dB")
    plt.plot(numsig, failavef3, label="FM 2dB")
    plt.legend()   
    plt.xlabel("$\sigma$")
    plt.ylabel("Failure post-reach est. (%)")
    plt.savefig('zfailvsnsig.pdf', dpi=200,bbox_inches='tight')
    plt.show() 
    
    plt.plot(numsig, thrptvp, label="GP")
    plt.plot(numsig, thrptf, label="FM 6dB")
    plt.plot(numsig, thrptf2, label="FM 4dB")
    plt.plot(numsig, thrptf3, label="FM 2dB")
    plt.legend()    
    plt.xlabel("$\sigma$")
    plt.ylabel("Spectral Efficiency (bits/sym)")
    plt.savefig('zThrptvsnsig.pdf', dpi=200,bbox_inches='tight')
    plt.show() 
    
    Rs = 32
    totthrptvp = thrptvp*avaavevp*Rs*1e-2
    totthrptf = thrptf*avaavef*Rs*1e-2
    totthrptf2 = thrptf2*avaavef2*Rs*1e-2
    totthrptf3 = thrptf3*avaavef3*Rs*1e-2
    
    plt.plot(numsig, totthrptvp, label="GP")
    plt.plot(numsig, totthrptf, label="FM 6dB")
    plt.plot(numsig, totthrptf2, label="FM 4dB")
    plt.plot(numsig, totthrptf3, label="FM 2dB")
    plt.legend()
    plt.xlabel("$\sigma$")
    plt.ylabel("Average throughput (Gb/s)")
    plt.savefig('zTotalthrptvsnsig.pdf', dpi=200,bbox_inches='tight')
    plt.show() 
    
    plt.plot(numsig, Umnvp, label="GP")
    plt.plot(numsig, Umnf, label="FM 6dB")
    plt.plot(numsig, Umnf2, label="FM 4dB")
    plt.plot(numsig, Umnf3, label="FM 2dB")
    plt.legend()
    plt.xlabel("$\sigma$")
    plt.ylabel("Unallocated margin (dB)")
    plt.savefig('zUmvsnsig.pdf', dpi=200,bbox_inches='tight')
    plt.show() 
    

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
if MIstuff:
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
    #MIripple = findMI(SNRripple)
# %% ================================== Reach calculation ==================================
if reachcalculation:
# find the BER from: On the Bit Error Probability of QAM Modulation - Michael P. Fitz 
    Lspans = 100
    PchreachdBm = np.linspace(-5,5,numpoints)
    lossreach = NDFISlossoverallnzmean
    dispreach = NDFISdispnzmean
    # =============================================================================
    SNRNYr = GNmain(Lspans, Nspans, 157, 101, 201, lossreach,dispreach, PchreachdBm, NF, NLco,False,numpoints)[0]
    SNRRSr = GNmain(Lspans, Nspans, 157, 101, 201, lossreach, dispreach, PchreachdBm, NF, NLco,False,numpoints)[1]
    SNRRS2r = GNmain(Lspans, Nspans, 157, 101, 201, lossreach, dispreach, PchreachdBm, NF, NLco,False,numpoints)[2]
    # Ny = 0 for Nyquist, 1 for RS and 2 for RS2
    def reachcalc(Ny, P, M):
        FECthreshold = 2e-2
        BER = np.zeros(numpoints)
        Ns = 20 # start at 2 spans because of the denominator of (22) in Poggiolini's GN model paper - divide by ln(Ns) = 0 for Ns = 1
        while BER[0] < FECthreshold:               
            SNR = GNmain(Lspans, Ns, 157, 101, 201, alpha, Disp, P, NF, NLco,False,numpoints)[Ny]                
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
       
