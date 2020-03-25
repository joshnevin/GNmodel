# %% ################### imports ####################


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#  ################### equipment characteristics ####################

# % fixed (for now)
numpoints = 100
Ls = 100.0 # span length [km]
#Ls = np.linspace(40,100, numpoints)
lam = 1550 # operating wavelength [nm]
#lam = np.linspace(1530,1570, numpoints)
f = 299792458/(lam*1e-9) # operating frequency [Hz]
c = 299792.458 # speed of light in vacuum [nm/ps] -> needed for calculation of beta2
Rs = 32 # symbol rate [GBaud]
h = 6.63*1e-34  # Planck's constant [Js]
BWNy = 5 # full BW of Nyquist signal [THz]
NchNy = 157 # number of channels of Nyquist signal (found from channel spacing = Rs -> makes a full BW of 5THz)
NchRS = 101 # number of channels of non-Nyquist signal
NchRS2 = 201
Df = 50 # channel frequency spacing for non-Nyquist [GHz]
Df2 = 25 
Ns = 10 # number of spans 
BchRS = 41.6 # channel BW for non-Nyquist signal [GHz]
BchRS2 = 20.8
#al = np.linspace(0.20,0.30,numpoints) # fibre loss [dB/km]

al = 0.2525 # fibre loss [dB/km]


#al = 0.3
allin = np.log((10**(al/10)))/2 # fibre loss [1/km] -> weird definition, due to exponential decay of electric field instead of power, which is standard 
allinn = np.log((10**(al/10)))/2
gam = 1.27 # fibre nonlinearity coefficient [1/W*km]
D = 15.95 # dispersion coefficient [ps/(nmkm)]

#D = 13.3
#D = np.linspace(13.3, 18.6, numpoints)
beta2 = (D*(lam**2))/(2*np.pi*c) # dispersion coefficient at given wavelength [ps^2/km]


Leff = (1 - np.exp(-2*allin*Ls ))/(2*allin)  # effective length [km]      
Leffa = 1/(2*allin)  # the asymptotic effective length [km]

LF = 1 # filtering effects coefficient, takes values 0 < LF <= 1 

# variables for SNR plot


PchdBm = np.linspace(-10, 10, num = numpoints, dtype =float) # power per channel [dBm] 
Pch = 1e-3*10**(PchdBm/10)  # ^ [W]

#PchdBm = 0 # power per channel [dBm] 
#Pch = 1e-3*10**(PchdBm/10)  # ^ [W]
Gwdm = (Pch*NchNy)/(BWNy*1e12) # flat-top value of PSD of signal [W/Hz]






# %% ################### analytical approximations  ####################

## equation 13, Gnli(0) for single-span Nyquist-WDM 

GnliEq13 = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BWNy**2)  ) )/(np.pi*beta2*Leffa ))

## equation 15, Gnli(0) for single-span non-Nyquist-WDM 

GnliEq15 = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS**2)*(NchRS**((2*BchRS)/Df))  ) )/(np.pi*beta2*Leffa ))

GnliEq15v2 = 1e24*(8/27)*(gam**2)*(Gwdm**3)*(Leff**2)*((np.arcsinh((np.pi**2)*0.5*beta2*Leffa*(BchRS2**2)*(NchRS2**((2*BchRS2)/Df2))  ) )/(np.pi*beta2*Leffa ))

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


#epsilon = 0.05 # exponent from (7) 
epsilon = (np.log(1 + (2*Leffa*(1 - Ns + Ns*nthHarmonic(int(Ns-1))))/( Ns*Ls*np.arcsinh((np.pi**2)*0.5*beta2*Leffa*BWNy**2 )   )  ))/(np.log(Ns))

GnliEq13mul = GnliEq13*Ns**(1+epsilon)   # implementation of (7) 
GnliEq13mulinc = GnliEq13*Ns  # incoherent NLI noise accumulation case (epsilon = 0)

GnliEq15mul = GnliEq15*Ns**(1+epsilon)   # implementation of (7) 
GnliEq15mul2 = GnliEq15v2*Ns**(1+epsilon)   # implementation of (7) 

# %% ############################ Monte Carlo - amplitude values are for Pch = 0dBm ############################

# =============================================================================
# af1 = -BWNy/2
# bf1 = BWNy/2
# af2 = -BWNy/2
# bf2 = BWNy/2
# 
# N = int(1e7)
# 
# def stepfunc(x, height, width):    
#     return height*(np.heaviside(x + width, 1) + np.heaviside(width - x, 1) - 1)
# 
# def NYquistfunc(f1, f2 ):
#     return stepfunc((f1 + f2), 1, BWNy/2)*(( (1 - np.exp(-2*allin*Ls))**2 + 4*np.exp(-2*allin*Ls)*(np.sin(2*np.pi**2*beta2*f1*f2*Ls))**2 )/(4*allin**2 + 16*np.pi**4*beta2**2*(f1**2)*(f2**2)  ) )
# 
# def MonteCarloGnli():
#     MCint = 0.0
#     for i in range(N):
#     
#         f1rand = np.random.uniform(af1, bf1)
#         f2rand = np.random.uniform(af2, bf2)
#     
#         MCint += NYquistfunc(f1rand, f2rand)
#  
#     return (MCint*(BWNy**2)/float(N))
#     
# GnliMC = MonteCarloGnli()*1e24*(16/27)*gam**2*Gwdm**3*Ns**(1+epsilon)
# GnliMCinc = MonteCarloGnli()*1e24*(16/27)*gam**2*Gwdm**3*Ns
# 
# =============================================================================

# %% ############################ ASE noise PSD ###########################

NF = 8.0 # noise figure of the optical amplifiers [dB]
#NF = 10.0
#G = 20 # EDFA gain [dB] --> exactly compensates the link 
#NF = np.linspace(6,10,numpoints)
#G = np.linspace(19.5,20.5,numpoints)
G = al*Ls
NFl = 10**(NF/10) 
Gl = 10**(G/10) 

OSNRmeasBW = 12.478*1e9 # OSNR measurement BW [Hz]
conv = (32*1e9)/OSNRmeasBW

Pasech = NFl*h*f*(Gl - 1)*Rs*1e9*Ns # [W] the ASE noise power in one Nyquist channel across all spans
PaseOSNR = NFl*h*f*(Gl - 1)*OSNRmeasBW*Ns # [W] the ASE noise power in the OSNR measurement BW, defined as Dlam = 0.1nm
PasechRS = NFl*h*f*(Gl - 1)*BchRS*1e9*Ns # [W] the ASE noise power in one non-Nyquist channel across all spans
PasechRS2 = NFl*h*f*(Gl - 1)*BchRS2*1e9*Ns


# %% SNR calc + plotting 

SNRMC = np.zeros(numpoints)

SNRanalytical = np.zeros(numpoints)
SNRanalyticalRS = np.zeros(numpoints)
SNRanalyticalRS2 = np.zeros(numpoints)


#SNRanalyticalOSNR = np.zeros(numpoints)
#SNRMCOSNR = np.zeros(numpoints)



for i in range(numpoints):

    #SNRanalytical[i] = 10*np.log10((LF*Pch)/(Pasech[i] + GnliEq13mul*Rs*1e9))
    #SNRanalyticalRS2[i] = 10*np.log10((LF*Pch)/(PasechRS2[i] + GnliEq15mul2*BchRS2*1e9))
    #SNRanalyticalRS[i] = 10*np.log10((LF*Pch)/(PasechRS[i] + GnliEq15mul*BchRS*1e9))
    
    #SNRMC[i] = 10*np.log10((LF*Pch)/(Pasech[i] + GnliMC*Rs*1e9))
    
    #SNRanalytical[i] = 10*np.log10((LF*Pch)/(Pasech[i] + GnliEq13mul[i]*Rs*1e9))
    #SNRanalyticalRS[i] = 10*np.log10((LF*Pch)/(PasechRS[i] + GnliEq15mul[i]*BchRS*1e9))
    #SNRanalyticalRS2[i] = 10*np.log10((LF*Pch)/(PasechRS2[i] + GnliEq15mul2[i]*BchRS2*1e9))
    
    #SNRanalyticalRS[i] = 10*np.log10((LF*Pch)/(PasechRS + GnliEq15mul[i]*BchRS*1e9))
    #SNRanalytical[i] = 10*np.log10((LF*Pch)/(Pasech + GnliEq13mul[i]*Rs*1e9))
    #SNRanalyticalRS2[i] = 10*np.log10((LF*Pch)/(PasechRS2 + GnliEq15mul2[i]*BchRS2*1e9))
    #SNRMC[i] = 10*np.log10((LF*Pch)/(Pasech[i] + GnliMC[i]*Rs*1e9))
   
    SNRanalytical[i] = 10*np.log10((LF*Pch[i])/(Pasech + GnliEq13mul[i]*Rs*1e9))
    SNRanalyticalRS[i] = 10*np.log10((LF*Pch[i])/(PasechRS + GnliEq15mul[i]*BchRS*1e9))
    SNRanalyticalRS2[i] = 10*np.log10((LF*Pch[i])/(PasechRS2 + GnliEq15mul2[i]*BchRS2*1e9))
    
    
    #SNRanalyticalOSNR[i] = 10*np.log10((LF*Pch)/(PaseOSNR[i] + GnliEq13mul[i]*OSNRmeasBW))
    #SNRMCOSNR[i] = 10*np.log10((LF*Pch)/(PaseOSNR + GnliMC[i]*OSNRmeasBW))
   


#  data saving and plotting section 

#np.savetxt("SNRNyquistpos.csv", SNRanalytical , delimiter=",")
#np.savetxt("SNRnonNyquist50pos.csv", SNRanalyticalRS , delimiter=",")
#np.savetxt("SNRnonNyquist25pos.csv", SNRanalyticalRS2 , delimiter=",")



SNRNyquistunperturbedDAT = pd.read_csv("SNRNyquistunperturbed.csv", header = None) 
SNRnonNyquist50unperturbedDAT = pd.read_csv("SNRnonNyquist50unperturbed.csv", header = None) 
SNRnonNyquist25unperturbedDAT = pd.read_csv("SNRnonNyquist25unperturbed.csv", header = None) 



#np.savetxt("SNRNyquistunperturbed.csv", SNRanalytical , delimiter=",")
#np.savetxt("SNRnonNyquist50unperturbed.csv", SNRanalyticalRS , delimiter=",")
#np.savetxt("SNRnonNyquist25unperturbed.csv", SNRanalyticalRS2 , delimiter=",")


plt.plot(PchdBm, SNRanalytical, label = 'Analytical Nyquist +1%')
plt.plot(PchdBm, SNRanalyticalRS, label = 'Analytical non-Nyquist 50GHz +1%')

#plt.plot(PchdBm, SNRanalyticalRS2, label = 'Analytical non-Nyquist 25GHz')

plt.plot(PchdBm, SNRNyquistunperturbedDAT, label = 'Nyquist UP')
plt.plot(PchdBm, SNRnonNyquist50unperturbedDAT, label = 'Non-Nyquist 50GHz UP')
#plt.plot(PchdBm, SNRnonNyquist25unperturbedDAT, label = 'Non-Nyquist 25GHz UP')


#plt.plot(al, SNRMC, label = 'Monte Carlo')
plt.legend()
plt.ylabel('SNR (dB)')
#plt.xlabel('loss coefficient (dB/km)')
#plt.xlabel('EDFA NF (dB)')
plt.xlabel('Pch (dBm)')
#plt.xlabel('dispersion (ps/nm*km)')
plt.title('Moderate case')
plt.show()



# =============================================================================
# plt.plot(PchDAT, SNRNyquistunperturbedDAT, label = 'Analytical Nyquist')
# plt.plot(PchDAT, SNRnonNyquist50unperturbedDAT, label = 'Analytical non-Nyquist 50GHz')
# plt.plot(PchDAT, SNRnonNyquist25unperturbedDAT, label = 'Analytical non-Nyquist 25GHz')
# #plt.plot(al, SNRMC, label = 'Monte Carlo')
# plt.legend()
# plt.ylabel('SNR (dB)')
# #plt.xlabel('loss coefficient (dB/km)')
# #plt.xlabel('EDFA NF (dB)')
# plt.xlabel('Pch (dBm)')
# #plt.xlabel('dispersion (ps/nm*km)')
# plt.title('Unperturbed case')
# plt.show()
# =============================================================================











