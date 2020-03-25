

def GHquad(n):
    import numpy as np    
    # ========================== even n case ==========================
    
    if np.mod(n,2) == 0:

        nn = 1 + n/2
        
        nni = int(nn)
        
        b = np.zeros((nni-1,nni))
        b[0,0] = -2
        
        for i in range(1, nni-1, 1):
            
            b[i,0] = ((-1)**(i+1))*(2 + 4*((i+1) - 1))*np.abs(b[i - 1,0])
        
        b[0,1] = 4  
        for i in range(1, nni-1, 1):
            b[i,i+1] = 4*b[i-1,i]
        
        for j in range(1, nni-1, 1):
                    valmm = np.abs(b[j-1,j]/b[j-1,j-1])
                    valm = valmm + valmm
                    for i in range(j, nni-1, 1):
                        b[i,j] = (-1)*b[i,j-1]*valm
                        valm=valm+valmm
                  
        kk = 0
        polyc = np.zeros(n+2)
        for i in  range(0, nni, 1): 
            poly = b[nni-2,nni-(i+1)]
            polyc[kk] = poly;
            polyc[kk+1] = 0;
            kk = kk + 2;
            
        ssp=np.size(polyc)
        polycc = [float(polyc[i]*((-1)**n)) for i in range(ssp-1)] 
        polyd = np.zeros(n)
        xx = np.roots(polycc) 
        for i in range(n):  
          polyd[i] = polycc[i]*(n-i);
        
        ww = np.zeros(n)
        for i in range(n):
            x=xx[i]
            solde = 0
            for k in range(n):
                solde = solde + polyd[k]*(x**(n-(k+1)))
          
            ww[i]=((2**(n+1))*np.math.factorial(n)*(np.pi**(0.5)))/(solde**2)
        
            
    else:  # if n is odd
        nn = (n+1)/2
        nni = int(nn)
        b = np.zeros((nni,nni))

        # find the first row values: b[n,0]
        b[0,0] = -2
        for i in range(1, nni, 1):
    
            b[i,0] = ((-1)**(i+1))*(2 + 4*((i+1) - 1))*np.abs(b[i - 1,0])

        # find the last row values: b[n,n]
        b[1,1] = -8
        for i in range(2,nni,1):
            b[i,i] = 4*b[i-1,i-1]

        # find the intermediate collumn values 
        for j in range(1,nni-1,1): 
            valmm=np.abs(b[j,j]/b[j,j-1])
            valm=valmm+valmm;
            for i in range(j+1,nni,1):   
                b[i,j]=(-1)*b[i,j-1]*valm
                valm=valm+valmm

        # find the zeros and weights
        kk = 0
        polyc = np.zeros(n+1)
        for i in  range(0, nni, 1): 
            poly = b[nni-1,nni-(i+1)]
            polyc[kk] = poly;
            polyc[kk+1] = 0;
            kk = kk + 2;

    #polycc = polyc*((-1)**n)
        ssp=np.size(polyc)
        polycc = [float(polyc[i]*((-1)**n)) for i in range(ssp)]
        for i in range(1,n+1,2):
            polycc[i] = polycc[i]*(-1)
        polyd = np.zeros(n)
        xx = np.roots(polycc) # find the zeros 
        xx = [xx[i-1] for i in range(len(polycc) - 1) ] # fix due to weird ordering of roots in NumPy - now agrees with MATLAB 

        for i in range(n):  
            polyd[i] = polycc[i]*(n-i);# find the coefficients of the first derivative of the polynomial 

        ww = np.zeros(n)
        for i in range(n):
            x=xx[i]
            solde = 0
            for k in range(n):
                solde = solde + polyd[k]*(x**(n-(k+1)))
  
            ww[i]=((2**(n+1))*np.math.factorial(n)*(np.pi**(0.5)))/(solde**2)
        
        
    return(ww,xx)
        
        


     
