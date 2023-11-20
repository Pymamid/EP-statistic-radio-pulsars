import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import statistics
import math

df = pd.read_csv('pulsar_data_all_surveys.csv',
                sep = "\s+",
                usecols = [0,1,3,4,6,7,9,10],
                names = ['Index','Name','Period','Error in Period','Flux','Error in Flux','Distance','Survey'],
                index_col=0)

required_cols = ['Flux', 'Distance']
df = df[required_cols]

# Dropping the rows which do not have measured values for flux or distance

NANfluxindices = df[df['Flux']=='*'].index
NANdistindices = df[df['Distance']=='*'].index
df = df.drop(NANfluxindices)
df = df.drop(NANdistindices,errors = 'ignore')
zeroflux = df[df['Flux'].astype(float)==0.0].index
df = df.drop(zeroflux)

# Changing flux from mJy to erg/s/cm2/Hz, units of distance from kpc to pc

df = df.astype({ 'Flux':'float', 'Distance':'float'})
df['Flux'] = df['Flux']*10**(-26)
df['Distance'] = df['Distance']*1000
df['Flux_log'] = [math.log(num,10) for num in [num for num in df['Flux'].values]]
df['Dist_log'] = [math.log(num,10) for num in [num for num in df['Distance'].values]]

Flux_log = df['Flux_log'].values
Farray=np.sort(Flux_log)
parray=np.empty(len(Farray))
c=0
while c>=0 and c<len(Farray):
    X=Farray[c:]
    K1,parray[c]=ks_2samp(Flux_log,X)
    if(parray[c] < 10**(-15)):
        break
    c=c+1

Farray = Farray[:c]
parray = parray[:c]

# plotting fig 1
plt.figure(figsize = (16,5))
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15, weight='bold')

ax1 = plt.subplot(121)
ax1.set_ylim(0,250)
ax1.hist(df['Flux_log'], bins=20, align='right', color='moccasin', edgecolor='orange')
ax1.set_xticks(np.arange(-27, -23, 1))
ax1.set_xlabel(r'\textbf{log [$\mathbf{Flux}$ (erg $\mathbf{cm^{-2}s^{-1}}$)]}')
ax1.set_ylabel(r'$\mathbf{Count}$')

ax2 = plt.subplot(122)
ax2.semilogy(Farray, parray)
ax2.set_ylim(10**-125,10*20)
ax2.set_ylabel(r'\textbf{$\mathbf{p_{ks}}$}',size = '15')
ax2.set_xlabel(r'\textbf{log [$\mathbf{S_{th}}$ (erg $\mathbf{cm^{-2} s^{-1}}$)]}',size = '15')

plt.savefig("real_hist_pks.pdf", format="pdf", bbox_inches="tight")

Farray = np.sort(df['Flux_log'])

# Scatter plot of flux and distance (fig 2)

plt.figure(figsize = (8,5))
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15, weight='bold')
plt.scatter(df['Dist_log'].values,df['Flux_log'], color='thistle', edgecolor='slateblue')
plt.ylabel(r'\textbf{log [$\mathbf{Flux}$ (erg $\mathbf{cm^{-2}s^{-1}}$)]}')
plt.xlabel(r'\textbf{log [$\mathbf{Distance}$ (pc)]}')
plt.savefig("1b.pdf", format="pdf", bbox_inches="tight")



def compute_Luminosity(D,S,alpha):
    return 4*math.pi*(l**2)*S*((D/l)**alpha)
l = statistics.median(df['Distance'])
k = 1.002764611617474
def compute_LogLth(D,S_th,alpha):
    return (math.log(4*math.pi*(k**2)*S_th*(l**(2-alpha)),10) + alpha*math.log(D,10))

alphalist = [num/100 for num in range(50,251,5)]

zipped = list(zip(df['Distance'],df['Flux']))

for i in alphalist:
    df[f'Luminosity_alpha{i}'] = [compute_Luminosity(D,S,i) for (D,S) in zipped]
    df[f'Luminosity{i}_Log'] = [math.log(num,10) for num in [num for num in df[f'Luminosity_alpha{i}'].values]]

logThresholds = [-27.276, -26.690, -26.001]
Thresholds = [10**x for x in logThresholds]

LogLth = []
i=0
for FT in Thresholds:
    LogLth.append([])
    for D in df['Distance']:
        LogLth[i].append(compute_LogLth(D,FT,2))
    i = i+1

# Scatter plot of flux and distance (fig 3)

plt.figure(figsize = (8,5))

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=15, weight='bold')
plt.scatter(df['Dist_log'].values,df['Luminosity2.0_Log'], color='thistle', edgecolor='slateblue')
plt.plot(df['Dist_log'],LogLth[1],color = 'brown')

d_x = 2.829303772831025
y_th = compute_LogLth(10**d_x,Thresholds[1],2)
plt.vlines(x = d_x, ymin=y_th, ymax=-15, lw=1.5, color='green', linestyle = '--')
plt.hlines(y = y_th, xmin=2.0, xmax=d_x, lw=1.5, color='green', linestyle='--')
plt.xlim(2.0, 4.42)
plt.ylim(-21.5, -15)
plt.ylabel(r'\textbf{log [$\mathbf{Luminosity}$ (erg $\mathbf{s^{-1}}$)]}')
plt.xlabel(r'\textbf{log [$\mathbf{Distance}$ (pc)]}')

plt.savefig("2a.pdf", format="pdf", bbox_inches="tight")


# calculating the efron petrosian statistic

logThresholds = np.linspace(start = Farray[0], stop = Farray[-1], num = 200)
zipped = list(zip(df['Distance'],df['Flux']))
threshlist = [10**threshold for threshold in logThresholds]

alphalist = [1, 1.5, 2, 0.5]
for i in alphalist:
    df[f'Luminosity_alpha{i}'] = [compute_Luminosity(D,S,i) for (D,S) in zipped]
    df[f'Luminosity{i}_Log'] = [math.log(num,10) for num in [num for num in df[f'Luminosity_alpha{i}'].values]]
       
LogLth = []
i=0
for FT in threshlist:
    LogLth.append([])
    for D in df['Distance']:
        LogLth[i].append(compute_LogLth(D,FT,2))
    i = i+1

# Calculating tau values for alpha = 1

alpha = 1
Tau_a1 = []
for i,thresh in enumerate(threshlist):   
    df['LogLthA_2'] = LogLth[i]
    df_A = df[df['Luminosity2_Log'] >= df['LogLthA_2']]  
    zipped = list(zip(df_A['Dist_log'],df_A[f'Luminosity{alpha}_Log']))

    N_a = []
    R_a = []
    i=0
    for (Di,Li) in zipped:
        L_thi = compute_LogLth(10**Di,thresh,alpha)
        N_a.append(0)
        R_a.append(0)
        for (D,L) in zipped:
            if((D<=Di) and (L>=L_thi)):
                N_a[i] = N_a[i]+1
                if(Li>=L):
                    R_a[i] = R_a[i] + 1
        i = i+1
    E_a = []
    V_a = []
    for num in N_a:
        E_a.append(0.5*(num+1))
        V_a.append((num**2 - 1)/12)
    numerator = 0
    for i in range(len(E_a)):
        numerator = numerator + (R_a[i] - E_a[i])
    denominator = np.sqrt(np.sum(V_a))

    Tau = numerator/denominator
    Tau_a1.append(Tau)

# Calculating tau values for alpha = 1.5

alpha = 1.5
Tau_a1_5 = []

for i,thresh in enumerate(threshlist):
    
    df['LogLthA_2'] = LogLth[i]
    df_A = df[df['Luminosity2_Log'] >= df['LogLthA_2']]  
    
    zipped = list(zip(df_A['Dist_log'],df_A[f'Luminosity{alpha}_Log']))

    N_a = []
    R_a = []
    i=0
    for (Di,Li) in zipped:
        L_thi = compute_LogLth(10**Di,thresh,alpha)
        N_a.append(0)
        R_a.append(0)
        for (D,L) in zipped:
            if((D<=Di) and (L>=L_thi)):
                N_a[i] = N_a[i]+1
                if(Li>=L):
                    R_a[i] = R_a[i] + 1
        i = i+1

    E_a = []
    V_a = []

    for num in N_a:
        E_a.append(0.5*(num+1))
        V_a.append((num**2 - 1)/12)

    numerator = 0
    for i in range(len(E_a)):
        numerator = numerator + (R_a[i] - E_a[i])

    denominator = np.sqrt(np.sum(V_a))

    Tau = numerator/denominator
    Tau_a1_5.append(Tau)

# Calculating tau values for alpha = 2

alpha = 2
Tau_a2 = []

for i,thresh in enumerate(threshlist):
    
    df['LogLthA_2'] = LogLth[i]
    df_A = df[df['Luminosity2_Log'] >= df['LogLthA_2']]  
    
    zipped = list(zip(df_A['Dist_log'],df_A[f'Luminosity{alpha}_Log']))

    N_a = []
    R_a = []
    i=0
    for (Di,Li) in zipped:
        L_thi = compute_LogLth(10**Di,thresh,alpha)
        N_a.append(0)
        R_a.append(0)
        for (D,L) in zipped:
            if((D<=Di) and (L>=L_thi)):
                N_a[i] = N_a[i]+1
                if(Li>=L):
                    R_a[i] = R_a[i] + 1
        i = i+1

    E_a = []
    V_a = []

    for num in N_a:
        E_a.append(0.5*(num+1))
        V_a.append((num**2 - 1)/12)

    numerator = 0
    for i in range(len(E_a)):
        numerator = numerator + (R_a[i] - E_a[i])

    denominator = np.sqrt(np.sum(V_a))

    Tau = numerator/denominator
    Tau_a2.append(Tau)

# Calculating tau values for alpha = 0.5

alpha = 0.5
Tau_a0_5 = []

for i,thresh in enumerate(threshlist):
    
    df['LogLthA_2'] = LogLth[i]
    df_A = df[df['Luminosity2_Log'] >= df['LogLthA_2']]  
    
    zipped = list(zip(df_A['Dist_log'],df_A[f'Luminosity{alpha}_Log']))

    N_a = []
    R_a = []
    i=0
    for (Di,Li) in zipped:
        L_thi = compute_LogLth(10**Di,thresh,alpha)
        N_a.append(0)
        R_a.append(0)
        for (D,L) in zipped:
            if((D<=Di) and (L>=L_thi)):
                N_a[i] = N_a[i]+1
                if(Li>=L):
                    R_a[i] = R_a[i] + 1
        i = i+1

    E_a = []
    V_a = []

    for num in N_a:
        E_a.append(0.5*(num+1))
        V_a.append((num**2 - 1)/12)

    numerator = 0
    for i in range(len(E_a)):
        numerator = numerator + (R_a[i] - E_a[i])

    denominator = np.sqrt(np.sum(V_a))

    Tau = numerator/denominator
    Tau_a0_5.append(Tau)

# Plot of Tau as a function of thresholds (fig 4)

plt.figure(figsize=(10,6))


plt.plot(logThresholds, Tau_a2, color = 'r', label = r'$\alpha$ = 2')
plt.plot(logThresholds, Tau_a1_5 ,color = 'b', label = r'$\alpha$ = 1.5')
plt.plot(logThresholds, Tau_a1 ,color = 'g', label = r'$\alpha$ = 1')
plt.plot(logThresholds, Tau_a0_5, color = 'cyan', label = r'$\alpha$ = 0.5')

plt.axhline(y = 0, color = 'black', linewidth=1.3)
plt.axhline(y = 1, color = 'black', linestyle = '--', linewidth=0.9)
plt.axhline(y = -1, color = 'black', linestyle = '--', linewidth=0.9)
plt.ylabel(r' $ \tau$',size='15')
plt.xlabel(r'\textbf{log [$\mathbf{S_{th}}$ (erg $\mathbf{cm^{-2} s^{-1}}$)]}',size = '15')
plt.legend()

plt.savefig("3a.pdf", format="pdf", bbox_inches="tight")
