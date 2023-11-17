#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 23:54:44 2023

@author: erc_magnesia_raj
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import glob
from scipy import fft
import warnings
from scipy import integrate
warnings.filterwarnings('ignore')

#Compute fractional variability
def Fracvar(rate,error):
    mean = np.mean(rate)
    var = 0
    varerr = 0
    for k in range(len(error)):
        var += (rate[k] - mean)**2
        varerr += error[k]**2
    var/=(len(error)-1)
    varerr/=len(error)
    normexcessvar = (var - varerr)/(mean**2) #Normalised excess variance
    
    Fvar = 0
    dFvar = 0
    if(normexcessvar>0):
        Fvar = np.sqrt(normexcessvar)
    elif(normexcessvar<0):
        Fvar = 0
        
    errnormexcessvar = np.sqrt(np.sqrt(2.0/len(error))*(varerr/mean**2) +\
                        (np.sqrt(varerr/len(error))*(2*Fvar/mean))**2)
    dFvar = np.sqrt(Fvar**2 + errnormexcessvar**2) - Fvar
    
    return Fvar, dFvar

#Estimate covariance spectrum in time domain (Wilkinson and Uttley 2009)
def covariance_time_domain(lc,lcerr,reflc,reflcerr,Mseg):
    
    sigcov,sigxs_x,sigxs_y,sigerr_x,sigerr_y = [[],[],[],[],[]]
    Numpt = 0
    for arr in range(Mseg):
        
        #Split LC into M equal segments
        div = int(len(reflc)/Mseg) 
        lctemp = lc[arr*div:(arr+1)*div]
        lctemperr = lcerr[arr*div:(arr+1)*div]
        reflctemp = reflc[arr*div:(arr+1)*div]
        reflctemperr = reflcerr[arr*div:(arr+1)*div]
        
        Numpt = len(lctemp)
        mutemp = np.mean(lctemp)
        mureftemp = np.mean(reflctemp)
        
        w1 = np.zeros(len(lctemp))
        w2 = np.zeros(len(lctemp))
        w3 = np.zeros(len(lctemp))
        w4 = np.zeros(len(lctemp))
        w5 = np.zeros(len(lctemp))

        for j in range(len(lctemp)):
            w1[j] = (lctemp[j] - mutemp)*(reflctemp[j] - mureftemp)
            w2[j] = lctemperr[j]**2
            w3[j] = reflctemperr[j]**2
            w4[j] = (lctemp[j]-mutemp)*(lctemp[j]-mutemp)
            w5[j] = (reflctemp[j] - mureftemp)*(reflctemp[j] - mureftemp)
            
        sigcov.append(np.mean(w1))
        sigerr_x.append(np.mean(w2))
        sigerr_y.append(np.mean(w3))
        sigxs_x.append(np.mean(w4) - np.mean(w2))
        sigxs_y.append(np.mean(w5) - np.mean(w3))
    
    mu_sigcov = np.mean(sigcov)
    mu_sigxs_x = np.mean(sigxs_x)
    mu_sigxs_y = np.mean(sigxs_y)
    mu_sigerr_x = np.mean(sigerr_x)
    mu_sigerr_y = np.mean(sigerr_y)
    
    mean_covariance = mu_sigcov/np.sqrt(mu_sigxs_y)
    cov_error = np.sqrt((mu_sigxs_x*mu_sigerr_y + mu_sigxs_y*mu_sigerr_x +\
        mu_sigerr_x*mu_sigerr_y)/(Mseg*Numpt*mu_sigxs_y))
    if(mean_covariance<0):
        mean_covariance = 0
        cov_error = 0
    
    print(mean_covariance/1e-4,cov_error/1e-4)
        
    return mean_covariance,cov_error
        

#Function to average PSD and CPSD over M segments
def Pbin(M,freqsarr,Pxarr,Pyarr,Cxyarr):
    
    favg,Pxavg,dPxavg,Pyavg,dPyavg,Cxyavg,dCxyavg = [[],[],[],[],[],[],[]]
                    
    #Average CPSD and PSD over M segments
    for i in range(np.shape(Cxyarr)[1]):
        
        px = 0
        py = 0
        cxy = 0
        freq = freqsarr[0][i]
        
        for j in range(np.shape(Cxyarr)[0]):
            
            cxy += Cxyarr[j][i]/M
            px += Pxarr[j][i]/M
            py += Pyarr[j][i]/M
                
        favg.append(freq)
        Cxyavg.append(cxy)
        Pxavg.append(px)
        Pyavg.append(py)
        dPxavg.append(px/np.sqrt(M))
        dPyavg.append(py/np.sqrt(M))
        dCxyavg.append(cxy/np.sqrt(M))
    
    favg = np.array(favg)
    Pxavg = np.array(np.real(Pxavg))
    Pyavg = np.array(np.real(Pyavg))
    dPxavg = np.array(np.real(dPxavg))
    dPyavg = np.array(np.real(dPyavg))
    Cxyavg = np.array(Cxyavg)
    dCxyavg = np.array(dCxyavg)
    
    #Return averaged quantities
    return favg,Pxavg,dPxavg,Pyavg,dPyavg,Cxyavg,dCxyavg

#Function to bin PSD and CPSD in frequency space
def fbin(bfact,farr,PXarr,PYarr,CXYarr,dPXarr,dPYarr,dCXYarr):
    
    avgf,avgPx,davgPx,avgPy,davgPy,avgCxy,davgCxy = [[],[],[],[],[],[],[]]
    Karr = []
        
    bmin = 0
    bmax = 0
    while(bmax<(len(farr))):
        
        if((bmax-bmin)!=0):
            bmin = bmax
        fmax = bfact*farr[bmax]        
        for index in range(bmin,len(farr)):
            if(farr[index]>=fmax):
                bmax = index
                break
            if(index==len(farr)-1 and farr[index]<=fmax):
                bmax = index + 1
                break
        if((bmax-bmin)==0):
            bfact+=0.1
                
        #Append relevant quantities        
        af = 0
        apx = 0
        apy = 0
        acxy = 0
        errpx = 0
        errpy = 0
        errcxy = 0
                
        for k2 in range(bmin,bmax):
            af += farr[k2]
            apx += PXarr[k2]
            apy += PYarr[k2]
            acxy += CXYarr[k2]
            errpx += dPXarr[k2]**2
            errpy += dPYarr[k2]**2
            errcxy += dCXYarr[k2]**2
        
        af/=(bmax-bmin)
        apx/=(bmax-bmin)
        apy/=(bmax-bmin)
        acxy/=(bmax-bmin)
        errpx = np.sqrt(errpx)/(bmax-bmin)
        errpy = np.sqrt(errpy)/(bmax-bmin)
        errcxy = np.sqrt(errcxy)/(bmax-bmin)
        
        avgf.append(af)
        avgPx.append(apx)
        avgPy.append(apy)
        avgCxy.append(acxy)
        davgPx.append(errpx)
        davgPy.append(errpy)
        davgCxy.append(errcxy)        
        Karr.append(bmax-bmin)
    
    #Return binned quantities
    return avgf,avgPx,avgPy,avgCxy,davgPx,davgPy,davgCxy,Karr
    
#Estimate covariance spectrum in Fourier domain (Uttley et al. 2014)
def covariance_spectrum(lc,lcerr,reflc,reflcerr,lcbkg,refbkg,Mseg,\
                        bfactor,dt,plot,stat,fbmin,fbmax):
    
    fnyq = 0.5*(dt**-1)
    freqs,Px,Py,Cxy = [[],[],[],[]]
    
    #Compute noise level depending on whether the counting statistics
    #are Poissonian or not 
    Pnoise = 0
    Prefnoise = 0

    if(stat=="Poissonian"):
        Pnoise = (2*(np.mean(lc) + np.mean(lcbkg))/(np.mean(lc))**2)
        Prefnoise = (2*(np.mean(reflc) + np.mean(refbkg))/(np.mean(reflc))**2)
        
    elif(stat!="Poissonian"):
        errsq = 0
        errrefsq = 0
        for l in range(len(lcerr)):
            errsq += lcerr[l]**2
            errrefsq += reflcerr[l]**2
        errsq/=len(lcerr)
        errrefsq/=len(reflcerr)
        Pnoise = (errsq/(fnyq*(np.mean(lc))**2))
        Prefnoise = (errrefsq/(fnyq*(np.mean(reflc))**2))
    
        
    #Average power spectrum and cross spectrum over M segments
    for k3 in range(Mseg):
        
        #Split LC into M equal segments
        div = int(len(reflc)/Mseg) 
        lctemp = lc[k3*div:(k3+1)*div]
        reflctemp = reflc[k3*div:(k3+1)*div]
        
        #FFT of comparison-band LC
        Xn = fft.fft(lctemp) 
        fxn = fft.fftfreq(len(lctemp),d=dt)
        #Remove negative frequency bins
        Xn = Xn[fxn>0] 
        fxn = fxn[fxn>0]
        
        #FFT of reference LC
        Yn = fft.fft(reflctemp) 
        fyn = fft.fftfreq(len(reflctemp),d=dt)
        #Remove negative frequency bins 
        Yn = Yn[fyn>0]
        fyn = fyn[fyn>0]
                
        #Compute PSD and CPSD with 
        #rms-squared normalisation for each segment
        normpsdx = (2.0*dt)/((len(lctemp))*(np.mean(lctemp))**2)
        normpsdy = (2.0*dt)/((len(reflctemp))*(np.mean(reflctemp))**2)
        normcross = (2.0*dt)/((len(lctemp))*(np.mean(lctemp))*\
                             (np.mean(reflctemp)))
                
        Psdx = normpsdx*((np.conj(Xn))*(Xn))
        Psdy = normpsdy*((np.conj(Yn))*(Yn))
        crossxy = normcross*((np.conj(Xn))*(Yn))
        
        #Append CPSD and PSDs for each segment to 
        #pass to functions for averaging and binning
        freqs.append(fxn)
        Px.append(Psdx)
        Py.append(Psdy)
        Cxy.append(crossxy)
    
    freqs = np.array(freqs)
    Px = np.array(Px)
    Py = np.array(Py)
    Cxy = np.array(Cxy)
    
    # Average PSDs and CPSD over M segements
    favg,Pxavg,dPxavg,Pyavg,dPyavg,Cxyavg,dCxyavg = Pbin(Mseg,freqs,Px,Py,Cxy)
        
    # Implement frequency dependent binning of averaged PSDs and CPSD
    fb,avgPx,avgPy,avgCxy,avgPxerr,avgPyerr,avgCxyerr,Karr =\
    fbin(bfactor,favg,Pxavg,Pyavg,Cxyavg,dPxavg,dPyavg,dCxyavg)
    
    # #In case frequency dependent binning is not implemented:
    # fb = favg
    # avgPx = Pxavg
    # avgPy = Pyavg
    # avgCxy = Cxyavg
    # avgPxerr = dPxavg
    # avgPyerr = dPyavg
    # avgCxyerr = dCxyavg
    
    avgPx = np.real(avgPx)
    avgPy = np.real(avgPy)
    avgPx = np.array(avgPx)
    avgPy = np.array(avgPy)
    avgCxy = np.array(avgCxy)
    fb = np.array(fb)
    
    #Frequency bin widths
    dfblo = np.zeros(len(fb))
    for w in range(1,len(fb)):
        dfblo[w] = 0.5*(fb[w]-fb[w-1])
    dfblo[0] = dfblo[1]
    dfbhi = np.zeros(len(fb))
    for w2 in range(len(fb)-1):
        dfbhi[w2] = 0.5*(fb[w2+1]-fb[w2])
    dfbhi[-1] = dfbhi[-2]
    dfb = 0.5*(dfblo + dfbhi)
        
    #Averaged number of samples
    Karr = np.array(Karr)
    nsamples = Mseg*Karr
        
    #Noise level of CPSD
    nbias = ((avgPx-Pnoise)*Prefnoise + (avgPy-Prefnoise)*Pnoise +\
            (Pnoise*Prefnoise))/nsamples
        
    #Compute coherence from complex-valued cross spectrum
    Cxyamp = (np.real(avgCxy))**2 + (np.imag(avgCxy))**2 - nbias
    coherence = Cxyamp/((avgPx)*(avgPy)) #Raw
    intcoherence = Cxyamp/((avgPx-Pnoise)*(avgPy-Prefnoise)) #Intrinsic
    
    #Error in intrinsic coherence (from Vaughan and Nowak 1997)
    coherr = np.zeros(len(intcoherence))
    fact = 5
    dcoh = 0
    for u in range(len(coherence)):
        
        #High measured coherence 
        if((intcoherence[u]>fact*(nbias[u]**2/(avgPx[u]-Pnoise))) and\
          (avgPx[u]-Pnoise>fact*Prefnoise/np.sqrt(nsamples[u])) and\
          (avgPy[u]-Prefnoise>fact*Pnoise/np.sqrt(nsamples[u]))):
            
            dcoh = ((2.0/nsamples[u])**0.5)*(1-intcoherence[u]**2)/\
                   (np.abs(intcoherence[u]))
            coherr[u] = (nsamples[u]**-0.5)*\
                        (np.sqrt((2*nsamples[u]*nbias[u]**2)/\
                        (Cxyamp[u]**2) + (Prefnoise/(avgPx[u]-Pnoise))**2 +\
                        (Pnoise/(avgPy[u]-Prefnoise))**2 +\
                        (nsamples[u]*dcoh**2)/(intcoherence[u]**2)))
            coherr[u] *= intcoherence[u]
            coherr[u] = abs(coherr[u])
        
        #Low measured coherence 
        else:    
            coherr[u] = np.sqrt(Prefnoise**2/(avgPx[u]-Prefnoise)**2/\
                        nsamples[u] + Pnoise**2/(avgPy[u]-Pnoise)**2/\
                        nsamples[u] + (dcoh/intcoherence[u])**2)
                        
            coherr[u] *= intcoherence[u]
            coherr[u] = abs(coherr[u])
            
    #Compute phase lag as a function of frequency between the two energy bands
    phaselag = np.arctan(np.imag(avgCxy)/np.real(avgCxy))
    #Error in phase lag 
    dphaselag = np.sqrt((1.0-coherence)/(2.0*coherence*nsamples))
    #Time lag
    timelag = phaselag/(2.0*np.pi*fb) 
    #Error in time lag
    dtimelag = dphaselag/(2.0*np.pi*fb)
    
    coherr = np.array(coherr)
    avgPxerr = np.array(avgPxerr)
    avgPyerr = np.array(avgPyerr)
        
    if(plot=="yes"):
        
        #Power spectrum
        plt.figure()
        plt.errorbar(fb,avgPx,xerr=(dfblo,dfbhi),yerr=avgPxerr,fmt='k-')
        plt.xlabel("Frequency [Hz]",fontsize=12)
        plt.ylabel("rms power",fontsize=12)
        plt.xscale("log")
        plt.yscale("log")
        plt.show()
        
        # Time lag
        plt.figure()
        plt.errorbar(fb,timelag,xerr=(dfblo,dfbhi),yerr=dtimelag,fmt='k.')
        plt.xlabel("Frequency [Hz]",fontsize=12)
        plt.ylabel("Time lag [s]",fontsize=12)
        plt.xscale("log")
        plt.show()
        
        # Coherence
        plt.figure()
        plt.errorbar(fb,coherence,xerr=(dfblo,dfbhi),yerr=coherr,fmt='g.')
        plt.xlabel("Frequency [Hz]",fontsize=12)
        plt.ylabel("Intrinsic coherence",fontsize=12)
        plt.ylim(-1,2)
        plt.xscale("log")
        plt.show()
        
    #Compute covariance and its error over a desired frequency range
    rmsx = np.sqrt((avgPx-Pnoise)*(dfb)*(np.mean(lc))**2)
    rmsy = np.sqrt((avgPy-Prefnoise)*(dfb)*(np.mean(reflc))**2)
    rmsx_noise = np.sqrt((Pnoise)*((np.mean(lc))**2)*(dfb))
    rmsy_noise = np.sqrt((Prefnoise)*((np.mean(reflc))**2)*(dfb))
    covariance = (np.mean(lc))*np.sqrt((Cxyamp*dfb)/(avgPy-Prefnoise))
    dcovsqterm = ((covariance**2)*(rmsy_noise**2)+(rmsy**2)*(rmsx_noise**2)+\
                  (rmsy_noise**2)*(rmsx_noise**2))/(2*nsamples*rmsy**2)
    covariance_err = np.sqrt(dcovsqterm)
        
    covariance = covariance[fb>fbmin]
    covariance_err = covariance_err[fb>fbmin]
    fb = fb[fb>fbmin]
    
    covariance = covariance[fb<fbmax]
    covariance_err = covariance_err[fb<fbmax]
    fb = fb[fb<fbmax]
        
    #Remove NANs from arrays
    isnanarr = np.isnan(covariance)    
    fb = fb[isnanarr==False]
    covariance = covariance[isnanarr==False]
    covariance_err = covariance_err[isnanarr==False]
    isnanarr = np.isnan(covariance_err)    
    fb = fb[isnanarr==False]
    covariance = covariance[isnanarr==False]
    covariance_err = covariance_err[isnanarr==False]
    
    # Average over desired frequency range and propagate errors on covariance    
    summed_covariance = np.sum(covariance)    
    err_mcov = 0
    for index in range(len(covariance)):
        err_mcov += (covariance_err[index])**2
    err_mcov = np.sqrt(err_mcov)
    
    print(summed_covariance/1e-4,err_mcov/1e-4)
    
    return summed_covariance, err_mcov

# Reference band light-curve (source)
fname = "ref.fits"
hdulist = fits.open(fname)
data = hdulist[1].data
time = data['TIME']
dt = time[1]-time[0] #Time resolution in seconds
rateref = data['RATE']
errorref = data['ERROR']

# Reference band light-curve (background)
fname = "ref_bkg.fits"
hdulist = fits.open(fname)
data = hdulist[1].data
bgrateref = data['RATE']
errbgrateref = data['ERROR']

#Remove NANs in array
isnanarr = np.isnan(rateref)
rateref = rateref[isnanarr==False]
errorref = errorref[isnanarr==False]
bgrateref = bgrateref[isnanarr==False]
errbgrateref = errbgrateref[isnanarr==False]
time = time[isnanarr==False]

# plt.figure()
# plt.errorbar(time,rateref,yerr=errorref,fmt='k.')
# plt.show()

# fvar,fvarerr = Fracvar(rateref,errorref)
# print(fvar,fvarerr)

#Deal with gaps in LC
tstart = hdulist[2].data['START']
tstop = hdulist[2].data['STOP']
rnewref,errnewref,bgrnewref,bgerrnewref, = [[],[],[],[]]
tnew = []

for j in range(len(tstart)):    
    
    arr1ref = rateref[time>=tstart[j]]
    arr2ref = errorref[time>=tstart[j]]
    arr3ref = bgrateref[time>=tstart[j]]
    arr4ref = errbgrateref[time>=tstart[j]]
    arr5ref = time[time>=tstart[j]]
    
    arr1ref = arr1ref[arr5ref<=tstop[j]]
    arr2ref = arr2ref[arr5ref<=tstop[j]]
    arr3ref = arr3ref[arr5ref<=tstop[j]]
    arr4ref = arr4ref[arr5ref<=tstop[j]]
    arr5ref = arr5ref[arr5ref<=tstop[j]]
    
    for k in range(len(arr1ref)):
        rnewref.append(arr1ref[k])
        errnewref.append(arr2ref[k])
        bgrnewref.append(arr3ref[k])
        bgerrnewref.append(arr4ref[k])

rnewref = np.array(rnewref)
errnewref = np.array(errnewref)
tnew = dt*np.arange(0,len(rnewref),1)
telapse = tnew[-1]-tnew[0]

bfactor = 1.2
M = 1
fmi = 0
fmx = 3e-2
plot = "no"
stat = "NP"
cov,dcov = [[],[]]

emin = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.3,1.5,1.7,2.0,3.0,4.0,5.0,7.0]
emax = [0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.3,1.5,1.7,2.0,3.0,4.0,5.0,7.0,10.0]

# Comparison-band light-curves
fvarc,fvarcerr = [[],[]]
count = 1
for file in sorted(glob.glob("epn*keV*.lc")):
    
    hdulist = fits.open(file)
    data = hdulist[1].data
    header = hdulist[1].header
    time = data['TIME']  
    rate = data['RATE']
    error = data['ERROR']
    
    fbkg = 'epnbkg' + str(count) + '.lc'
    hdulist = fits.open(fbkg)
    data = hdulist[1].data
    header = hdulist[1].header
    bgrate = data['RATE']
    errbgrate = data['ERROR']
    
    #Remove NANs in array
    isnanarr = np.isnan(rate)
    time = time[isnanarr==False]
    rate = rate[isnanarr==False]
    error = error[isnanarr==False]
    bgrate = bgrate[isnanarr==False]
    errbgrate = errbgrate[isnanarr==False]
                    
    #Deal with gaps in LC
    tstart = hdulist[2].data['START']
    tstop = hdulist[2].data['STOP']
    rnew,errnew,bgrnew,bgerrnew = [[],[],[],[]]
    
    for j in range(len(tstart)):
        
        arr1 = rate[time>=tstart[j]]
        arr2 = error[time>=tstart[j]]
        arr3 = bgrate[time>=tstart[j]]
        arr4 = errbgrate[time>=tstart[j]]
        arr5 = time[time>=tstart[j]]
        
        arr1 = arr1[arr5<=tstop[j]]
        arr2 = arr2[arr5<=tstop[j]]
        arr3 = arr3[arr5<=tstop[j]]
        arr4 = arr4[arr5<=tstop[j]]
        arr5 = arr5[arr5<=tstop[j]]
        
        for k in range(len(arr1)):
            rnew.append(arr1[k])
            errnew.append(arr2[k])
            bgrnew.append(arr3[k])
            bgerrnew.append(arr4[k])
    
    rnew = np.array(rnew)
    errnew = np.array(errnew)
    bgrnew = np.array(bgrnew)
    bgerrnew = np.array(bgerrnew)
        
    # # Fractional variability
    # fvar,fvarerr = Fracvar(rate,error)
    # fvarc.append(fvar)
    # fvarcerr.append(fvarerr)
    
    # Covariance    
    
    # #Time domain
    # intcovtd,intcoverrtd = covariance_time_domain(rnew,errnew,rnewref,\
    #                                               errnewref,M)
    
    #Frequency domain
    intcov,intcoverr = covariance_spectrum(rnew,errnew,rnewref,errnewref,\
                        bgrnewref,bgerrnewref,M,bfactor,dt,plot,stat,\
                        fmi,fmx)
        
    cov.append(intcov)
    dcov.append(intcoverr)
    
    count += 1

# Convert to XSPEC readable format using response file
hdulist = fits.open("EPN.rmf")
data = hdulist[2].data
channel = data['CHANNEL']
EMIN = data['E_MIN']
EMAX = data['E_MAX']

chans = np.arange(1,4097,1)
fluxes = np.zeros(len(chans))
dfluxes = np.zeros(len(chans))

for j in range(len(emin)):
    for k3 in range(len(channel)):
            
        if(0.5*(EMIN[k3]+EMAX[k3])>=emax[j]):
            fluxes[channel[k3]] = cov[j]
            dfluxes[channel[k3]] = dcov[j]
            break

Q = np.column_stack((chans,fluxes,dfluxes))
np.savetxt("covflux_chan.dat",Q,fmt='%i %s %s',delimiter='   ')

