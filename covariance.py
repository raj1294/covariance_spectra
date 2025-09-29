import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import glob
from scipy import fft
import warnings
from scipy import integrate
from stingray import Lightcurve
from stingray import AveragedCrossspectrum

import os, subprocess
from scipy import signal
warnings.filterwarnings('ignore')

def remove_nans(arrays,tS,tE):
        
    tref = arrays[-1]   
    oldarrays_list,newarrays_list = [[],[]]
        
    for qd in range(len(arrays)):
                                                                        
        #Remove NANs
        isnanarr = np.isnan(arrays[qd])
        for qd2 in range(len(arrays[qd])):
            if(isnanarr[qd2]=='True'):
                arrays[qd2] = -1e10
        
        #Ignore BTIs
        for qd3 in range(len(tS)-1):
            
            btiS = tE[qd3]
            btiE = tS[qd3+1]
                        
            for qd4 in range(len(tref)):
                
                if(tref[qd4]>=btiS and tref[qd4]<=btiE\
                   and qd!=len(arrays)-1):
                    
                    arrays[qd][qd4] = -1e10
        
        if(qd!=len(arrays)-1):
        
            newarray = arrays[qd][arrays[qd]>-100]
            newarrays_list.append(newarray)
            newtime = tref[arrays[qd]>-100]
            oldarrays_list.append(arrays[qd])
            
    newarrays_list.append(newtime)
    oldarrays_list.append(tref)
    
    for pind2 in range(len(newarrays_list)):
        for pind3 in range(len(newarrays_list[pind2])):
            newarrays_list[pind2][pind3] =\
                float(newarrays_list[pind2][pind3])
            oldarrays_list[pind2][pind3] =\
                float(oldarrays_list[pind2][pind3])
        
    return oldarrays_list, newarrays_list

def rect_window(rate_arr,tS,tE):
    
    twref,rwref = rate_arr        
    
    ywin = np.zeros(len(rwref))
    for lw in range(len(tS)):
        
        gtimin = tS[lw]
        gtimax = tE[lw]
        
        for lw2 in range(len(twref)):
            
            if(twref[lw2]>=gtimin and twref[lw2]<=gtimax):
                
                ywin[lw2] = 1
    
    # #OR
    # ywin = np.ones(len(rwref))
    
    ywin[0] = 0
    ywin[-1] = 0

    return ywin

#Compute fractional variability
def Fracvar(rate,error):
    mean = np.mean(rate)
    var = 0
    varerr = 0
    for kfvar in range(len(error)):
        var += (rate[kfvar] - mean)**2
        varerr += error[kfvar]**2
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
def covariance_time_domain(lc,lcerr,reflc,reflcerr,Msegs):
        
    sigcov,sigxs_x,sigxs_y,sigerr_x,sigerr_y = [[],[],[],[],[]]
    Numpt = 0
    for arr in range(Msegs):
        
        #Split LC into M equal segments
        divs = int(len(reflc)/Msegs) 
        lctemp = lc[arr*divs:(arr+1)*divs]
        lctemperr = lcerr[arr*divs:(arr+1)*divs]
        reflctemp = reflc[arr*divs:(arr+1)*divs]
        reflctemperr = reflcerr[arr*divs:(arr+1)*divs]
        
        Numpt = len(lctemp)
        mutemp = np.mean(lctemp)
        mureftemp = np.mean(reflctemp)
                
        w1 = np.zeros(len(lctemp))
        w2 = np.zeros(len(lctemp))
        w3 = np.zeros(len(lctemp))
        w4 = np.zeros(len(lctemp))
        w5 = np.zeros(len(lctemp))

        for jp in range(len(lctemp)):
            w1[jp] = (lctemp[jp] - mutemp)*(reflctemp[jp] - mureftemp)
            w2[jp] = lctemperr[jp]**2
            w3[jp] = reflctemperr[jp]**2
            w4[jp] = (lctemp[jp]-mutemp)*(lctemp[jp]-mutemp)
            w5[jp] = (reflctemp[jp] - mureftemp)*(reflctemp[jp] - mureftemp)
                
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
                    
    if(np.isnan(mean_covariance)==True):
        mean_covariance = 0
        cov_error = 0
                
    return mean_covariance, cov_error

#Function to average PSD and CPSD over M segments
def Pbin(MsegPbin,freqsarr,Pxarr,Pyarr,Cxyarr):
        
    favg,Pxavg,dPxavg,Pyavg,dPyavg,Cxyavg,dCxyavg = [[],[],[],[],[],[],[]]
                    
    #Average CPSD and PSD over M segments
    for ipbin in range(np.shape(Cxyarr)[1]):
        
        px = 0
        py = 0
        cxy = 0
        freq = freqsarr[0][ipbin]
        
        for jpbin in range(np.shape(Cxyarr)[0]):
            
            cxy += Cxyarr[jpbin][ipbin]/MsegPbin
            px += Pxarr[jpbin][ipbin]/MsegPbin
            py += Pyarr[jpbin][ipbin]/MsegPbin
                
        favg.append(freq)
        Cxyavg.append(cxy)
        Pxavg.append(px)
        Pyavg.append(py)
        dPxavg.append(px/np.sqrt(MsegPbin))
        dPyavg.append(py/np.sqrt(MsegPbin))
        dCxyavg.append(cxy/np.sqrt(MsegPbin))
    
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
                
        for k5 in range(bmin,bmax):
            af += farr[k5]
            apx += PXarr[k5]
            apy += PYarr[k5]
            acxy += CXYarr[k5]
            errpx += dPXarr[k5]**2
            errpy += dPYarr[k5]**2
            errcxy += dCXYarr[k5]**2
        
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
                        bfactor,Dt,plot,stats,fbmin,fbmax,\
                        window):
            
    #Compute noise level depending on whether the counting statistics
    #are Poissonian or not 
    fnyq = 0.5*(Dt**-1)
    Pnoise = 0
    Prefnoise = 0
    Msegnew = 0
    freqs,Pxn,Pyn,Cxyn,dPxn,dPyn,dCxyn = [[],[],[],[],[],[],[]]
        
    if(Mseg%2==0 and len(reflc)%2!=0):
                
        reflc = reflc[0:-1]
        reflcerr = reflcerr[0:-1]
        refbkg = refbkg[0:-1]
        
    #Average power spectrum and cross spectrum over M segments
    for kz in range(Mseg):
        
        #Split LC into M equal segments
        div = int(len(reflc)/Mseg)
        lctemp = lc[kz*div:(kz+1)*div]
        lctemperr = lcerr[kz*div:(kz+1)*div]
        reflctemp = reflc[kz*div:(kz+1)*div]
        reflctemperr = reflcerr[kz*div:(kz+1)*div]
        lcbkgtemp = lcbkg[kz*div:(kz+1)*div]
        refbkgtemp = refbkg[kz*div:(kz+1)*div]
        windowtemp = window[kz*div:(kz+1)*div]
                
        #Ambient noise level in PSD
        if(stats=="Poissonian"):
            
            Pnoise += (2*(np.mean(lctemp) + np.mean(lcbkgtemp))/\
                      (np.mean(lctemp))**2)
            Prefnoise += (2*(np.mean(reflctemp) + np.mean(refbkgtemp))/\
                         (np.mean(reflctemp))**2)

        if(stats!="Poissonian"):
            
            errsq = 0
            errrefsq = 0
            
            for l in range(len(lctemperr)):
                errsq += lctemperr[l]**2
                errrefsq += reflctemperr[l]**2
                
            errsq/=len(lctemperr)
            errrefsq/=len(reflctemperr)
            
            Pnoise += errsq/(fnyq*(np.mean(lctemp))**2)
            Prefnoise += errrefsq/(fnyq*(np.mean(reflctemp))**2) 
                
        if(np.sum(lctemp)>0):
                                                            
            #FFT of comparison-band LC
            Xn = 0.5*(fft.fft(lctemp+lctemperr)+fft.fft(lctemp-lctemperr)) 
            Xnerr = 0.5*(fft.fft(lctemp+lctemperr)-fft.fft(lctemp-lctemperr))
            Xnconj = 0.5*(np.conj(Xn+Xnerr)+np.conj(Xn-Xnerr))
            Xnconjerr = 0.5*(np.conj(Xn+Xnerr)-np.conj(Xn-Xnerr))
            fxn = fft.fftfreq(len(lctemp),d=Dt)
                        
            #FFT of reference-band LC
            Yn = 0.5*(fft.fft(reflctemp+reflctemperr)+\
                 fft.fft(reflctemp-reflctemperr))
            Ynerr = 0.5*(fft.fft(reflctemp+reflctemperr)-\
                    fft.fft(reflctemp-reflctemperr))
            Ynconj = 0.5*(np.conj(Yn+Ynerr)+np.conj(Yn-Ynerr))
            Ynconjerr = 0.5*(np.conj(Yn+Ynerr)-np.conj(Yn-Ynerr))
            fyn = fft.fftfreq(len(reflctemp),d=Dt)
                        
            # FFT of window function
            Wn = fft.fft(windowtemp)
            remx,Xn = signal.deconvolve(Xn,Wn)
            remy,Yn = signal.deconvolve(Yn,Wn)
            
            Xn = Xn[fxn>0] 
            Xnerr = Xnerr[fxn>0]
            Xnconj = Xnconj[fxn>0]
            Xnconjerr = Xnconjerr[fxn>0]
            
            Yn = Yn[fyn>0]
            Ynerr = Ynerr[fyn>0]
            Ynconj = Ynconj[fyn>0]
            Ynconjerr = Ynconjerr[fyn>0]
            
            fxn = fxn[fxn>0]
            fyn = fyn[fyn>0]
                                                                                        
            # #Compute PSD and CPSD with 
            # rms-squared normalisation for each segment
            normpsdx = (2.0*Dt)/((len(lctemp))*(np.mean(lctemp))**2)
            normpsdy = (2.0*Dt)/((len(reflctemp))*(np.mean(reflctemp))**2)
            normcross = (2.0*Dt)/((len(lctemp))*(np.mean(lctemp))*\
                                 (np.mean(reflctemp)))
                            
            Psdx = normpsdx*Xnconj*Xn
            dPsdx = normpsdx*(Xnconjerr*Xn + Xnconj*Xnerr)
            Psdy = normpsdy*Ynconj*Yn
            dPsdy = normpsdy*(Ynconjerr*Yn + Ynconj*Ynerr)
            Crossxy = normcross*Ynconj*Xn
            dCrossxy = normcross*(Ynconjerr*Xn + Ynconj*Xnerr)
                                                
            if(len(Crossxy)>0 and len(Psdx)>0 and len(Psdy)>0):
                                
                # Append CPSD and PSDs for each segment to 
                # pass to functions for averaging and binning
                freqs.append(fxn)
                Pxn.append(Psdx)
                Pyn.append(Psdy)
                Cxyn.append(Crossxy)
                dPxn.append(dPsdx)
                dPyn.append(dPsdy)
                dCxyn.append(dCrossxy)
                Msegnew += 1
    
    #Average ref band and comp band noise powers
    Pnoise /= Mseg
    Prefnoise /= Mseg

    freqs = np.array(freqs)
    Pxn = np.array(Pxn)
    dPxn = np.array(dPxn)
    Pyn = np.array(Pyn)
    dPyn = np.array(dPyn)
    Cxyn = np.array(Cxyn)
    dCxyn = np.array(dCxyn)
    Mseg = Msegnew
                
    mean_covariance = 0
    err_mcov = 0

    if(len(Cxyn)>0):
                        
        # Average PSDs and CPSD over M segements
        freqx,Pxavg,dPxavg,Pyavg,dPyavg,Cxyavg,dCxyavg =\
        Pbin(Mseg,freqs,Pxn,Pyn,Cxyn)
        
        avgPx = Pxavg
        avgPy = Pyavg
        avgCxy = Cxyavg
        avgPxerr = dPxavg
        avgPyerr = dPyavg
        avgCxyerr = dCxyavg
        dfreqx = freqx[1]-freqx[0]
        Karr = np.ones(len(Pxavg))

        # Implement frequency dependent binning of averaged PSDs and CPSD
        if(bfactor>1):
            freqx,avgPx,avgPy,avgCxy,avgPxerr,avgPyerr,avgCxyerr,Karr =\
            fbin(bfactor,freqx,Pxavg,Pyavg,Cxyavg,dPxavg,dPyavg,dCxyavg)
                                                            
        # Averaged number of samples
        nsamples = Mseg*Karr
                    
        # Noise level of CPSD
        nbias = ((avgPx-Pnoise)*Prefnoise + (avgPy-Prefnoise)*Pnoise +\
                (Pnoise*Prefnoise))/nsamples
                            
        # Compute CPSD amplitude from complex-valued cross spectrum
        Cxyamp = (np.real(avgCxy))**2 + (np.imag(avgCxy))**2 - nbias
        Cxyamperr = 2*(np.real(avgCxy))*(np.real(avgCxyerr)) +\
                    2*(np.imag(avgCxy))*(np.imag(avgCxyerr)) -\
                    ((avgPxerr*Prefnoise) + (avgPyerr*Pnoise))/nsamples
                                        
        # Compute covariance and its error over a desired frequency range
        rmsx = np.sqrt((avgPx-Pnoise)*(dfreqx)*(np.mean(lc))**2)
        rmsy = np.sqrt((avgPy-Prefnoise)*(dfreqx)*(np.mean(reflc))**2)
        rmsx_noise = np.sqrt((Pnoise)*((np.mean(lc))**2)*(dfreqx))
        rmsy_noise = np.sqrt((Prefnoise)*((np.mean(reflc))**2)*(dfreqx))
        
        covariance = (np.mean(lc))*np.sqrt((Cxyamp*dfreqx)/(avgPy-Prefnoise))
        dcovsqterm = ((covariance**2)*(rmsy_noise**2)+\
                      (rmsy**2)*(rmsx_noise**2)+\
                      (rmsy_noise**2)*(rmsx_noise**2))/(2*nsamples*rmsy**2)
        covariance_err = np.sqrt(dcovsqterm)
                                        
        # Filter over a specified frequency range
        covariance = covariance[freqx>fbmin]
        covariance_err = covariance_err[freqx>fbmin]
        avgCxy = avgCxy[freqx>fbmin]
        avgCxyerr = avgCxyerr[freqx>fbmin]
        freqxfilt = freqx[freqx>fbmin]
        
        covariance = covariance[freqxfilt<fbmax]
        covariance_err = covariance_err[freqxfilt<fbmax]
        avgCxy = avgCxy[freqxfilt<fbmax]
        avgCxyerr = avgCxyerr[freqxfilt<fbmax]
        freqxfilt = freqxfilt[freqxfilt<fbmax]
        
        # Remove NANs
        isnancov = np.isnan(covariance)
        covariance = covariance[isnancov==False]
        avgCxy = avgCxy[isnancov==False]
        avgCxyerr = avgCxyerr[isnancov==False]
        covariance_err = covariance_err[isnancov==False]
        freqxfilt = freqxfilt[isnancov==False]
        
        isnancov = np.isnan(covariance_err)
        covariance = covariance[isnancov==False]
        covariance_err = covariance_err[isnancov==False]
        avgCxy = avgCxy[isnancov==False]
        avgCxyerr = avgCxyerr[isnancov==False]
        freqxfilt = freqxfilt[isnancov==False]
        
        # Average over the above frequency range and propagate errors 
        # on covariance 
        mean_covariance = np.mean(covariance)
        err_mcov = 0
        for index in range(len(covariance)):
            err_mcov += covariance_err[index]**2
        err_mcov = np.sqrt(err_mcov)/len(covariance)
                
        if(len(covariance)==0):
            mean_covariance = 0
            err_mcov = 0
        
    return freqx, mean_covariance, err_mcov, avgCxy, avgCxyerr

#Comparison with stingray
from stingray.varenergyspectrum import CovarianceSpectrum
from stingray import EventList
from stingray import AveragedPowerspectrum, AveragedCrossspectrum

ks = 1000
    
def cross_spectrum_stingray(eventsst_ref,eventsst_comp,\
                            Msegstref,bwidth,normalisation,fbinmin,fbinmax):
    
    telapse_ref = eventsst_ref.time[-1]-eventsst_ref.time[0]
    segsize_ref = telapse_ref/Msegstref
        
    crossspec = AveragedCrossspectrum(eventsst_ref,eventsst_comp,\
                                      segment_size=segsize_ref,\
                                      norm=normalisation,dt=bwidth)
    cspec = crossspec.power
    cspecerr = crossspec.power_err
    cspecfreq = crossspec.freq
            
    cspec = cspec[cspecfreq>fbinmin]
    cspecerr = cspecerr[cspecfreq>fbinmin]
    cspecfreq = cspecfreq[cspecfreq>fbinmin]
    cspec = cspec[cspecfreq<fbinmax]
    cspecerr = cspecerr[cspecfreq<fbinmax]
    cspecfreq = cspecfreq[cspecfreq<fbinmax]
                
    return cspecfreq,cspec,cspecerr

def covariance_spectrum_stingray(evfile,\
                                 Msegstref,bwidth,fbmin,fbmax,\
                                 refemin,refemax,egrid,normalisation):
        
    eventsst_ref = EventList.read(evfile,"hea",\
                                  additional_columns=["DET_ID"])
    frq_interval = [fbmin,fbmax]
    rf_band = [refemin,refemax]      
    telapse_ref = eventsst_ref.time[-1]-eventsst_ref.time[0]         
    segsize_ref = telapse_ref/Msegstref
    
    covspec = CovarianceSpectrum(eventsst_ref,\
              freq_interval=frq_interval,segment_size=segsize_ref,\
              bin_time=bwidth,energy_spec=egrid,\
              norm=normalisation,ref_band=rf_band)
    
    covspecE = covspec.energy
    covspecspt = covspec.spectrum
    covspecspterr = covspec.spectrum_error
    
    isnanarrcov = np.isnan(covspecspt)
    covspecE = covspecE[isnanarrcov==False]
    covspecspt = covspecspt[isnanarrcov==False]
    covspecspterr = covspecspterr[isnanarrcov==False]
    isnanarrcov = np.isnan(covspecspterr)
    covspecE = covspecE[isnanarrcov==False]
    covspecspt = covspecspt[isnanarrcov==False]
    covspecspterr = covspecspterr[isnanarrcov==False]

    return covspecE,covspecspt,covspecspterr
    

def make_lc(tarr,bintime,tstart,tstop,statlc):
    
    tobs = np.arange(tstart,tstop+bintime,bintime)
    robs = np.zeros(len(tobs))
    errobs = np.zeros(len(tobs))
    
    for k2 in range(len(tarr)):
        diffindex = np.argmin(abs(tarr[k2]-tobs))
        robs[diffindex] += 1
    
    if(statlc=='gauss'):
        errobs = np.std(robs)
    if(statlc=='poissonian'):
        errobs = np.sqrt(robs)
    
    robs/=bintime
    errobs/=bintime
    
    return tobs,robs,errobs
                   
########################### Covariance spectrum #############################

ks = 1000

# Energy grid
Nenergies = 12
energies_stingray = np.log10(np.logspace(0.3,12.0,Nenergies))
refemin = 0.3
refemax = 12.0

# PSD parameters 
bfactor = 1.0
fminb = 1e-5
fmaxb = 1e-1
statpow = "NP"
plot = "no"
normpow = "abs"

# Detector parameters
tthresh = 10*ks
groupscale = 1

cov,dcov,covtd,dcovtd,fvarc,fvarcerr = [[],[],[],[],[],[]]
energies,denergies = [[],[]]
obsidnum = ["0921360201"]

#Covspec from stingray
refevfilest = "epn_net_obs0921360201_1_en11_ref_stingray.fits"
Mseg = 8
bsize = 366.82480001449585
covspecEst,covspecst,covspecerrst =\
covariance_spectrum_stingray(refevfilest,\
Mseg,bsize,fminb,fmaxb,refemin,refemax,\
energies_stingray,normpow)
isnanarr = np.isnan(covspecst)
covspecst = covspecst[isnanarr==False]
covspecerrst = covspecerrst[isnanarr==False]
covspecEst = covspecEst[isnanarr==False]
isnanarr = np.isnan(covspecerrst)
covspecst = covspecst[isnanarr==False]
covspecerrst = covspecerrst[isnanarr==False]
covspecEst = covspecEst[isnanarr==False]

# infilest = "covfluxst1" + "_" + str(obsid) + ".dat"
# outfilest = "covspecst1" + "_" + str(obsid) + ".pha"
# groupfilest = "covspec_grouped_" + str(obsid) + "_st.pha"
            
for k in range(Nenergies-1):
    
    reflccomb,errreflccomb,complccomb,errcomplccomb = [[],[],[],[]]
    reflcbkgcomb,errreflcbkgcomb,complcbkgcomb,errcomplcbkgcomb =\
    [[],[],[],[]]
    
    windowcomb = []
    
    for k2 in range(len(obsidnum)):
    
        for reflcfile in glob.glob("epn_net_obs" + obsidnum[k2] +\
                                   "*en4*ref.lc"):
                        
            ennum = str(k+1)
                        
            #Reference band (source)
            ObsId = reflcfile.split("obs")[1].split("_")[0]        
                                    
            reflcfile = "epn_net_obs" + str(ObsId) + "_1_en" +\
                         str(ennum) + "_ref.lc"
            
            refbkgfile = "epn_bkg_obs" + ObsId + "_1_en" +\
                         str(ennum) + "_ref.lc"
            
            refevfile = "epn_net_obs" + ObsId + "_1_en" +\
                         str(ennum) + "_ref.fits"
                        
            complcfile = "epn_net_obs" + ObsId + "_1_en" +\
                         str(ennum) + "_comp.lc"
            
            compbkgfile = "epn_bkg_obs" + ObsId + "_1_en" +\
                          str(ennum) + "_comp.lc"
            
            compevfile = "epn_net_obs" + ObsId + "_1_en" +\
                         str(ennum) + "_comp.fits"
            
            infilecov = "covflux" + "_" + str(ObsId) + ".dat"
            outfilecov = "covspec" + "_" + str(ObsId) + ".pha"
            groupfilecov = "covspec_grouped_" + str(ObsId) + ".pha"
                        
            hdulistref = fits.open(reflcfile)
            dataref = hdulistref[1].data
            tstartR = hdulistref[2].data['START']
            tstopR = hdulistref[2].data['STOP']
            bsize = hdulistref[1].header['TIMEDEL']
            telapse = hdulistref[2].header['ONTIME']
            timeref = dataref['TIME']  
            rateref = dataref['RATE']
            errorref = dataref['ERROR']
            
            arraysW = np.transpose(np.column_stack((timeref,rateref)))
            windowref = rect_window(arraysW,tstartR,tstopR)
                            
            #Choose Mseg depending on exposure time
            Mseg = 0
                        
            if (telapse > 100*ks):
                Mseg = 8
        
            if (telapse > 50*ks and telapse <= 100*ks):
                Mseg = 7
        
            if (telapse > 25*ks and telapse <= 50*ks):
                Mseg = 4
        
            if (telapse > 15*ks and telapse <= 25*ks):
                Mseg = 2
        
            if (telapse <= 15*ks):
                Mseg = 1
                        
            #Reference band (background)
            hdulistref_bkg = fits.open(refbkgfile)
            dataref_bkg = hdulistref_bkg[1].data
            timerefbkg = dataref_bkg['TIME']  
            raterefbkg = dataref_bkg['RATE']
            errorrefbkg = dataref_bkg['ERROR']
        
            dateobs = hdulistref_bkg[0].header['DATE-OBS'].split("T")[0]
            dateend = hdulistref_bkg[0].header['DATE-END'].split("T")[0]
            timeobs = hdulistref_bkg[0].header['DATE-OBS'].split("T")[1]
            timeend = hdulistref_bkg[0].header['DATE-END'].split("T")[1]
            raobj = hdulistref_bkg[0].header['RA_OBJ']
            decobj = hdulistref_bkg[0].header['DEC_OBJ']
            telescope = hdulistref_bkg[0].header['TELESCOP']
            filterobs = hdulistref_bkg[0].header['FILTER']
            inst = hdulistref_bkg[0].header['INSTRUME']
            tstartRbkg = hdulistref[2].data['START']
            tstopRbkg = hdulistref[2].data['STOP']
        
            timeref = np.array(timeref)
            rateref = np.array(rateref)
            errorref = np.array(errorref)
            raterefbkg = np.array(raterefbkg)
            errorrefbkg = np.array(errorrefbkg)
                                                                    
            hdu3 = fits.open(complcfile)    
            timecomp = hdu3[1].data['TIME']
            ratecomp = hdu3[1].data['RATE']
            errorcomp = hdu3[1].data['ERROR']
            obsidcomp = hdu3[0].header['OBS_ID']
            
            hdu4 = fits.open(compbkgfile)    
            ratecompbkg = hdu4[1].data['RATE']
            errorcompbkg = hdu4[1].data['ERROR']
            tstartC = hdu3[2].data['START']
            tstopC = hdu3[2].data['STOP']
            
            #Remove BTIs and NANs from reference band and comparison band
            arraysR = np.transpose(np.column_stack((rateref,errorref,\
                      raterefbkg,errorrefbkg,ratecomp,errorcomp,\
                      ratecompbkg,errorcompbkg,timeref)))
            arraysR, arraysN = remove_nans(arraysR,tstartR,tstopR) 
            timeref -= timeref[0]
            timecomp = timeref
            telapse = timecomp[-1] - timecomp[0]
            
            #Concatenate LC from a single observation horizontally 
            #(to remove NANs)
            rateref,errorref,raterefbkg,\
            errorrefbkg,ratecomp,errorcomp,ratecompbkg,errorcompbkg,\
            timeref = arraysN
            bsizeref = timeref[1]-timeref[0]
            timeref = np.arange(0,len(rateref),1)*bsizeref
            timecomp = timeref   
                                    
            #Concatenate LCs from different observations
            for k3 in range(len(rateref)):
                
                windowcomb.append(windowref[k3])
                reflccomb.append(rateref[k3])
                errreflccomb.append(errorref[k3])
                complccomb.append(ratecomp[k3])
                errcomplccomb.append(errorcomp[k3])
                
                reflcbkgcomb.append(raterefbkg[k3])
                errreflcbkgcomb.append(errorrefbkg[k3])
                complcbkgcomb.append(ratecompbkg[k3])
                errcomplcbkgcomb.append(errorcompbkg[k3])
                
    if(len(reflccomb)>0):
        
        windowcomb = np.array(windowcomb)
        reflccomb = np.array(reflccomb)
        errreflccomb = np.array(errreflccomb)
        reflcbkgcomb = np.array(reflcbkgcomb)
        errreflcbkgcomb = np.array(errreflcbkgcomb)
        complccomb = np.array(complccomb)
        errcomplccomb = np.array(errcomplccomb)
        complcbkgcomb = np.array(complcbkgcomb)
        errcomplcbkgcomb = np.array(errcomplcbkgcomb)
        
        complccomb = complccomb[reflccomb>0]
        errcomplccomb = errcomplccomb[reflccomb>0]
        errreflccomb = errreflccomb[reflccomb>0]
        complcbkgcomb = complcbkgcomb[reflccomb>0]
        errcomplcbkgcomb = errcomplcbkgcomb[reflccomb>0]
        reflcbkgcomb = reflcbkgcomb[reflccomb>0]
        errreflcbkgcomb = errreflcbkgcomb[reflccomb>0]
        windowcomb = windowcomb[reflccomb>0]
        reflccomb = reflccomb[reflccomb>0]
        timecomb = np.arange(0,len(reflccomb),1)*bsize
        
        #Unfold spectrum using instrumental response      
        rmffile = "EPN_" + str(ObsId) + "_1.rmf"
        ancrfile = "EPN_" + str(ObsId) + "_1.arf"
        specfile = "epn_spec1" + "_grp_" + str(ObsId) + ".fits"
        infile = "covflux_" + str(ObsId) + ".dat"
        outfile = "covspec_" + str(ObsId) + ".pha"
                
        hdulist2 = fits.open(specfile)
        header2 = hdulist2[1].header
        backscal = header2['BACKSCAL']
        corrscal = header2['CORRSCAL']
        areascal = header2['AREASCAL']
        backfile = "NONE"
            
        energymin = float(hdu3[1].header['CHANMIN'])/1000.
        energymax = float(hdu3[1].header['CHANMAX'])/1000.
        engy = 0.5*(energymin+energymax)
        dengy = 0.5*(energymax-energymin)
        
        energies.append(engy)
        denergies.append(dengy)
                                        
        # #Plot LCs
        # laben = str(float(energymin)) + "-" + str(float(energymax))
        # labelsrccomp = "Comparison band LC: " + laben + " keV"    
        
        # plt.errorbar(timecomb/ks,reflccomb,yerr=errreflccomb,fmt='k-')
        # plt.plot(timecomb/ks,windowcomb,'g-')
        # plt.errorbar(timecomb/ks,complccomb,yerr=errcomplccomb,\
        #              label=labelsrccomp,fmt='r-')
        # plt.tick_params(axis='both', which='major', labelsize=14)
        # plt.legend(loc="best")
        # plt.title("XMM-Newton (EPIC-PN) lightcurves")
        # plt.ylabel("Count rate [s$^{-1}$]",fontsize=14)
        # plt.xlabel("Time [ks]",fontsize=14)
        # plt.show()
                                                        
        # #Compare CPSDs
        # countcomp = ratecomp*bsize
        # cerrorcomp = errorcomp*bsize
        # countref = rateref*bsize
        # cerrorref = errorref*bsize
        # countcompbkg = ratecompbkg*bsize
        # countrefbkg = raterefbkg*bsize
    
        # LcComp = Lightcurve(timecomp,countcomp,error=cerrorcomp,\
        #                     dt=bsize)
        # LcRef = Lightcurve(timecomp,countref,error=cerrorref,\
        #                    dt=bsize)
        # evcomp = EventList.from_lc(LcComp)   
        # evref = EventList.from_lc(LcRef)
                        
        # cpsdfrq,cpsd,cspderr =\
        # cross_spectrum_stingray(evref,evcomp,Mseg,bsize,normpow,fminb,fmaxb)
                
        # fxy, cpxyav, cpxyaverr = covariance_spectrum(complccomb,\
        #                          errcomplccomb,reflccomb,errreflccomb,\
        #                          complcbkgcomb,reflcbkgcomb,\
        #                          Mseg,bfactor,bsize,plot,\
        #                          statpow,fminb,fmaxb,windowref)
        
        # plt.figure()
        # plt.subplot(211)
        # plt.errorbar(cpsdfrq,np.real(cpsd),\
        #              yerr=abs(np.real(cspderr)),fmt='g.')
        # plt.errorbar(fxy,np.real(cpxyav),\
        #              yerr=abs(np.real(cpxyaverr)),fmt='m.')
        # plt.subplot(212)
        # plt.errorbar(cpsdfrq,np.imag(cpsd),\
        #              yerr=abs(np.imag(cspderr)),fmt='g.')
        # plt.errorbar(fxy,np.imag(cpxyav),\
        #              yerr=abs(np.imag(cpxyaverr)),fmt='m.')
        # plt.show()

        # Compute covariance
        # Time domain
        intcovtd,intcoverrtd=\
        covariance_time_domain(ratecomp,errorcomp,rateref,errorref,\
                               Mseg)
        covtd.append(intcovtd)
        dcovtd.append(intcoverrtd)
        
        # Frequency domain    
        windowcomb = np.array(windowcomb)          
        fcov,intcov,intcoverr,avgcpcov,avgcperrcov =\
        covariance_spectrum(complccomb,errcomplccomb,\
                            reflccomb,errreflccomb,complcbkgcomb,\
                            reflcbkgcomb,Mseg,bfactor,bsize,plot,\
                            statpow,fminb,fmaxb,windowcomb)
        cov.append(intcov)
        dcov.append(intcoverr)
            
cov = np.array(cov)
dcov = np.array(dcov)
covtd = np.array(covtd)
dcovtd = np.array(dcovtd)

plt.figure()
plt.subplot(211)
plt.errorbar(covspecEst,covspecst,yerr=covspecerrst,fmt='r.',\
             label="Method 1")
plt.errorbar(energies,cov,yerr=dcov,fmt='b.',label="Method 2")
plt.subplot(212)
plt.errorbar(energies,covtd,yerr=dcovtd,fmt='g.',label="Method 3")
plt.legend(loc="best")
plt.show()

# Convert to XSPEC readable format using response file
hdulist = fits.open(rmffile)
data = hdulist[2].data
channel = data['CHANNEL']
EMIN = data['E_MIN']
EMAX = data['E_MAX']
ndetchans = len(channel)
chans = np.arange(1,ndetchans+1,1)

if(np.sum(covspecst)>0 and np.sum(cov)>0):
        
    fluxes = np.zeros(len(chans))
    dfluxes = np.zeros(len(chans))
    
    for j3st in range(len(cov)):
        for k3st in range(len(chans)):
            if(0.5*(EMIN[k3st]+EMIN[k3st])>=energies[j3st]):
                fluxes[chans[k3st]] = cov[j3st]
                dfluxes[chans[k3st]] = dcov[j3st]
                break

    # for j3st in range(len(covspecst)):
    #     for k3st in range(len(chans)):
    #         if(0.5*(EMIN[k3st]+EMIN[k3st])>=covspecEst[j3st]):
    #             fluxes[chans[k3st]] = covspecst[j3st]
    #             dfluxes[chans[k3st]] = covspecerrst[j3st]
    #             break
        
    Qcov = np.column_stack((chans,fluxes,dfluxes))
    np.savetxt(infilecov,Qcov,fmt='%i %s %s',delimiter='   ')
                
    comm_unfold = "ascii2pha infile=" + infilecov + " outfile=" + outfilecov +\
    " chanpres=yes dtype=2 rows=- qerror=yes tlmin=1 detchans=" +\
    str(ndetchans) + " telescope=" + str(telescope) +\
    " instrume=" + str(inst) + " detnam=EPIC-PN" +\
    " filter=" + str(filterobs) + " phaversn=1.1.0 " +\
    "exposure=" + str(telapse/Mseg) + " backscal=" + str(backscal) +\
    " backfile=" + backfile + " corrscal=" + str(corrscal) +\
    " corrfile=NONE areascal=" + str(areascal) +\
    " ancrfile=" + ancrfile + " respfile=" + rmffile +\
    " date_obs=" + str(dateobs) + " time_obs=" + str(timeobs) +\
    " date_end=" + str(dateend) + " time_end=" + str(timeend) +\
    " ra_obj=" + str(raobj) + " dec_obj=" + str(decobj) +\
    " equinox=2000.0 hduclas2=TOTAL chantype=PI clobber=yes"
    os.system(comm_unfold)
    
    #Group spectrum using ftgrouppha
    comm_group = "ftgrouppha infile=" + outfilecov + " backfile=" +\
                  backfile + " respfile=" + rmffile +\
                  " outfile=" + groupfilecov +\
                  " grouptype=snmin groupscale=" +\
                  str(groupscale) + " minchannel=-1 maxchannel=-1"
    os.system(comm_group)



    

        
    
    
    
                                
        
    
