#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 23:54:44 2023

@author: erc_magnesia_raj
"""
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import glob, os
from scipy import fft
import warnings
from scipy import integrate
from stingray import Lightcurve
from scipy import integrate, signal
import subprocess
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
            fwn = fft.fftfreq(len(windowtemp),d=Dt)
            
            # remx,Xn = signal.deconvolve(Xn,Wn)
            # remy,Yn = signal.deconvolve(Yn,Wn)
                        
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
                Msegnew += 1
    
    #Average ref band and comp band noise powers
    Mseg = Msegnew
    Pnoise /= Mseg
    Prefnoise /= Mseg

    freqs = np.array(freqs)
    Pxn = np.array(Pxn)
    dPxn = np.array(dPxn)
    Pyn = np.array(Pyn)
    dPyn = np.array(dPyn)
    Cxyn = np.array(Cxyn)
    dCxyn = np.array(dCxyn)
                
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
        
        Karr = np.array(Karr)
        avgPxerr = np.array(avgPxerr)
        avgPyerr = np.array(avgPyerr)
        avgCxyerr = np.array(avgCxyerr)
        avgPx = np.array(avgPx)
        avgPy = np.array(avgPy)
        avgCxy = np.array(avgCxy)
        freqx = np.array(freqx)

        # Averaged number of samples
        nsamples = Mseg*Karr
                                    
        # Noise level of CPSD
        nbias = ((avgPx-Pnoise)*Prefnoise + (avgPy-Prefnoise)*Pnoise +\
                (Pnoise*Prefnoise))/nsamples
                            
        # Compute CPSD amplitude from complex-valued cross spectrum
        Cxyamp = (np.real(avgCxy))**2 + (np.imag(avgCxy))**2 - nbias
        avcxyreal = np.real(avgCxy)
        avcxyrealerr = np.real(avgCxyerr)
        avcxyimag = np.imag(avgCxy)
        avcxyimagerr = np.imag(avgCxyerr)
        
        dnbias = ((avgPxerr**2)*(Prefnoise**2) +\
                  (avgPyerr**2)*(Pnoise**2))/(nsamples**2)
            
        Cxyamperr = 4*((avcxyreal**2)*(avcxyrealerr**2) +\
                     (avcxyimag**2)*(avcxyimagerr**2)) + dnbias
            
        # Raw Coherence
        coherence = Cxyamp/((avgPx)*(avgPy))
        intcoherence = Cxyamp/((avgPx-Pnoise)*(avgPy-Prefnoise))
                        
        # # Raw uncertainty on coherence
        # dcoherence = (avgPxerr)*((-Cxyamp*avgPx**-2)/(avgPy))**2 +\
        #              (avgPyerr)*((-Cxyamp*avgPy**-2)/(avgPx))**2 +\
        #              (Cxyamperr**2)/(avgPx*avgPy)**2
        

        # Statistical uncertainty on raw coherence
        dcoherence = ((2.0/(nsamples))**(0.5))*(1 - intcoherence**2)/\
                     (abs(intcoherence))
        coherence = np.sqrt(coherence)
        dcoherence = 0.5*(dcoherence)/(coherence)
        
        # plt.plot(freqx,coherence,'k.')
        # plt.show()
        
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
        coherence = coherence[freqx>fbmin]
        dcoherence = dcoherence[freqx>fbmin]
        avgCxy = avgCxy[freqx>fbmin]
        avgCxyerr = avgCxyerr[freqx>fbmin]
        freqxfilt = freqx[freqx>fbmin]
        covariance = covariance[freqxfilt<fbmax]
        covariance_err = covariance_err[freqxfilt<fbmax]
        coherence = coherence[freqxfilt<fbmax]
        dcoherence = dcoherence[freqxfilt<fbmax]
        avgCxy = avgCxy[freqxfilt<fbmax]
        avgCxyerr = avgCxyerr[freqxfilt<fbmax]
        freqxfilt = freqxfilt[freqxfilt<fbmax]
                
        # Remove NANs
        isnancov = np.isnan(covariance)
        covariance = covariance[isnancov==False]
        coherence = coherence[isnancov==False]
        dcoherence = dcoherence[isnancov==False]
        covariance_err = covariance_err[isnancov==False]
        freqxfilt = freqxfilt[isnancov==False]
        
        isnancov = np.isnan(covariance_err)
        covariance = covariance[isnancov==False]
        covariance_err = covariance_err[isnancov==False]
        coherence = coherence[isnancov==False]
        dcoherence = dcoherence[isnancov==False]
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
        
    return mean_covariance, err_mcov

#Comparison with stingray
from stingray.varenergyspectrum import CovarianceSpectrum
from stingray import EventList
from stingray import AveragedPowerspectrum, AveragedCrossspectrum
    
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
    rf_band = [int(refemin),int(refemax)]      
    telapse_ref = eventsst_ref.time[-1]-eventsst_ref.time[0]         
    segsize_ref = telapse_ref/Msegstref
        
    for m3 in range(len(egrid)):
        egrid[m3] = int(egrid[m3])
            
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
    
    return covspecspt,covspecspterr
    

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
        
    
############### Generate covariance spectrum with stingray ###################
ks = 1000

#Energy grid
energies = [0.5,0.7,1.0,1.3,2.0,3.0,4.0,6.0,8.0] 
energies = np.array(energies)
energies = 1000*energies
eminst = energies[0:-1]
emaxst = energies[1:]

#Reference band
refemin = 1.0*1000
refemax = 1.3*1000

#PSD parameters 
fmi = 0.03125e-3
fmx = 0.25e-3
Mseg = 0
ndetchans = 1024
groupscale = 1
binsize = 2000.0
normcov = "abs"

for evfile in glob.glob("acisf13813_repro_evt2_bary_filt5.fits"):
    
    hdulist = fits.open(evfile)
    hdr = hdulist[1].header
        
    telapse = hdr['ONTIME']
    obsid = hdr['OBS_ID']
    dateobs = hdr['DATE-OBS'].split("T")[0]
    dateend = hdr['DATE-END'].split("T")[0]
    timeobs = hdr['DATE-OBS'].split("T")[1]
    timeend = hdr['DATE-END'].split("T")[1]
    raobj = hdr['RA_BARY']
    decobj = hdr['DEC_BARY']
    telescope = hdr['TELESCOP']
    filterobs = 'NONE'
    inst = hdr['INSTRUME']
                
    ##Choose Mseg depending on exposure time
    if (telapse > 100*ks):
        Mseg = 5
        
    if (telapse > 50*ks and telapse <= 100*ks):
        Mseg = 8
    
    if (telapse > 25*ks and telapse <= 50*ks):
        Mseg = 5
    
    if (telapse > 15*ks and telapse <= 25*ks):
        Mseg = 3
    
    if (telapse <= 15*ks):
        Mseg = 2
    
    covst,dcovst = covariance_spectrum_stingray(evfile,Mseg,\
                   binsize,fmi,fmx,refemin,refemax,energies,normcov)   
    
    covst = np.array(covst)
    dcovst = np.array(dcovst)
            
    rmffile = "acis_13813_5_spec.rmf"
    ancrfile = "acis_13813_5_spec.arf"

    # Convert to XSPEC readable format using response file
    hdulist = fits.open(rmffile)
    data = hdulist[2].data
    channel = data['CHANNEL']
    EMINC = data['E_MIN']
    EMAXC = data['E_MAX']
    
    chans = np.arange(1,ndetchans+1,1)
    fluxesst = np.zeros(len(chans))
    dfluxesst = np.zeros(len(chans))
                        
    for j3st in range(len(eminst)):
        
        emaxst[j3st] /= 1000.0
        
        for k3st in range(len(channel)):
                            
            if(0.5*(EMINC[k3st]+EMAXC[k3st])>=emaxst[j3st]):
                fluxesst[channel[k3st]] = covst[j3st]
                dfluxesst[channel[k3st]] = dcovst[j3st]
                break
        
    infile = "covflux_stingray_13813_5.dat"
    Q = np.column_stack((chans,fluxesst,dfluxesst))
    np.savetxt(infile,Q,fmt='%i %s %s',delimiter='   ')
    
    #Unfold spectrum using instrumental response
    outfile = "covspec_stingray_13813_5.pha"
    specfile = "acis_13813_5_spec_grp.pi"
    
    hdulist = fits.open(specfile)
    header = hdulist[1].header
    backscal = header['BACKSCAL']
    corrscal = header['CORRSCAL']
    areascal = header['AREASCAL']
    backfile = "NONE"
        
    comm_unfold = "ascii2pha infile=" + infile + " outfile=" + outfile +\
        " chanpres=yes dtype=2 rows=- qerror=yes tlmin=1 detchans=" +\
        str(ndetchans) + " telescope=" + str(telescope) +\
        " instrume=" + str(inst) + " detnam=ACIS-7" +\
        " filter=" + str(filterobs) + " phaversn=1.1.0 " +\
        "exposure=" + str(telapse) + " backscal=" + str(backscal) +\
        " backfile=" + backfile + " corrscal=" + str(corrscal) +\
        " corrfile=NONE areascal=" + str(areascal) +\
        " ancrfile=" + ancrfile + " respfile=" + rmffile +\
        " date_obs=" + str(dateobs) + " time_obs=" + str(timeobs) +\
        " date_end=" + str(dateend) + " time_end=" + str(timeend) +\
        " ra_obj=" + str(raobj) + " dec_obj=" + str(decobj) +\
        " equinox=2000.0 hduclas2=TOTAL chantype=PI clobber=yes"
        
    os.system(comm_unfold)
    
    #Group spectrum using ftgrouppha
    groupfile = "covspec_stingray_grouped_13813_5.pha"
    comm_group = "ftgrouppha infile=" + outfile + " backfile=" +\
                  backfile + " respfile=" + rmffile +\
                  " outfile=" + groupfile + " grouptype=min groupscale=" +\
                  str(groupscale) + " minchannel=-1 maxchannel=-1"
    os.system(comm_group)

################### Covariance spectrum using own code #####################
binsize = 2000.0
bfactor = 1.1
mseg = 0

#Energy grid for covariance
energies = [0.5,0.7,1.0,1.3,2.0,3.0,4.0,6.0,8.0] 
emin = energies[0:-1]
emax = energies[1:]
energies = np.array(energies)
plot = "no"
statpow = "NP"
key = "acisf23813_repro_evt2_bary5.fits"
energies_mean = []

# Reference band light-curve (source)
for evfile in sorted(glob.glob(key)):
                    
    cov,dcov = [[],[]]
    fvarc,fvarcerr = [[],[]]
    aCPSD,aCPSDerr = [[],[]]
        
    hdulist = fits.open(evfile)    
    hdr = hdulist[1].header
    telapse = hdr['ONTIME']
    obsid = hdr['OBS_ID']
    dateobs = hdr['DATE-OBS'].split("T")[0]
    dateend = hdr['DATE-END'].split("T")[0]
    timeobs = hdr['DATE-OBS'].split("T")[1]
    timeend = hdr['DATE-END'].split("T")[1]
    raobj = hdr['RA_TARG']
    decobj = hdr['DEC_TARG']
    telescope = hdr['TELESCOP']
    filterobs = 'NONE'
    inst = hdr['INSTRUME']
    detnam = hdr['DETNAM']
    tstart = hdulist[2].data['START']
    tstop = hdulist[2].data['STOP']
            
    # Reference band light-curve
    reflcfile = 'acisf13813_lc5_ref.fits'
    hdu2 = fits.open(reflcfile)   
    timeref = hdu2[1].data['TIME']
    rateref = hdu2[1].data['NET_RATE']
    errorref = hdu2[1].data['ERR_RATE']
    dt = timeref[1]-timeref[0]
    tstartR = hdu2[2].data['START']
    tstopR = hdu2[2].data['STOP']
                
    arraysW = np.transpose(np.column_stack((timeref,rateref)))
    windowref = rect_window(arraysW,tstartR,tstopR)
                        
    #Choose Mseg depending on exposure time
    if (telapse > 100*ks):
        mseg = 5
    
    if (telapse > 50*ks and telapse <= 100*ks):
        mseg = 8
    
    if (telapse > 25*ks and telapse <= 50*ks):
        mseg = 5
    
    if (telapse > 15*ks and telapse <= 25*ks):
        mseg = 3
    
    if (telapse <= 15*ks):
        mseg = 2

    Mseg = mseg
            
    #Reference band light-curve (background)
    bkgfile = 'acisf13813_lc5_ref_bkg.fits'
    hdulistbkg = fits.open(bkgfile)
    timerefbkg = hdulistbkg[1].data['TIME']
    bgrateref = hdulistbkg[1].data['COUNT_RATE']
    errbgrateref = hdulistbkg[1].data['COUNT_RATE_ERR']
    
    #Remove NANs in rate arrays
    isnanarr = np.isnan(rateref)
    timeref = timeref[isnanarr==False]
    rateref = rateref[isnanarr==False]
    errorref = errorref[isnanarr==False]
    bgrateref = bgrateref[isnanarr==False]
    errbgrateref = errbgrateref[isnanarr==False]
    
    isnanarr = np.isnan(errorref)
    timeref = timeref[isnanarr==False]
    rateref = rateref[isnanarr==False]
    errorref = errorref[isnanarr==False]
    bgrateref = bgrateref[isnanarr==False]
    errbgrateref = errbgrateref[isnanarr==False]
    
    rnewref,errnewref = [[],[]]
    
    for j in range(len(tstartR)):    
        
        rateref = rateref[timeref>=tstartR[j]]
        errorref = errorref[timeref>=tstartR[j]]
        timeref = timeref[timeref>=tstartR[j]]
        
        rateref = rateref[timeref<=tstopR[j]]
        errorref = errorref[timeref<=tstopR[j]]
        timeref = timeref[timeref<=tstopR[j]]
                
        for k in range(len(rateref)):
            rnewref.append(rateref[k])
            errnewref.append(errorref[k])
    
    rateref = np.array(rnewref)
    errorref = np.array(errnewref)
    
    #Create comparison band light-curves
    for l2 in range(len(energies)-1):
        
        #Deal with gaps in LC
        rnewcomp,errnewcomp,rnewcompbkg,errnewcompbkg,\
        windowcomb = [[],[],[],[],[]]     
        
        EMIN = energies[l2]
        EMAX = energies[l2+1]        
        energies_mean.append(0.5*(EMIN+EMAX))

        lcfile = "acisf13813_lc_5_" + str(l2+1) + ".fits"
        bkglcfile = "acisf13813_lc_5_" + str(l2+1) + "_bkg.fits"
        
        hdu3 = fits.open(lcfile)    
        timecomp = hdu3[1].data['TIME']
        ratecomp = hdu3[1].data['NET_RATE']
        errorcomp = hdu3[1].data['ERR_RATE']
        tstartC = hdu2[2].data['START']
        tstopC = hdu2[2].data['STOP']
        
        hdu4 = fits.open(bkglcfile)    
        ratecompbkg = hdu4[1].data['COUNT_RATE']
        errorcompbkg = hdu4[1].data['COUNT_RATE_ERR']
                
        arraysC = np.transpose(np.column_stack((timecomp,ratecomp)))
        windowC = rect_window(arraysC,tstartC,tstopC)
        
        for j2 in range(len(tstartC)):    
            
            ratecomp = ratecomp[timecomp>=tstartC[j2]]
            errorcomp = errorcomp[timecomp>=tstartC[j2]]
            ratecompbkg = ratecompbkg[timecomp>=tstartC[j2]]
            errorcompbkg = errorcompbkg[timecomp>=tstartC[j2]]
            timecomp = timecomp[timecomp>=tstartC[j2]]
            
            ratecomp = ratecomp[timecomp<=tstopC[j2]]
            errorcomp = errorcomp[timecomp<=tstopC[j2]]
            ratecompbkg = ratecompbkg[timecomp<=tstopC[j2]]
            errorcompbkg = errorcompbkg[timecomp<=tstopC[j2]]
            timecomp = timecomp[timecomp<=tstopC[j2]]
                
            for k2 in range(len(rateref)):
                windowcomb.append(windowC[k2])
                rnewcomp.append(ratecomp[k2])
                errnewcomp.append(errorcomp[k2])
                rnewcompbkg.append(ratecompbkg[k2])
                errnewcompbkg.append(errorcompbkg[k2])
        
        rateref = np.array(rnewref)
        errorref = np.array(errnewref)
        ratecomp = np.array(rnewcomp)
        errorcomp = np.array(errnewcomp)
        ratecompbkg = np.array(rnewcompbkg)
        errorcompbkg = np.array(errnewcompbkg)
        windowcomb = np.array(windowcomb)
        timeref = dt*np.arange(0,len(rnewref),1)
                                
        # plt.errorbar(timeref/ks,rateref,yerr=errorref,fmt='k-')
        # plt.errorbar(timeref/ks,ratecomp,yerr=errorcomp,fmt='b-')
        # plt.errorbar(timeref/ks,ratecompbkg,\
        #              yerr=errorcompbkg,fmt='g-')
        # # plt.plot(timeref/ks,windowcomb,'r-')
        # plt.tick_params(axis='both', which='major', labelsize=14)
        # plt.legend(loc="best")
        # plt.title("Chandra (ACIS-S) lightcurve")
        # plt.ylabel("Count rate [s$^{-1}$]",fontsize=14)
        # plt.xlabel("Time [ks]",fontsize=14)
        # plt.show()
        
        # Frequency domain    
        windowcomb = np.array(windowcomb)   
        intcov,intcoverr =\
        covariance_spectrum(ratecomp,errorcomp,\
                            rateref,errorref,ratecompbkg,\
                            errorcompbkg,Mseg,bfactor,binsize,plot,\
                            statpow,fmi,fmx,windowcomb)
        cov.append(intcov)
        dcov.append(intcoverr)
    
    cov = np.array(cov)
    dcov = np.array(dcov)
    cov *= 3
    dcov *= 3
    
    rmffile = "acis_13813_5_spec.rmf"
    ancrfile = "acis_13813_5_spec.arf"

    # Convert to XSPEC readable format using response file
    hdulist = fits.open(rmffile)
    data = hdulist[2].data
    channel = data['CHANNEL']
    EMIN = data['E_MIN']
    EMAX = data['E_MAX']

    chans = np.arange(1,ndetchans+1,1)
    fluxes = np.zeros(len(chans))
    dfluxes = np.zeros(len(chans))
        
    for j3 in range(len(emin)):
        for k3 in range(len(channel)):
                
            if(0.5*(EMIN[k3]+EMAX[k3])>=emax[j3]):
                fluxes[channel[k3]] = cov[j3]
                dfluxes[channel[k3]] = dcov[j3]
                break
        
    infile = "covflux_13813_5.dat"
    Q = np.column_stack((chans,fluxes,dfluxes))
    np.savetxt(infile,Q,fmt='%i %s %s',delimiter='   ')
    
    #Unfold spectrum using instrumental response
    outfile = "covspec_13813_5.pha"
    specfile = "acis_13813_5_spec_grp.pi"
    
    hdulist = fits.open(specfile)
    header = hdulist[1].header
    backscal = header['BACKSCAL']
    corrscal = header['CORRSCAL']
    areascal = header['AREASCAL']
    backfile = "NONE"
        
    comm_unfold = "ascii2pha infile=" + infile + " outfile=" + outfile +\
        " chanpres=yes dtype=2 rows=- qerror=yes tlmin=1 detchans=" +\
        str(ndetchans) + " telescope=" + str(telescope) +\
        " instrume=" + str(inst) + " detnam=ACIS-7" +\
        " filter=" + str(filterobs) + " phaversn=1.1.0 " +\
        "exposure=" + str(telapse) + " backscal=" + str(backscal) +\
        " backfile=" + backfile + " corrscal=" + str(corrscal) +\
        " corrfile=NONE areascal=" + str(areascal) +\
        " ancrfile=" + ancrfile + " respfile=" + rmffile +\
        " date_obs=" + str(dateobs) + " time_obs=" + str(timeobs) +\
        " date_end=" + str(dateend) + " time_end=" + str(timeend) +\
        " ra_obj=" + str(raobj) + " dec_obj=" + str(decobj) +\
        " equinox=2000.0 hduclas2=TOTAL chantype=PI clobber=yes"
    os.system(comm_unfold)
    
    #Group spectrum using ftgrouppha
    groupfile = "covspec_13813_5_grouped.pha"
    comm_group = "ftgrouppha infile=" + outfile + " backfile=" +\
                  backfile + " respfile=" + rmffile +\
                  " outfile=" + groupfile +\
                  " grouptype=min groupscale=" +\
                  str(groupscale) + " minchannel=-1 maxchannel=-1"
    os.system(comm_group)
