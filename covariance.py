import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import glob
import warnings
from stingray import Lightcurve
from stingray import AveragedCrossspectrum
import scipy
from scipy import fft
from scipy import integrate, signal
import os, subprocess
from kapteyn import kmpfit
from sklearn.gaussian_process.kernels import RBF, Matern,\
ExpSineSquared, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
import random
from scipy import stats

warnings.filterwarnings('ignore')
ks = 1000
day = 86400

#Function to model Averaged PSD
def plmod(pars, xdata):
    
    amplitude,alpha = pars
    
    Norder = 8
    ybessel = np.zeros(len(xdata))
    for korder in range(Norder):
        ybessel += scipy.special.jv(korder,xdata)
    ymod = amplitude - alpha*xdata + ybessel
           
    return ymod

#Residuals
def residuals(pars, data):
    
    xdata, ydata, ydataerr = data
    amplitude,alpha = pars
    resid = (ydata - plmod(pars,xdata))/ydataerr
    
    return resid 

#Timmer & Koenig method to generate fake LC
def drawsamp(psdomega):
    
    r1 = np.random.normal(0.0,scale=np.sqrt(abs(0.5*psdomega)))
    r2 = np.random.normal(0.0,scale=np.sqrt(abs(0.5*psdomega)))
        
    compnumpos = complex(r1,r2)
    compnumneg = np.conj(compnumpos)
    
    return compnumpos,compnumneg

#Estimation of random noise contribution to lag energy spectrum
def genpsd(bin_time_mcmc,tdur,freqmin,freqmax,psdomega,murate1):
                 
    Nomega = len(psdomega)
    complexfft1pos = np.zeros(int(len(psdomega))).astype(complex)    
    complexfft1neg = np.zeros(int(len(psdomega))).astype(complex)    

    for irand in range(Nomega):
                        
        compnumber_pos1,compnumber_neg1 = drawsamp(psdomega[irand])
        complexfft1pos[irand] = compnumber_pos1
        complexfft1neg[irand] = compnumber_neg1

        if(irand==int(Nomega-1)):
            complexfft1neg[irand] = np.real(complexfft1neg[irand])
            
    complexfft1neg = np.flip(complexfft1neg)
    complexfft1pos = np.insert(complexfft1pos,0,complex(murate1))
    complexfft1 = np.hstack((complexfft1pos,complexfft1neg))
            
    #Artificial LCs generated from PSDs (Timmer & Köenig 1995)
    counts = np.fft.ifft(complexfft1,n=len(complexfft1)) +\
             0.5*complexfft1[0]
    counts = np.real(counts)
    
    #Add a floor
    cmin = abs(np.min(counts))
    counts -= cmin
    for gl in range(len(counts)):
        if(counts[gl]<0):
            counts[gl] = 0
    cmean = np.mean(counts)
    counts -= cmean
    
    error = np.sqrt(counts)  
    times = bin_time_mcmc*(np.arange(0,len(counts)-1,1))   
        
    return times, counts, error

def psdmod(tlcpsd,lcpsd,lcerrpsd,reflcpsd,reflcerrpsd,lcbkgpsd,refbkgpsd,\
           Msegpsd,bfactorpsd,Dtpsd,statspsd,windowpsd,rmbtpsd,returnsim):
           
    #Compute noise level depending on whether the counting statistics
    #are Poissonian or not 
    fnyqpsd = 0.5*(Dtpsd**-1)
    Pnoisepsd,Prefnoisepsd,Msegnewpsd,Msegpsd = 0,0,0,1
    freqspsd,Pxnpsd,Pynpsd,Cxynpsd,dPxnpsd,dPynpsd,dCxynpsd =\
    [[],[],[],[],[],[],[]]
                
    #Average power spectrum and cross spectrum over M segments
    for kzpsd in range(Msegpsd):
        
        #Split LC into M equal segments
        divpsd = int(len(reflcpsd)/Msegpsd)
        lctemppsd = lcpsd[kzpsd*divpsd:(kzpsd+1)*divpsd]
        lctemperrpsd = lcerrpsd[kzpsd*divpsd:(kzpsd+1)*divpsd]
        reflctemppsd = reflcpsd[kzpsd*divpsd:(kzpsd+1)*divpsd]
        reflctemperrpsd = reflcerrpsd[kzpsd*divpsd:(kzpsd+1)*divpsd]
        lcbkgtemppsd = lcbkgpsd[kzpsd*divpsd:(kzpsd+1)*divpsd]
        refbkgtemppsd = refbkgpsd[kzpsd*divpsd:(kzpsd+1)*divpsd]
        windowtemppsd = windowpsd[kzpsd*divpsd:(kzpsd+1)*divpsd]
                        
        #Ambient noise level in PSD
        if(statspsd=="Poissonian"):
            
            Pnoisepsd += (2*(np.mean(lctemppsd) + np.mean(lcbkgtemppsd))/\
                         (np.mean(lctemppsd))**2)
            Prefnoisepsd += (2*(np.mean(reflctemppsd) +\
                             np.mean(refbkgtemppsd))/\
                            (np.mean(reflctemppsd))**2)

        if(statspsd!="Poissonian"):
            
            errsqpsd = 0
            errrefsqpsd = 0
            
            for lpsd in range(len(lctemperrpsd)):
                errsqpsd += lctemperrpsd[lpsd]**2
                errrefsqpsd += reflctemperrpsd[lpsd]**2
                
            errsqpsd/=len(lctemperrpsd)
            errrefsqpsd/=len(reflctemperrpsd)
            Pnoisepsd += errsqpsd/(fnyqpsd*(np.mean(lctemppsd))**2)
            Prefnoisepsd += errrefsqpsd/(fnyqpsd*(np.mean(reflctemppsd))**2) 
        
        if(np.sum(lctemppsd)>0):
            
            lctemppsd = np.array(lctemppsd)
            lctemperrpsd = np.array(lctemperrpsd)
                                                            
            #FFT of comparison-band LC
            Xnpsd =\
            0.5*(fft.fft(lctemppsd+lctemperrpsd)+\
                 fft.fft(lctemppsd-lctemperrpsd)) 
            Xnerrpsd = 0.5*(fft.fft(lctemppsd+lctemperrpsd)-\
                         fft.fft(lctemppsd-lctemperrpsd))
            Xnconjpsd = 0.5*(np.conj(Xnpsd+Xnerrpsd)+\
                             np.conj(Xnpsd-Xnerrpsd))
            Xnconjerrpsd = 0.5*(np.conj(Xnpsd+Xnerrpsd)-\
                                np.conj(Xnpsd-Xnerrpsd))
            fxnpsd = fft.fftfreq(len(lctemppsd),d=Dtpsd)
                                    
            #FFT of reference-band LC
            Ynpsd = 0.5*(fft.fft(reflctemppsd+reflctemperrpsd)+\
                         fft.fft(reflctemppsd-reflctemperrpsd))
            Ynerrpsd = 0.5*(fft.fft(reflctemppsd+reflctemperrpsd)-\
                    fft.fft(reflctemppsd-reflctemperrpsd))
            Ynconjpsd = 0.5*(np.conj(Ynpsd+Ynerrpsd)+\
                             np.conj(Ynpsd-Ynerrpsd))
            Ynconjerrpsd = 0.5*(np.conj(Ynpsd+Ynerrpsd)-\
                                np.conj(Ynpsd-Ynerrpsd))
            fynpsd = fft.fftfreq(len(reflctemppsd),d=Dtpsd)
            
            # FFT of window function
            Wnpsd = fft.fft(windowtemppsd)
            Wnconjpsd = np.conj(Wnpsd)
            
            if(rmbtpsd=="yes"):
                
                #Remove beats due to window
                Xnpsd /= Wnpsd
                Ynpsd /= Wnpsd
                Xnconjpsd /= Wnconjpsd
                Ynconjpsd /= Wnconjpsd
                        
            Xnpsd = Xnpsd[fxnpsd>0] 
            Xnerrpsd = Xnerrpsd[fxnpsd>0]
            Xnconjpsd = Xnconjpsd[fxnpsd>0]
            Xnconjerrpsd = Xnconjerrpsd[fxnpsd>0]
            Wnpsd = Wnpsd[fxnpsd>0]
            Wnconjpsd = Wnconjpsd[fxnpsd>0]
            Ynpsd = Ynpsd[fxnpsd>0]
            Ynerrpsd = Ynerrpsd[fxnpsd>0]
            Ynconjpsd = Ynconjpsd[fxnpsd>0]
            Ynconjerrpsd = Ynconjerrpsd[fxnpsd>0]
            fynpsd = fynpsd[fxnpsd>0]
            fxnpsd = fxnpsd[fxnpsd>0]
                                                                                                    
            # #Compute PSD and CPSD with 
            # rms-squared normalisation for each segment
            normpsdxpsd =\
            (2.0*Dtpsd)/((len(lctemppsd))*(np.mean(lctemppsd))**2)
            normpsdypsd =\
            (2.0*Dtpsd)/((len(reflctemppsd))*(np.mean(reflctemppsd))**2)
            normcrosspsd =\
            (2.0*Dtpsd)/((len(lctemppsd))*(np.mean(lctemppsd))*\
            (np.mean(reflctemppsd)))
            
            #PSD
            Psdxpsd = normpsdxpsd*Xnconjpsd*Xnpsd
            dPsdxpsd = normpsdxpsd*(Xnconjerrpsd*Xnpsd + Xnconjpsd*Xnerrpsd)
            Psdypsd = normpsdypsd*Ynconjpsd*Ynpsd
            dPsdypsd = normpsdypsd*(Ynconjerrpsd*Ynpsd +\
                                    Ynconjpsd*Ynerrpsd)
            Crossxypsd = normcrosspsd*Ynconjpsd*Xnpsd
            dCrossxypsd = normcrosspsd*(Ynconjerrpsd*Xnpsd +\
                                        Ynconjpsd*Xnerrpsd)
            PsdWpsdpsd = normpsdxpsd*Wnconjpsd*Wnpsd
                                    
            if(len(Crossxypsd)>0 and len(Psdxpsd)>0 and len(Psdypsd)>0):
                                
                # Append CPSD and PSDs for each segment to 
                # pass to functions for averaging and binning
                freqspsd.append(fxnpsd)
                Pxnpsd.append(Psdxpsd)
                Pynpsd.append(Psdypsd)
                Cxynpsd.append(Crossxypsd)
    
    freqspsd = np.array(freqspsd)
    Pxnpsd = np.array(Pxnpsd)
    dPxnpsd = np.array(dPxnpsd)
    Pynpsd = np.array(Pynpsd)
    dPynpsd = np.array(dPynpsd)
    Cxynpsd = np.array(Cxynpsd)
    dCxynpsd = np.array(dCxynpsd)
    Nfmodrefpsd = 0
                        
    if(len(Cxynpsd)>0):
                        
        # Average PSDs and CPSD over M segements
        freqxpsd,Pxavgpsd,dPxavgpsd,Pyavgpsd,dPyavgpsd,\
        Cxyavgpsd,dCxyavgpsd =\
        Pbin(Msegpsd,freqspsd,Pxnpsd,Pynpsd,Cxynpsd)
        
        avgPxpsd = Pxavgpsd
        avgPypsd = Pyavgpsd
        avgCxypsd = Cxyavgpsd
        avgPxerrpsd = dPxavgpsd
        avgPyerrpsd = dPyavgpsd
        avgCxyerrpsd = dCxyavgpsd
        dfreqxpsd = freqxpsd[1]-freqxpsd[0]
        Karrpsd = np.ones(len(Pxavgpsd))

        # Implement frequency dependent binning of averaged PSDs and CPSD
        if(bfactorpsd>1):
            freqxpsd,avgPxpsd,avgPypsd,avgCxypsd,avgPxerrpsd,\
            avgPyerrpsd,avgCxyerrpsd,Karrpsd =\
            fbin(bfactorpsd,freqxpsd,Pxavgpsd,Pyavgpsd,\
                 Cxyavgpsd,dPxavgpsd,dPyavgpsd,dCxyavgpsd)
        
        Karrpsd = np.array(Karrpsd)
        avgPxerrpsd = np.array(avgPxerrpsd)
        avgPyerrpsd = np.array(avgPyerrpsd)
        avgCxyerrpsd = np.array(avgCxyerrpsd)
        avgPxpsd = np.array(avgPxpsd)
        avgPypsd = np.array(avgPypsd)
        avgCxypsd = np.array(avgCxypsd)
        freqxpsd = np.array(freqxpsd)
        
        lgfreqxpsd = np.log(freqxpsd)
        lgavgPxpsd = np.log(avgPxpsd)
        lgavgPypsd = np.log(avgPypsd)
        lgavgPxerrpsd = abs(avgPxerrpsd/avgPxpsd)
        lgavgPyerrpsd = abs(avgPyerrpsd/avgPypsd)
        
        if(len(Cxynpsd)==0):
            
            fmodrefpsd = np.zeros(Nfmodrefpsd)
            pmodrefpsd = np.zeros(Nfmodrefpsd)
            fmodcomppsd = np.zeros(Nfmodrefpsd)
            pmodcomppsd = np.zeros(Nfmodrefpsd)
            freqxpsd = np.zeros(Nfmodrefpsd)        
            avgPxpsd = np.zeros(Nfmodrefpsd)
            avgPxerrpsd = np.zeros(Nfmodrefpsd)
            avgPypsd = np.zeros(Nfmodrefpsd)
            avgPyerrpsd = np.zeros(Nfmodrefpsd)
        
        if(returnsim=="yes"):
            
            #Primary kernel parameters (RBF)
            lscale = 20.0
            sigf = 10
            sign = 4.37e-4
            dim = 1
            lgfreqrefre = lgfreqxpsd.reshape(len(lgfreqxpsd),dim)
            lgfreqcompre = lgfreqxpsd.reshape(len(lgfreqxpsd),dim)
            kern = (sigf**2)*RBF(length_scale=lscale) +\
            WhiteKernel(noise_level=sign)
            
            gp = GaussianProcessRegressor(kernel=kern,alpha=1e-10,\
                 n_restarts_optimizer=200,normalize_y=True)
            gp.fit(lgfreqcompre,lgavgPypsd)
            scorecomp = gp.score(lgfreqcompre,lgavgPypsd)
            paramscomp = gp.kernel_
            
            #Best-fit prediction
            Npsfmod = int(0.5*len(tlcpsd))
            lgfreqypsdmod =\
            np.linspace(np.min(lgfreqxpsd),np.max(lgfreqxpsd),Npsfmod)
            
            lgfmodcomppsdre = lgfreqypsdmod.reshape(len(lgfreqypsdmod),dim)
            lgpmodcomppsd,\
            lgpmodcomppsderr = gp.predict(lgfmodcomppsdre,return_std=True)
                        
            fmodcomppsd = np.exp(lgfreqxpsd)
            pmodcomppsdinterp = np.exp(lgpmodcomppsd)   
            mulccomppsd = np.mean(lcpsd)
            Nfmodrefpsd = len(reflcpsd)

        # # Fit PSD 
        # amppsdinitc = np.mean(lgavgPypsd)
        # alphapsdinitc = 0.2
        # initparams = [amppsdinitc,alphapsdinitc]
        # fitobjc = kmpfit.Fitter(residuals=residuals,\
        #           data=(lgfreqxpsd,lgavgPypsd,lgavgPyerrpsd))
        # fitobjc.fit(params0=initparams)
        # chi2min = fitobjc.chi2_min
        # dof = fitobjc.dof
        # ampbestcomppsd = fitobjc.params[0]
        # ampbestcomperrpsd = fitobjc.xerror[0]
        # alphabestcomppsd = fitobjc.params[1]
        # alphabestcomperrpsd = fitobjc.xerror[1]
                
        # bestfitpars = [ampbestcomppsd,alphabestcomppsd]
        # lgfmodrefpsd = np.linspace(np.min(lgfreqxpsd),\
        #                            np.max(lgfreqxpsd),1000)
        # lgpmodrefpsd = plmod(bestfitpars,lgfreqypsdmod)
                
        plt.errorbar(lgfreqxpsd,lgavgPypsd,yerr=lgavgPyerrpsd,\
                     fmt='k.')
        plt.plot(lgfreqypsdmod,lgpmodcomppsd,'r-')
        plt.show()
                        
    return lgfreqxpsd,lgavgPypsd,lgavgPyerrpsd,fmodcomppsd,lgpmodcomppsd,\
           mulccomppsd
           
def remove_nans(arrays,tS,tE):
        
    tref = arrays[-1]   
    oldarrays_list,newarrays_list = [[],[]]
    deepmin, mind = -1e10, -100
        
    for qd in range(len(arrays)):
                                                                        
        #Remove NANs
        isnanarr = np.isnan(arrays[qd])
        for qd2 in range(len(arrays[qd])):
            if(isnanarr[qd2]=='True'):
                arrays[qd2] = deepmin
        
        #Ignore BTIs
        for qd3 in range(len(tS)-1):
            
            btiS = tE[qd3]
            btiE = tS[qd3+1]
                        
            for qd4 in range(len(tref)):
                
                if(tref[qd4]>=btiS and tref[qd4]<=btiE\
                   and qd!=len(arrays)-1):
                    
                    arrays[qd][qd4] = deepmin
        
        if(qd!=len(arrays)-1):
        
            newarray = arrays[qd][arrays[qd]>mind]
            newarrays_list.append(newarray)
            newtime = tref[arrays[qd]>mind]
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
    mu_sigerr_x*mu_sigerr_y)/(Msegs*Numpt*mu_sigxs_y))
                    
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

#Covariance spectrum (stingray)
def covariance_spectrum_stingray(evfile,\
                                 bwidth,fbmin,fbmax,\
                                 refemin,refemax,egrid,normalisation):
        
    eventsst_ref = EventList.read(evfile,"hea",\
                                  additional_columns=["DET_ID"])
    frq_interval = [fbmin,fbmax]
    rf_band = [refemin,refemax]      
    telapse_ref = eventsst_ref.time[-1]-eventsst_ref.time[0] 
    Msegstref = 0

    if (telapse_ref > 100*ks):
        Msegstref = 10

    if (telapse_ref > 50*ks and telapse_ref <= 100*ks):
        Msegstref = 8

    if (telapse_ref > 25*ks and telapse_ref <= 50*ks):
        Msegstref = 6

    if (telapse_ref > 15*ks and telapse_ref <= 25*ks):
        Msegstref = 3

    if (telapse_ref <= 15*ks):
        Msegstref = 1
     
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

#Estimate covariance spectrum in Fourier domain (Uttley et al. 2014)
def covariance_spectrum(tlc,lc,lcerr,reflc,reflcerr,lcbkg,refbkg,Mseg,\
                        bfactor,Dt,stats,fbmin,fbmax,window,rmbt):
                        
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
            
            lctemp = np.array(lctemp)
            lctemperr = np.array(lctemperr)
                                                            
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
            Wnconj = np.conj(Wn)
            
            if(rmbt=="yes"):
                
                #Remove beats due to window
                Xn /= Wn
                Yn /= Wn
                Xnconj /= Wnconj
                Ynconj /= Wnconj
                        
            Xn = Xn[fxn>0] 
            Xnerr = Xnerr[fxn>0]
            Xnconj = Xnconj[fxn>0]
            Xnconjerr = Xnconjerr[fxn>0]
            Yn = Yn[fyn>0]
            Ynerr = Ynerr[fyn>0]
            Ynconj = Ynconj[fyn>0]
            Ynconjerr = Ynconjerr[fyn>0]
            Wn = Wn[fxn>0]
            Wnconj = Wnconj[fxn>0]
            fxn = fxn[fxn>0]
            fyn = fyn[fyn>0]
                                                                                                    
            # #Compute PSD and CPSD with 
            # rms-squared normalisation for each segment
            normpsdx = (2.0*Dt)/((len(lctemp))*(np.mean(lctemp))**2)
            normpsdw = (2.0*Dt)/((len(windowtemp))*(np.mean(windowtemp))**2)
            normpsdy = (2.0*Dt)/((len(reflctemp))*(np.mean(reflctemp))**2)
            normcross = (2.0*Dt)/((len(lctemp))*(np.mean(lctemp))*\
                        (np.mean(reflctemp)))
            
            #PSD
            Psdw = normpsdw*Wnconj*Wn
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
        
        Karr = np.array(Karr)
        avgPxerr = np.array(avgPxerr)
        avgPyerr = np.array(avgPyerr)
        avgCxyerr = np.array(avgCxyerr)
        avgPx = np.array(avgPx)
        avgPy = np.array(avgPy)
        avgCxy = np.array(avgCxy)
        freqx = np.array(freqx)
        
        lgfreqx = np.log(freqx)
        lgavgPx = np.log(avgPx)
        lgavgPy = np.log(avgPy)
        lgavgPxerr = abs(avgPxerr/avgPx)
        lgavgPyerr = abs(avgPyerr/avgPy)

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
        
        # Statistical uncertainty on raw coherence
        dcoherence = ((2.0/(nsamples))**(0.5))*(1 - intcoherence**2)/\
                     (abs(intcoherence))
        coherence = np.sqrt(coherence)
        dcoherence = 0.5*(dcoherence)/(coherence)
        
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
    
    cspecamp = cspec
    cspecamperr = cspecerr
                
    return cspecfreq,cspecamp,cspecamperr

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

# ObsID(s)
obsidnum = ["0951800101"]

# Energy grid
Emin = 0.3
Emax = 12.0
refemin = Emin
refemax = Emax
Nenergies = 0
for kn in range(len(obsidnum)):
    for jn in sorted(glob.glob("epn*net*" + str(obsidnum[kn]) +\
                               "*_1_*ref*.lc")):
        Nenergies += 1

# PSD parameters 
bfactor = 1.1
fminb = [1e-4]
fmaxb = [5e-4]
Mseg = 1
fminb = np.array(fminb)
fmaxb = np.array(fmaxb)
statpow = "Poissonian" #NP or Poissonian
normpow = "abs" # Norm of PSD

# Decisions
iggaps = "yes"
plotpsd = "no"
genpsds = "no"
plotlc = "no"
rmnans = "no"
comparecpsd = "no"
removebt = "no"
plotcov = "no"

# Detector parameters
tthresh = (np.min(fminb))**-1
groupscale = 1

#LC segmentation
tmin = 0*ks
tmax = 200*ks
segmentlc = "yes"

for k3 in range(len(fminb)):
    
    energiesref,denergiesref = [[],[]]
    cov,dcov,covtd,dcovtd,fvarc,fvarcerr = [[],[],[],[],[],[]]
            
    for k in range(Nenergies-1):
                
        reflcbkgcomb,errreflcbkgcomb,complcbkgcomb,errcomplcbkgcomb,\
        timecombref,windowcomb,reflccomb,errreflccomb,complccomb,\
        errcomplccomb = [[],[],[],[],[],[],[],[],[],[]]
                                
        for k2 in range(len(obsidnum)):
                                                                
            for tempreflcfile in sorted(glob.glob("epn_net_obs" +\
                                        obsidnum[k2] + "*en4*ref.lc")):
                                                
                visnum = int(tempreflcfile.split("_")[3])
                ennum = str(k+1)
                                
                #Reference band (source)
                ObsId = tempreflcfile.split("obs")[1].split("_")[0]  
                                                        
                reflcfile = "epn_net_obs" + str(ObsId) + "_" + str(visnum) +\
                            "_" + "en" + str(ennum) + "_ref.lc"
                            
                refbkgfile = "epn_bkg_obs" + ObsId + "_" + str(visnum) +\
                             "_" + "en" + str(ennum) + "_ref.lc"
                             
                refevfile = "epn_net_obs" + ObsId + "_" + str(visnum) +\
                            "_" + "en" + str(ennum) + "_ref.lc"
                            
                complcfile = "epn_net_obs" + str(ObsId) + "_" + str(visnum) +\
                            "_" + "en" + str(ennum) + "_comp.lc"
                compbkgfile = "epn_net_obs" + str(ObsId) + "_" + str(visnum) +\
                            "_" + "en" + str(ennum) + "_comp.lc"
                compevfile = "epn_net_obs" + str(ObsId) + "_" + str(visnum) +\
                            "_" + "en" + str(ennum) + "_comp.lc"

                infilecov = "covflux" + str(k3+1) +\
                "_" + str(ObsId) + ".dat"
                outfilecov = "covspec" + str(k3+1) +\
                "_" + str(ObsId) + ".pha"
                groupfilecov = "covspec_grouped_" + str(k3+1) + "_" +\
                str(ObsId) + ".pha"
                                
                hdulistref = fits.open(reflcfile)
                dataref = hdulistref[1].data
                tstartR = hdulistref[2].data['START']
                tstopR = hdulistref[2].data['STOP']
                timeref = dataref['TIME']  
                rateref = dataref['RATE']
                errorref = dataref['ERROR']
                bsizeref = timeref[1]-timeref[0]
                telapse = timeref[-1]-timeref[0]

                murateref = np.mean(rateref)
                muerrorref = np.sum(errorref**2)/len(rateref)
                arraysW = np.transpose(np.column_stack((timeref,rateref)))
                windowref = rect_window(arraysW,tstartR,tstopR)
                                                                                                                            
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
                                                                                                        
                hducomp = fits.open(complcfile)    
                timecomp = hducomp[1].data['TIME']
                ratecomp = hducomp[1].data['RATE']
                errorcomp = hducomp[1].data['ERROR']
                obsidcomp = hducomp[0].header['OBS_ID']
                
                muratecomp = np.mean(ratecomp)
                muerrorcomp = np.sum(errorcomp**2)/len(ratecomp)
                
                hdubkgcomp = fits.open(compbkgfile)    
                ratecompbkg = hdubkgcomp[1].data['RATE']
                errorcompbkg = hdubkgcomp[1].data['ERROR']
                tstartC = hducomp[2].data['START']
                tstopC = hducomp[2].data['STOP']
                mjdref = hducomp[2].header['MJDREF']
                tstartref = hducomp[1].header['TSTART']
                keyen = hducomp[1].header['DSVAL6']
                if(keyen=='TABLE'):
                    keyen = hducomp[1].header['DSVAL5']
                energymin = float(keyen.split(":")[0])/1000.
                energymax = float(keyen.split(":")[1])/1000.
                energyminref = energymin
                energymaxref = energymax     
                                
                energiesref.append(energymin)
                denergiesref.append(energymax)
    
                if (telapse > 100*ks):
                    Mseg = 12
                    
                if (telapse > 75*ks and telapse <= 100*ks):
                    Mseg = 10

                if (telapse > 50*ks and telapse <= 75*ks):
                    Mseg = 8
            
                if (telapse > 25*ks and telapse <= 50*ks):
                    Mseg = 6
            
                if (telapse > 15*ks and telapse <= 25*ks):
                    Mseg = 3
            
                if (telapse <= 15*ks):
                    Mseg = 1
                                
                if(rmnans=="yes"):

                    #Remove BTIs and NANs from reference band and comparison 
                    #band
                    arraysR =\
                    np.transpose(np.column_stack((rateref,errorref,\
                    raterefbkg,errorrefbkg,ratecomp,errorcomp,\
                    ratecompbkg,errorcompbkg,timeref)))
                    arraysR, arraysN = remove_nans(arraysR,tstartR,tstopR) 
                    rateref,errorref,raterefbkg,\
                    errorrefbkg,ratecomp,errorcomp,ratecompbkg,errorcompbkg,\
                    timeref = arraysN
                    bsizeref = timeref[1]-timeref[0]
                                                
                    #Concatenate LC from a single observation horizontally 
                    #(remove NANs)
                    rateref,errorref,raterefbkg,\
                    errorrefbkg,ratecomp,errorcomp,ratecompbkg,errorcompbkg,\
                    timeref = arraysN
                    bsizeref = timeref[1]-timeref[0]
                                                    
                #Concatenate LCs from different observations
                for k4 in range(len(rateref)):
                    
                    windowcomb.append(windowref[k4])
                    reflccomb.append(rateref[k4])
                    errreflccomb.append(errorref[k4])
                    complccomb.append(ratecomp[k4])
                    errcomplccomb.append(errorcomp[k4])
                    reflcbkgcomb.append(raterefbkg[k4])
                    errreflcbkgcomb.append(errorrefbkg[k4])
                    complcbkgcomb.append(ratecompbkg[k4])
                    errcomplcbkgcomb.append(errorcompbkg[k4])
        
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
            timecombref = bsizeref*np.arange(0,len(reflccomb),1)
            reflccomb *= windowcomb
            complccomb *= windowcomb
            
            #Unfold spectrum using instrumental response      
            rmffile = "epn_" + str(ObsId) + "_" + str(visnum) + ".rmf"
            ancrfile = "epn_" + str(ObsId) + "_" + str(visnum) + ".arf"
            specfile = "epn_spec" + str(visnum) +\
                       "_grp_" + str(ObsId) + ".fits"
            infile = "covflux_" + str(ObsId) + ".dat"
            outfile = "covspec_" + str(ObsId) + ".pha"
            
            hdulist2 = fits.open(specfile)
            header2 = hdulist2[1].header
            backscal = header2['BACKSCAL']
            corrscal = header2['CORRSCAL']
            areascal = header2['AREASCAL']
            backfile = "NONE"
                                                                        
            #Plot LCs
            laben = str(float(energymin)) + "-" + str(float(energymax))
            labelsrccomp = "Comparison band LC: " + laben + " keV"    
            
            fvarcomp, dfvarcomp = Fracvar(complccomb,errcomplccomb)
            fvarref, dfvarref = Fracvar(reflccomb,errreflccomb)
            
            #Compare CPSDs
            if(comparecpsd=="yes"):
                
                #Stingray LCs and events
                countcomp = complccomb*bsizeref
                cerrorcomp = errcomplccomb*bsizeref
                countref = reflccomb*bsizeref
                cerrorref = errreflccomb*bsizeref
                countcompbkg = complcbkgcomb*bsizeref
                countrefbkg = raterefbkg*bsizeref
                LcComp = Lightcurve(timecomp,countcomp,error=cerrorcomp,\
                                    dt=bsizeref)
                LcRef = Lightcurve(timecomp,countref,error=cerrorref,\
                                   dt=bsizeref)
                evcomp = EventList.from_lc(LcComp)   
                evref = EventList.from_lc(LcRef)
                
                #CPSD (stingray)
                cspecfrq, crosspec, crosspecerr =\
                cross_spectrum_stingray(evref,evcomp,\
                Mseg,bsizeref,normpow,fminb,fmaxb)
                
                #CPSD
                fxy, cpxy, cpxyerr =\
                covariance_spectrum(complccomb,\
                                    errcomplccomb,reflccomb,\
                                    errreflccomb,\
                                    complcbkgcomb,reflcbkgcomb,\
                                    Mseg,bfactor,bsizeref,\
                                    statpow,fminb,fmaxb,windowref)
                
                plt.figure()
                plt.subplot(211)
                plt.errorbar(fxy,np.real(cpxy),\
                             yerr=abs(np.real(cpxyerr)),fmt='g.')
                plt.errorbar(cspecfrq,np.real(crosspec),\
                             yerr=abs(np.real(crosspecerr)),fmt='m.')
                plt.subplot(212)
                plt.errorbar(fxy,np.imag(cpxy),\
                             yerr=abs(np.imag(cpxyerr)),fmt='g.')
                plt.errorbar(cspecfrq,np.imag(crosspec),\
                             yerr=abs(np.imag(crosspecerr)),fmt='m.')
                plt.show()
            
            if(segmentlc=="yes"):
                
                complccomb = complccomb[timecombref>tmin]
                errcomplccomb = errcomplccomb[timecombref>tmin]
                reflccomb = reflccomb[timecombref>tmin]
                errreflccomb = errreflccomb[timecombref>tmin]
                complcbkgcomb = complcbkgcomb[timecombref>tmin]
                errcomplcbkgcomb = errcomplcbkgcomb[timecombref>tmin]
                reflcbkgcomb = reflcbkgcomb[timecombref>tmin]
                errreflcbkgcomb = errreflcbkgcomb[timecombref>tmin]
                windowcomb = windowcomb[timecombref>tmin]
                timecombref = timecombref[timecombref>tmin]
                
                complccomb = complccomb[timecombref<tmax]
                errcomplccomb = errcomplccomb[timecombref<tmax]
                reflccomb = reflccomb[timecombref<tmax]
                errreflccomb = errreflccomb[timecombref<tmax]
                complcbkgcomb = complcbkgcomb[timecombref<tmax]
                errcomplcbkgcomb = errcomplcbkgcomb[timecombref<tmax]
                reflcbkgcomb = reflcbkgcomb[timecombref<tmax]
                errreflcbkgcomb = errreflcbkgcomb[timecombref<tmax]
                windowcomb = windowcomb[timecombref<tmax]
                timecombref = timecombref[timecombref<tmax]

            if(plotlc=="yes"):
                                
                plt.plot(timecombref/ks,windowcomb,'b-')
                plt.errorbar(timecombref/ks,reflccomb,yerr=errreflccomb,\
                             fmt='k-')
                plt.errorbar(timecombref/ks,complccomb,yerr=errcomplccomb,\
                             label=labelsrccomp,fmt='r-')
                plt.tick_params(axis='both', which='major', labelsize=14)
                plt.legend(loc="best")
                plt.title("XMM-Newton (EPIC-PN) lightcurves")
                plt.ylabel("Count rate [s$^{-1}$]",fontsize=14)
                plt.xlabel("Time [ks]",fontsize=14)
                plt.show()
            
            if(plotpsd=="yes"):
                                
                retsim = "yes"
                logfreqpsd,logavgpypsd,logavgpyerrpsd,_,_,_ =\
                psdmod(timecombref,reflccomb,errreflccomb,\
                       complccomb,errcomplccomb,reflcbkgcomb,\
                       complcbkgcomb,Mseg,bfactor,bsizeref,\
                       statpow,windowcomb,removebt,retsim)
                
                # plt.figure()
                # plt.errorbar(logfreqpsd,logavgpypsd,yerr=logavgpyerrpsd,\
                #              fmt='b.')
                # plt.plot(fmodcpsd,pmodcpsd,'r-')
                # plt.errorbar(favgxpsd,avgpypsd,yerr=avgpypsderr,\
                #              fmt='g.')
                # plt.plot(fmodrpsd,pmodrpsd,'k-')
                # plt.xscale("log")
                # plt.yscale("log")
                # plt.show()
            
            if(iggaps=="yes"):
                
                complccombgap = complccomb[reflccomb>0]
                errcomplccombgap = errcomplccomb[reflccomb>0]
                errreflccombgap = errreflccomb[reflccomb>0]
                complcbkgcombgap = complcbkgcomb[reflccomb>0]
                errcomplcbkgcombgap = errcomplcbkgcomb[reflccomb>0]
                reflcbkgcombgap = reflcbkgcomb[reflccomb>0]
                errreflcbkgcombgap = errreflcbkgcomb[reflccomb>0]
                windowcombgap = windowcomb[reflccomb>0]
                timecombrefgap = timecombref[reflccomb>0]
                reflccombgap = reflccomb[reflccomb>0]
                                
                #Generate fake LC by sampling directly from original LC  
                #distribution
                Nsamples = len(complccomb)
                ctsthreshpoisson = 15
                
                timerefsim,timecompsim,reflccombsim,complccombsim,\
                errreflccompsim,errcomplccombsim = [[],[],[],[],[],[]]
                
                for jdsamp in range(Nsamples):
                                                            
                    #Reference band
                    randintref = np.random.randint(0,len(reflccombgap),1)[0]
                    ctsbinref = int(reflccombgap[randintref]*bsizeref)
                    
                    
                    if(ctsbinref<ctsthreshpoisson):
                    
                        mureflc = np.random.poisson(ctsbinref,1)[0]/bsizeref 
                        reflccombsim.append(mureflc)
                        timerefsim.append(timecombrefgap[randintref])
                        errreflccompsim.append(errreflccombgap[randintref])

                    if(ctsbinref>=ctsthreshpoisson):
                        
                        mureflc =\
                        np.random.normal(reflccombgap[randintref],\
                        errreflccombgap[randintref],1)[0]
                            
                        timerefsim.append(timecombrefgap[randintref])
                        reflccombsim.append(mureflc)
                        errreflccompsim.append(errreflccombgap[randintref])

                    #Comparison band
                    randintcomp = np.random.randint(0,len(complccombgap),1)[0]
                    ctsbincomp = int(complccombgap[randintcomp]*bsizeref)
                    if(ctsbincomp<ctsthreshpoisson):
                        
                        mucomplc =\
                        np.random.poisson(ctsbincomp,1)[0]/bsizeref
                        
                        timecompsim.append(timecombrefgap[randintcomp])
                        complccombsim.append(mucomplc)
                        errcomplccombsim.append(errcomplccombgap[randintcomp])

                    if(ctsbincomp>=ctsthreshpoisson):
                        
                        mucomplc =\
                        np.random.normal(complccombgap[randintcomp],\
                        errcomplccombgap[randintcomp],1)[0]
                        
                        timecompsim.append(timecombrefgap[randintcomp])
                        complccombsim.append(mucomplc)
                        errcomplccombsim.append(errcomplccombgap[randintcomp])
                
                timerefsim = np.array(timerefsim)
                timecompsim = np.array(timecompsim)
                reflccombsim = np.array(reflccombsim)
                errreflccompsim = np.array(errreflccompsim)
                complccombsim = np.array(complccombsim)
                errcomplccombsim = np.array(errcomplccombsim)                                                            
                resultks = stats.ks_2samp(reflccombsim,reflccombgap)
                                
                for jwinref in range(len(reflccombsim)):
                                                        
                    if(windowcomb[jwinref]==0):
                                                
                        reflccomb[jwinref] = reflccombsim[jwinref]
                        errreflccomb[jwinref] = errreflccompsim[jwinref]
                        complccomb[jwinref] = complccombsim[jwinref]
                        errcomplccomb[jwinref] = errcomplccombsim[jwinref]

                # plt.figure()
                # plt.errorbar(timecombref/ks,complccomb,\
                #              yerr=errcomplccomb,fmt='b.')
                # plt.plot(timecombref/ks,windowcomb,'k-')
                # plt.show()
                            
            if(genpsds=='yes'):
                
                retsim = "yes"
                fmodrpsd, pmodcomppsd, mulccomppsdW =\
                psdmod(timecombref,complccomb,errcomplccomb,\
                       reflccomb,errreflccomb,complcbkgcomb,\
                       reflcbkgcomb,Mseg,bfactor,bsizeref,\
                       statpow,windowcomb,removebt,retsim)
                                
                fmodrpsd, pmodrefpsd, mulcrefpsdW =\
                psdmod(timecombref,reflccomb,errreflccomb,\
                       complccomb,errcomplccomb,reflcbkgcomb,\
                       complcbkgcomb,Mseg,bfactor,bsizeref,\
                       statpow,windowcomb,removebt,retsim)
            
                # Sample from PSD with GPR to Generate fake LC
                if(len(fmodrpsd)>0):
                
                    telapsecomb = timecombref[-1]-timecombref[0]
                    freqgenmin = np.min(fmodrpsd)
                    freqgenmax = np.max(fmodrpsd)
                                                                
                    tgenref,rgenref,errgenref =\
                    genpsd(bsizeref,telapsecomb,freqgenmin,\
                           freqgenmax,pmodrefpsd,mulcrefpsdW)
                    
                    tgencomp,rgencomp,errgencomp =\
                    genpsd(bsizeref,telapsecomb,freqgenmin,freqgenmax,\
                           pmodcomppsd,mulccomppsdW)
                                    
                    rgenref = rgenref[0:-1]
                    errgenref = errgenref[0:-1]
                    rgencomp = rgencomp[0:-1]
                    errgencomp = errgencomp[0:-1]
                                                                                                                                
                    #Add a floor and subtract from mean rate
                    reflccombmin = abs(np.min(reflccomb))
                    complccombmin = abs(np.min(complccomb))
                    reflccomb -= reflccombmin
                    complccomb -= complccombmin
                    reflccombmean = np.mean(reflccomb)
                    complccombmean = np.mean(complccomb)
                    reflccomb -= reflccombmean
                    complccomb -= complccombmean
                    
                    for jwin in range(len(tgencomp)):
                                        
                        if(windowcomb[jwin]==0):
                            reflccomb[jwin] = rgenref[jwin]
                            errreflccomb[jwin] = errgenref[jwin]
                            complccomb[jwin] = rgencomp[jwin]
                            errcomplccomb[jwin] = errgencomp[jwin]
    
            # Compute covariance
            # Time domain
            intcovtd,intcoverrtd =\
            covariance_time_domain(complccomb,errcomplccomb,\
                                   reflccomb,errreflccomb,\
                                   Mseg)
            covtd.append(intcovtd)
            dcovtd.append(intcoverrtd)
                        
            print('Exposure: ',telapse/ks," ks")
            print('Fvar [reference-band]: ',fvarref," ± ",dfvarref)
            print('Rate [reference-band]',murateref," ± ",muerrorref)
            print('Fvar [comparison-band]: ',fvarcomp," ± ",dfvarcomp)
            print('Rate [comparison-band]',muratecomp," ± ",muerrorcomp)
            print("")
                        
            # Frequency domain                
            intcov,intcoverr =\
            covariance_spectrum(timecombref,complccomb,errcomplccomb,\
                                reflccomb,errreflccomb,complcbkgcomb,\
                                reflcbkgcomb,Mseg,bfactor,bsizeref,\
                                statpow,fminb[k3],fmaxb[k3],windowcomb,\
                                removebt)
            cov.append(intcov)
            dcov.append(intcoverr)

    cov = np.array(cov)
    dcov = np.array(dcov)
    energiesref = np.array(energiesref)
    denergiesref = np.array(denergiesref)
        
    if(plotcov=="yes"):
        
        plt.figure()
        plt.errorbar(energiesref,cov,yerr=dcov,fmt='b.',label="Method 1")
        plt.errorbar(energiesref,covtd,yerr=dcovtd,fmt='g.',label="Method 2")
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
    energiesref = np.array(energiesref)
    denergiesref = np.array(denergiesref)
        
    if(np.sum(cov)>0 and np.sum(dcov)>0):
                    
        fluxes = np.zeros(len(chans))
        dfluxes = np.zeros(len(chans))
        
        for j3 in range(len(cov)):
            for k3p in range(len(chans)):
                if(0.5*(EMIN[k3p]+EMIN[k3p])>=energiesref[j3]):
                    fluxes[chans[k3p]] = cov[j3]
                    dfluxes[chans[k3p]] = dcov[j3]
                    break
    
        Qcov = np.column_stack((chans,fluxes,dfluxes))
        np.savetxt(infilecov,Qcov,fmt='%i %s %s',delimiter='   ')
                                        
        comm_unfold = "ascii2pha infile=" + infilecov +\
        " outfile=" + outfilecov +\
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
                      " grouptype=optsnmin groupscale=" +\
                      str(groupscale) + " minchannel=-1 maxchannel=-1"
        os.system(comm_group)
    



    

        
    
    
    
                                
        
    
