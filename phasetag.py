import numpy as np
from astropy.io import fits
import os, glob
from stingray.pulse.pulsar import fold_events
from stingray.pulse.pulsar import htest
import matplotlib.pyplot as plt
import re
from kapteyn import kmpfit
from astropy.time import Time

day = 86400

def writepar(pfile,fout,freqs,fdots,reftime):
        
    with open(pfile) as fi:
        fwrite = open(fout,'w')
        for line in fi:
            line = line.split()
            
            if(line[0]=='F0'):
                line[1] = str(freqs)
            if(line[0]=='F1'):
                line[1] = str(fdots)
            if(line[0]=='PEPOCH'):
                line[1] = str(reftime)
            if(line[0]=='TZRMJD'):
                line[1] = str(reftime)

            line = "         ".join(line)
            fwrite.write(line + "\n")
        
        fwrite.close()

def readtoa(tfile,t0,f0,f1):
    
    toas,dtoas,phi,dphi,obsid = [[],[],[],[],[]]
    
    with open(tfile) as fi:
        for line in fi:
            line = line.split()
            if(len(line)>0 and line[0]=='photon_toa'):
                
                times = np.float64(line[2])
                dtimes = np.float64(line[3])
                freq = f0 + f1*(times-t0)*day
                dphis = (dtimes/1e6)*freq
                
                toas.append(np.float64(line[2]))
                dtoas.append(np.float64(line[3]))
                phi.append(np.float64(line[12]))
                dphi.append(dphis)
                obsid.append(line[14])
                
    return toas, dtoas, phi, dphi, obsid

def writetoa(timarr,dtimarr):
    
    string = []
    string.append("FORMAT 1")
    s0 = "HEN 0 "
    s1 = "@"
    for j in range(len(timarr)):
        comm = s0 + str(timarr[j])  + " " + str(dtimarr[j]) +\
               " " + s1
        string.append(comm)
    np.savetxt("par.tim",string,fmt='%s',delimiter='   ')

def loadtimfile(tfile):
    
    reftime,obsidref,instr = [[],[],[]]
        
    with open(tfile) as fi:
        for line in fi:
            line = line.split()
            
            if(len(line)>10):
                if(line[0]=='photon_toa'):
                    
                    
                    reftime.append(float(line[2]))
                    obsidref.append(line[16])
                    instr.append(line[-1])
    
    reftime = np.array(reftime)
    obsidref = np.array(obsidref)  
    instr = np.array(instr)
    
    return reftime,obsidref,instr


def loadparfile(pfile):
    
    t0 = 0
    F0 = 0
    F1 = 0
    F2 = 0
    TG1 = 0
    GLPH1 = 0
    GLF01 = 0
    GLF11 = 0
    
    with open(pfile) as fi:
        for line in fi:
            line = line.split()
            if(line[0]=='PEPOCH'):
                t0 = np.float64(line[1])
            if(line[0]=='F0'):
                F0 = np.float64(line[1])
            if(line[0]=='F1'):
                F1 = np.float64(line[1])
            if(line[0]=='F2'):
                F2 = np.float64(line[1])
            
            if(line[0]=='GLPH_1'):
                GLPH1 = np.float64(line[1])
            if(line[0]=='GLEP_1'):
                TG1 = np.float64(line[1])
            if(line[0]=='GLF0_1'):
                GLF01 = np.float64(line[1])
            if(line[0]=='GLF1_1'):
                GLF11 = np.float64(line[1])
    
    return t0,F0,F1,F2,TG1,GLPH1,GLF01,GLF11

def pulse_model(p, x):
    a, b, c = p
    y = c + a*np.sin(2.0*np.pi*x + b)
    return y

def pulse_model2(p, x):
    a, b, c, d, e = p
    y = c + a*np.sin(2.0*np.pi*x + b) + d*np.sin(4.0*np.pi*x + e)
    return y

def residuals(p, data):
    x, y, xerr, yerr = data
    a, b, c = p
    fprime = 2.0*np.pi*a*np.cos(2*np.pi*x + b)
    weights = yerr*yerr + (fprime*fprime)*xerr*xerr
    resid = (y - pulse_model(p,x))/np.sqrt(weights)
    return resid # Weighted residuals

def residuals2(p, data):
    x, y, xerr, yerr = data
    a, b, c, d, e = p
    fprime = 2.0*np.pi*a*np.cos(2.0*np.pi*x + b) +\
             4.0*np.pi*d*np.cos(4.0*np.pi*x + e)
    weights2 = yerr*yerr + (fprime*fprime)*xerr*xerr
    resid2 = (y - pulse_model2(p,x))/np.sqrt(weights2)
    return resid2 # Weighted residuals

def fit_mod(ph,pherr,rate,rateerr,redthresh):
    
    #Initialise fitting parameters
    ainit = 0.1
    binit = 0.2
    cinit = 1.0
    
    #Perform pulse profile fitting
    paramsinitial = [ainit, binit, cinit]
    fitobj = kmpfit.Fitter(residuals=residuals,data=(ph,rate,pherr,rateerr))   
    fitobj.fit(params0=paramsinitial)
    
    #Goodness of fit
    chi2min = fitobj.chi2_min
    dof = fitobj.dof   
    N = 1000
    
    redthresh = 1.5
    
    #First fit a single sinusoid to determine goodness-of-fit
    if(chi2min/dof<redthresh):
        
        #Obtain best-fit coefficients of pulse profile
        A = fitobj.params[0]
        B = fitobj.params[1]
        Berr = fitobj.xerror[1]
        C = fitobj.params[2]
        
        #Best-fit model to pulse profile
        xphase = np.linspace(np.min(ph),np.max(ph),N)
        yfunc = C + A*np.sin(2.0*np.pi*xphase + B)
            
        #Find peak of pulse profile in the range below numerically
        cutoffmin = 0.2
        cutoffmax = 1.2
        
        yf = yfunc[xphase<=cutoffmax]
        xp = xphase[xphase<=cutoffmax]
        yf = yf[xp>=cutoffmin]
        xp = xp[xp>=cutoffmin]
        
        #Pulse profile peak (numerical)
        phasemax = xp[np.argmax(yf)] 
        phasemin = xp[np.argmin(yf)]
        
        #Error in pulse profile peak
        phasemax_err = Berr/(2.0*np.pi)
        
    #Fit a double harmonic if previous goodness-of-fit is unacceptable
    if(chi2min/dof>=redthresh):

        ainit = 0.1
        binit = 0.2
        cinit = 1.0
        dinit = 0.1
        einit = 0.1

        paramsinitialnu = [ainit, binit, cinit, dinit, einit]
        fobj = kmpfit.Fitter(residuals=residuals2,data=(ph,rate,pherr,rateerr))
        fobj.fit(params0=paramsinitialnu)

        #Obtain best-fit coefficients of pulse profile
        A = fobj.params[0]
        B = fobj.params[1]
        Berr = fobj.xerror[1]
        C = fobj.params[2]
        D = fobj.params[3]
        E = fobj.params[4]
        
        #Best-fit model
        xphase = np.linspace(np.min(ph),np.max(ph),N)
        yfunc = C + A*np.sin(2.0*np.pi*xphase + B) +\
                D*np.sin(4.0*np.pi*xphase + E)
                
        #Find peak of pulse profile in the range below numerically
        cutoffmin = 0.2
        cutoffmax = 1.2
        yf = yfunc[xphase<=cutoffmax]
        xp = xphase[xphase<=cutoffmax]
        yf = yf[xp>=cutoffmin]
        xp = xp[xp>=cutoffmin]
        
        phasemax = xp[np.argmax(yf)]
        phasemin = xp[np.argmin(yf)]
        
        phasemax_err = Berr/(2.0*np.pi)
        
        chi2min = fobj.chi2_min
        dof = fobj.dof    
        
        # if(chi2min/dof>redthresh):
                        
        #     print("Fitting procedure not successful with 2 harmonics")
        #     phasemax = -99
        #     phasemin = -99
        #     phasemax_err = -99
    
    return phasemin, phasemax, phasemax_err, xphase, yfunc

def arr_time_corr(evnts,t0,f0,fdot0,fddot0,\
                  ifgltch,tg1,glph1,glf01):
                  
    evnts = np.array(evnts)
    modsec = 0.5*(fdot0/f0)*((evnts-t0)**2) +\
            (1./6.)*(fddot0/f0)*((evnts-t0)**3)
    
    modglitch = np.zeros(len(evnts))
    if(ifgltch=="yes" ):
        for revn in range(len(evnts)):
            
            if(evnts[revn]>tg1):
                modglitch[revn] = glph1 + (glf01/f0)*(evnts[revn]-tg1) 
                        
    evnts = evnts + modsec + modglitch

    return evnts

def phaseintervals(ev,fsol,pep,nbins,pwid,plot):
        
    ev = np.array(ev)
    phase,rate,rerr = fold_events(ev,fsol,nbin=nbins,ref_time=pep)
    
    dphase = phase[1]-phase[0]
    phasecopy = np.arange(phase[-1]+dphase,2.0*phase[-1]+dphase,dphase)
    pnew = np.append(phase,phasecopy)
    xerrnew = np.zeros(len(pnew)) + dphase
    ratenew = np.append(rate,rate)
    rerrnew = np.append(rerr,rerr)
    xerrnew = 0.5*xerrnew
    rerrnew = 0.5*rerrnew
            
    pmin = 0
    pmax = 0
    pmaxerr = 0
    m,htestscore = htest(ratenew,nmax=2)
    hthresh = 1.0
    if(htestscore>hthresh):
        chithresh = 2.0
        pmin,pmax,pmaxerr,xmod,ymod =\
        fit_mod(pnew,xerrnew,ratenew,rerrnew,chithresh)
        
        if(pmin>1.0):
            pmin-=1.0
        if(pmax>1.0):
            pmax -= 1.0
    
    phaseintmin = pmin
    phaseintmax = pmax
    
    if(plot=='yes'):
        
        plt.figure()
        plt.errorbar(pnew,ratenew,xerr=xerrnew,yerr=rerrnew,fmt='k.')
        plt.plot(xmod,ymod,'r-')
        plt.ylim(np.min(ratenew)-30,np.max(ratenew)+30)
        # plt.legend(loc="best")
        plt.show()
    
    return phaseintmin,phaseintmax

def phasetag(evfileptag,outparfile,outparfile2,instr):
    
    x = re.findall(r'\d+',evfileptag)[0]
    if(np.int64(x)==0):
        x = x + str(evfileptag[3])

    string0 = "photonphase "
    string1 = " --outfile "
    string2 = " --ephem DE430"

    if(instr=="NICER"):
        fout1 = "ni" + str(x) + "a_phase.fits"
        fout2 = "ni" + str(x) + "b_phase.fits"
    if(instr=="NuSTAR"):
        fout1 = "nu" + str(x) + "a_phase.fits"
        fout2 = "nu" + str(x) + "b_phase.fits"
    if(instr=="XMM"):
        fout1 = "xmm" + str(x) + "a_phase.fits"
        fout2 = "xmm" + str(x) + "b_phase.fits"

    commsA = string0 +  evfileptag + " " + outparfile + string1 +\
              fout1 + string2 
        
    commsB = string0 +  evfileptag + " " + outparfile2 + string1 +\
              fout2 + string2
            
    return commsA, commsB

def addphasecol(origfile,instr):
    
    x = re.findall(r'\d+',origfile)[0]
    
    string0 = "ftpaste " + "'" + origfile + "[EVENTS]" + "'" 
    
    if(instr=='NICER'):
        
        string1 = "'" + 'ni' + x + 'a_phase.fits' + \
                  "[EVENTS][col Phase==PULSE_PHASE]" + "'"
        string2 = "origev" + x + "A.evt"
        
        string3 = "ftpaste " + "'" + origfile + "[EVENTS]" + "'" 
        string4 = "'" + 'ni' + x + 'b_phase.fits' + \
                  "[EVENTS][col Phase==PULSE_PHASE]" + "'"
        string5 = "origev" + x + "B.evt"
    
    if(instr=='NuSTAR'):
        
        string1 = "'" + 'nu' + x + 'a_phase.fits' + \
                  "[EVENTS][col Phase==PULSE_PHASE]" + "'"
        string2 = "origev" + x + "A.evt"
        
        string3 = "ftpaste " + "'" + origfile + "[EVENTS]" + "'" 
        string4 = "'" + 'nu' + x + 'b_phase.fits' + \
                  "[EVENTS][col Phase==PULSE_PHASE]" + "'"
        string5 = "origev" + x + "B.evt"

    comm1 = string0 + " " + string1 + " " + string2 + " clobber=yes"
    comm2 = string3 + " " + string4 + " " + string5 + " clobber=yes"
                
    return comm1,comm2
    
def phasefilter(evfile,phasemin,phasemax,instr):
    
    if(instr=='NICER'):
    
        string3 = "niextract-events "
        string4 = "Phase="
        
        x = re.findall(r'\d+',evfile)[0]
        file1 = 'origev' + x + 'A.evt'
        file2 = 'origev' + x + 'B.evt'
        
        foutfilt1 = "ni" + str(x) + "_filtered_high.evt"
        foutfilt2 = "ni" + str(x) + "_filtered_low.evt"
        
        commsA = (string3 + "'" + file1 + "[" + string4 +\
                  str(phasemin) + ":" +\
                  str(phasemax) + "]' " + foutfilt1)
        
        commsB = (string3 + "'" + file2 + "[" + string4 +\
                  str(phasemin) + ":" +\
                  str(phasemax) + "]' " + foutfilt2)
    
    if(instr=='NuSTAR'):
                
        x = re.findall(r'\d+',evfile)[0]
        file1 = 'origev' + x + 'A.evt'
        file2 = 'origev' + x + 'B.evt'
        gtifile1 = 'gti_' + x + '_high.fits'
        gtifile2 = 'gti_' + x + '_low.fits'
                
        expr = '"(Phase > ' + str(phasemin) + ').and.(Phase <= ' +\
               str(phasemax) + ')"' + ' anything anything TIME NO'
                       
        commsA = 'maketime ' + file1 + " " + gtifile1 + " " + expr
        commsB = 'maketime ' + file2 + " " + gtifile2 + " " + expr
                
    return commsA, commsB

def phaext(inevfile,outphafile):
    
    string1 = "niextspect infile="
    string2 = "outfile="
    string3 = "clobber=YES"
    
    comms = string1 + inevfile + " " + string2 + outphafile + " " +\
            string3

    return comms

def arfrmfgen(phafile):
    
    hdulist = fits.open(phafile)
    hdr = hdulist[1].header
    obsid = hdr['OBS_ID']
    y = re.findall(r'\d+',phafile)[0]
    
    if(np.int64(y)==0):
        y = y + str(phafile[3])
    
    arffile = "ni" + y + ".arf"
    rmffile = "ni" + y + ".rmf"
    wtfile = "ni" + y + "_wt.lis"
    mkfile = "ni" + str(obsid) + ".mkf"
    
    s0 = "nicerarf infile="
    s1 = " ra=281.6039167 dec=-2.9750278 "
    s2 = "attfile="
    s3 = "selfile="
    s4 = "outfile="
    s5 = "outwtfile="
    s0b = "nicerrmf infile="
    s6 = " mkfile="
    s7 = "detlist=@"
    
    commA = s0 + phafile + s1 + s2 + mkfile + " " + s3 + mkfile + " " + s4 +\
            arffile + " " + s5 + wtfile + " clobber=yes"
    commB = s0b + phafile + s6 + mkfile + " " + s4 + rmffile + " " + s7 +\
            wtfile + " clobber=yes"
    
    return commA, commB

def groupspec(inphafile):

    s0 = "ftgrouppha infile="
    s1 = "backfile="
    s2 = "respfile="
    s3 = "outfile="
    s4 = "grouptype=optmin"
    s5 = "groupscale=25 minchannel=20 maxchannel=1000"
    
    z = re.findall(r'\d+',inphafile)[0]
    if(np.int64(z)==0):
        z = z + str(inphafile[3])
    backfile = "ni" + z + "_low.pha"
    rmffile = "ni" + z + ".rmf"
    outfile = "ni" + z + "_grouped.pha"
    
    comm = s0 + inphafile + " " + s1 + backfile + " " + s2 + rmffile + " " +\
            s3 + outfile + " " + s4 + " " + s5
    
    return comm

def xspeccomms(phfile):
    
    w = re.findall(r'\d+',phfile)[0]
    
    if(np.int64(w)==0):
        w = w + str(phfile[3])
    
    comms = []
    comms.append("data " + 'ni' + str(w) + '_grouped.pha')
    comms.append("notice 0.3-10.0")
    comms.append("ig **-0.3 10.0-**")
    comms.append("ig bad")
    comms.append("query yes")
    comms.append("setplot energy")
    comms.append("abund wilm")   
    
    # comms.append("mo tbabs*(bbodyrad+powerlaw)")
    # comms.append("5.5 -1")
    # comms.append("1.0 0.1")
    # comms.append("1.0 0.1")
    # comms.append("1.35 -1")
    # comms.append("8e-4 -1")
    # comms.append("fit")
    
    comms.append("mo tbabs*powerlaw")
    comms.append("5.5 -1")
    comms.append("1.35 -1")
    comms.append("8e-4")
    comms.append("fit")
    comms.append("addc 2 cflux")
    comms.append("2")
    comms.append("10.0")
    comms.append("-12.0")
    comms.append("freeze 6")
    comms.append("fit")

    # comms.append("addc 1 cflux")
    # comms.append("1")
    # comms.append("10.0")
    # comms.append("-12.0")
    # comms.append("delc 2")
    # comms.append("addc 1 tbabs")
    # comms.append("5.5 -1")
    # comms.append("freeze 5 6")
    # comms.append("fit")
    
    comms.append("log >ni" + str(w) + "flux.dat") 
    comms.append("show free")
    comms.append("error 4")
    comms.append("log none")    
    comms.append("")
    
    return comms

##############################################################################
##############################################################################

presidfile = "fdot_evol.dat"
tpost,fpost,dfpost,fdotpost,dfdotpost,instpost,obsidpost =\
[[],[],[],[],[],[],[]]
with open(presidfile) as fi:
    for line in fi:
        line = line.split()
        
        tpost.append(float(line[0]))
        fpost.append(float(line[1]))
        dfpost.append(float(line[2]))
        fdotpost.append(float(line[3]))
        dfdotpost.append(float(line[4]))
        instpost.append(line[5])
        obsidpost.append(line[6])

tpost = np.array(tpost)
fpost = np.array(fpost)
dfpost = np.array(dfpost)
fdotpost = np.array(fdotpost)
dfdotpost = np.array(dfdotpost)
instpost = np.array(instpost)
obsidpost = np.array(obsidpost)
                                                                              
parfile = 'post_outburst.par'
timfile = 'post_outburst.tim'
ifglitch = "no"
plts = "no"
t0,f0,fdot0,fddot0,Tg1,gLph1,gLf01,gLf11 = loadparfile(parfile)

stringphasegen,stringphagen,stringgroup = [[],[],[]]
srcra = 281.6039171
srcdec = -2.9750278
bkgra = 281.4861058
bkgdec = -2.9766642
nbins = 30
pwidth = 1.0
instoffset = 0.0
mincts = 50
inst = "NICER"

if(inst=="NuSTAR"):
    
    nufiltkey = "nu*80602315006*fpma_filt*bary*.evt"
    
    for ctsobs in range(len(tpost)):
        
        for evfile in sorted(glob.glob(nufiltkey)):
            
            hdulist = fits.open(evfile)
            events = hdulist[1].data['TIME']
            hdr = hdulist[1].header
            tstart = hdr['TSTART']
            tstop = hdr['TSTOP'] 
            mjdrefi = hdr['MJDREFI']
            mjdref = hdr['MJDREFI'] + hdr['MJDREFF']
            obsidnu = hdr['OBS_ID']
            
            mjdstart = mjdref + tstart/day
            mjdstop = mjdref + tstop/day    
            tobsnu = 0.5*(mjdstart + mjdstop)
            
            if(instpost[ctsobs]=='NuSTAR' and\
               obsidnu==obsidpost[ctsobs]):
                                
                pepreflow = (tpost[ctsobs]-mjdrefi)*day
                peprefhigh = pepreflow + (0.5)*(fpost[ctsobs]**-1)
                
                evcorrlow = arr_time_corr(events,pepreflow,\
                            fpost[ctsobs],fdotpost[ctsobs],0,\
                            ifglitch,Tg1,gLph1,gLf01)
                evcorrhigh = arr_time_corr(events,peprefhigh,\
                             fpost[ctsobs],fdotpost[ctsobs],0,\
                             ifglitch,Tg1,gLph1,gLf01)
            
                pmilow,pmxlow =\
                phaseintervals(evcorrlow,fpost[ctsobs],pepreflow,nbins,\
                               pwidth,plot=plts)
                pmihigh,pmxhigh =\
                phaseintervals(evcorrhigh,fpost[ctsobs],peprefhigh,nbins,\
                               pwidth,plot=plts)
                
                pmilow -= instoffset
                pmxlow -= instoffset
                pmihigh -= instoffset
                pmxhigh -= instoffset
                                                                        
                evfilefiltlow = evfile.split("_bary.evt")[0] +\
                                "_low_filtered.fits"
                evfilefilthigh = evfile.split("_bary.evt")[0] +\
                                "_high_filtered.fits"
                            
                tpostphhigh = tpost[ctsobs] 
                tpostphlow = tpostphhigh + 0.5*(fpost[ctsobs]**-1)/day
                
                pfilelow = "phase" + str(obsidnu) + "_high.par" 
                writepar("phase.par",pfilelow,fpost[ctsobs],\
                         fdotpost[ctsobs],tpostphlow)
        
                pfilehigh = "phase" + str(obsidnu) + "_low.par" 
                writepar("phase.par",pfilehigh,fpost[ctsobs],\
                         fdotpost[ctsobs],tpostphhigh)
                                    
                c1,c2 = phasetag(evfile,pfilelow,pfilehigh,inst)
                            
                pmilowmin = pmilow - 0.5*pwidth
                pmilowmax = pmilow + 0.5*pwidth
                
                pmxhighmin = pmxhigh - 0.5*pwidth
                pmxhighmax = pmxhigh + 0.5*pwidth
                                                                
                nuorigfile = "nu" + obsidnu + "_fpma_filt.evt"                
                c3,c4 = addphasecol(nuorigfile,inst)
                c5,c6 = phasefilter(evfile,pmilowmin,pmilowmax,inst)
                
                stringphasegen.append(c1)
                stringphasegen.append(c2)
                stringphasegen.append("")
                stringphasegen.append(c3)
                stringphasegen.append(c4)
                stringphasegen.append("")
                stringphasegen.append(c5)
                stringphasegen.append(c6)
                stringphasegen.append("")
                
                gtihigh = "gti_" + str(obsidnu) + "_high.fits"
                gtilow = "gti_" + str(obsidnu) + "_low.fits"
                
                c7 = "cp " + gtihigh + " ../"
                c8 = "cp " + gtilow + " ../"
                
                c8a = "cp ../" + obsidnu + "/auxil/* " +\
                      "../" + obsidnu + "/"
                c8b = "cp ../" + obsidnu + "/event_cl/* " +\
                      "../" + obsidnu + "/"
                c8c = "cp ../" + obsidnu + "/event_uf/* " +\
                      "../" + obsidnu + "/"
                c8d = "cp ../" + obsidnu + "/hk/* " +\
                      "../" + obsidnu + "/"
                
                c8e = "gunzip -f ../" + obsidnu + "/*.gz"

                c9 = "cd ../"
                
                stringphasegen.append(c7)
                stringphasegen.append(c8)
                stringphasegen.append(c8a)
                stringphasegen.append(c8b)
                stringphasegen.append(c8c)
                stringphasegen.append(c8d)
                stringphasegen.append(c8e)
                stringphasegen.append(c9)
                stringphasegen.append("")
                
                c10 = "nuproducts indir=" + str(obsidnu) +\
                " instrument=FPMA steminputs=nu" + str(obsidnu) +\
                " outdir=reduced/ stemout=nu" + str(obsidnu) + "_high" +\
                " srcra=" + str(srcra) + " srcdec=" + str(srcdec) +\
                " srcradius=15 " +\
                " bkgra=" + str(bkgra) + " bkgdec=" + str(bkgdec) +\
                " bkgradius1=5.0 bkgradius2=20.0 bkgextract=yes" +\
                " rungrppha=no grpmincounts=15" +\
                " grppibadlow=35 grppibadhigh=1935 barycorr=no" +\
                " usrgtifile=" + gtihigh + " clobber=yes"

                c11 = "nuproducts indir=" + str(obsidnu) +\
                " instrument=FPMA steminputs=nu" + str(obsidnu) +\
                " outdir=reduced/ stemout=nu" + str(obsidnu) + "_low" +\
                " srcra=" + str(srcra) + " srcdec=" + str(srcdec) +\
                " srcradius=15.0" +\
                " bkgra=" + str(bkgra) + " bkgdec=" + str(bkgdec) + " " +\
                "bkgradius1=5.0 bkgradius2=20.0 bkgextract=yes" +\
                " rungrppha=no grpmincounts=15" +\
                " grppibadlow=35 grppibadhigh=1935 barycorr=no" +\
                " usrgtifile=" + gtilow + " clobber=yes"
                
                nuphahigh = "nu" + str(obsidnu) + "_high_sr.pha"
                nuphalow = "nu" + str(obsidnu) + "_low_sr.pha"
                
                c12 = "cd reduced/"
                c13 = "ftgrouppha infile=nu" + str(obsidnu) +\
                "_high_sr.pha" + " backfile=nu" + str(obsidnu) +\
                "_low_sr.pha respfile=nu" + str(obsidnu) +\
                "_high_sr.rmf" + " outfile=nu" + str(obsidnu) +\
                "_grouped.pha grouptype=optsnmin groupscale=1 " +\
                "minchannel=35 maxchannel=1935"                
                
                stringphagen.append(c10)
                stringphagen.append(c11)
                stringphagen.append("")
                stringphagen.append(c12)
                stringphagen.append(c13)

    np.savetxt("phasegen.sh",stringphasegen,fmt='%s')
    os.system("chmod u+x phasegen.sh")
    np.savetxt("specgen.sh",stringphagen,fmt='%s')
    os.system("chmod u+x specgen.sh")

if(inst=="NICER"):
    for ctsobs in range(len(tpost)):
                
        for evfile in sorted(glob.glob("ni*bary.evt")):
                        
            hdulist = fits.open(evfile)
            events = hdulist[1].data['TIME']
            hdr = hdulist[1].header
            tstart = hdr['TSTART']
            tstop = hdr['TSTOP'] 
            mjdrefi = hdr['MJDREFI']
            mjdref = hdr['MJDREFI'] + hdr['MJDREFF']
            obsidnicer = hdr['OBS_ID']
            
            mjdstart = mjdref + tstart/day
            mjdstop = mjdref + tstop/day    
            tobsnicer = 0.5*(mjdstart + mjdstop)
                                                
            if(instpost[ctsobs]=='NICER' and\
               obsidnicer==obsidpost[ctsobs]):
                                            
                pepreflow = (tpost[ctsobs]-mjdrefi)*day
                peprefhigh = pepreflow + (0.5)*(fpost[ctsobs]**-1)
                
                evcorrlow = arr_time_corr(events,pepreflow,\
                            fpost[ctsobs],fdotpost[ctsobs],0,\
                            ifglitch,Tg1,gLph1,gLf01)
                evcorrhigh = arr_time_corr(events,peprefhigh,\
                             fpost[ctsobs],fdotpost[ctsobs],0,\
                             ifglitch,Tg1,gLph1,gLf01)
    
                pmilow,pmxlow =\
                phaseintervals(evcorrlow,fpost[ctsobs],pepreflow,nbins,\
                               pwidth,plot=plts)
                pmihigh,pmxhigh =\
                phaseintervals(evcorrhigh,fpost[ctsobs],peprefhigh,nbins,\
                               pwidth,plot=plts)
                                                        
                evfilefiltlow = evfile.split("_bary.evt")[0] +\
                                "_low_filtered.fits"
                evfilefilthigh = evfile.split("_bary.evt")[0] +\
                                "_high_filtered.fits"
                            
                tpostphhigh = tpost[ctsobs] 
                tpostphlow = tpostphhigh + 0.5*(fpost[ctsobs]**-1)/day
                
                pfilelow = "phase" + str(obsidnicer) + "_high.par" 
                writepar("phase.par",pfilelow,fpost[ctsobs],\
                         fdotpost[ctsobs],tpostphlow)
        
                pfilehigh = "phase" + str(obsidnicer) + "_low.par" 
                writepar("phase.par",pfilehigh,fpost[ctsobs],\
                         fdotpost[ctsobs],tpostphhigh)
                                    
                c2,c3 = phasetag(evfile,pfilelow,pfilehigh,inst)
                
                pmilow += instoffset
                pmxlow += instoffset
                pmihigh += instoffset
                pmxhigh += instoffset
                
                pmilowmin = pmilow - 0.5*pwidth
                pmilowmax = pmilow + 0.5*pwidth
                
                pmxhighmin = pmxhigh - 0.5*pwidth
                pmxhighmax = pmxhigh + 0.5*pwidth
                
                phmean = 0.5*(pmilowmin + pmilowmax)
                
                #For this case
                pmilowmin = 0.31
                pmilowmax = 0.51
                                
                commcporigev = "cp ../" + obsidnicer +\
                "/xti/event_cl/*mpu7*cl.evt ."
                # os.system(commcporigev)
                
                niorigfile = "ni" + obsidnicer + "_0mpu7_cl.evt"
                c4,c5 = addphasecol(niorigfile,instpost[ctsobs])
                c6,c7 = phasefilter(evfile,pmilowmin,pmilowmax,\
                                    instpost[ctsobs])
                
                stringphasegen.append(c2)
                stringphasegen.append(c3)
                stringphasegen.append("")
                stringphasegen.append(c4)
                stringphasegen.append(c5)
                stringphasegen.append("")
                stringphasegen.append(c6)
                stringphasegen.append(c7)
                stringphasegen.append("")
                
                infiltfile = 'ni' + str(obsidnicer) + '_filtered_low.evt'
                outphafile = 'ni' + str(obsidnicer) + '_low.pha'
                stringphasegen.append(phaext(infiltfile,outphafile))
                infiltfile = 'ni' + str(obsidnicer) + '_filtered_high.evt'
                outphafile = 'ni' + str(obsidnicer) + '_high.pha'
                stringphasegen.append(phaext(infiltfile,outphafile))
                stringphasegen.append("")
                stringgroup.append(groupspec(outphafile))
                stringphasegen.append("")
                    
    #Update keyword
    stringresp = []
    for niphafilehigh in sorted(glob.glob("ni*high.pha")):
                                    
        hdupha = fits.open(niphafilehigh,mode='update')
        hdrpha = hdupha[1].header
        obsid = hdrpha['OBS_ID']
        telapse = hdrpha['ONTIME']
        
        niphafilelow = 'ni' + str(obsid) + "_low.pha"
        
        hdr['BACKFILE'] = 'ni' + obsid + '_low.pha'
        hdr['ANCRFILE'] = 'ni' + obsid + '.arf'
        hdr['RESPFILE'] = 'ni' + obsid + '.rmf'
        
        hdupha.flush()
        hdupha.close()
                
        commcpauxil = "cp ../" + obsid + "/auxil/* ."  
        # os.system(commcpauxil)
        
        effexposure = telapse*pwidth
        
        commpar1 = "fparkey " + str(effexposure) + ' "' +\
        niphafilehigh + '[0]" EXPOSURE'
        commpar2 = "fparkey " + str(effexposure) + ' "' +\
        niphafilelow + '[0]" EXPOSURE'
        commpar3 = "fparkey " + str(effexposure) + ' "' +\
        niphafilehigh + '[1]" EXPOSURE'
        commpar4 = "fparkey " + str(effexposure) + ' "' +\
        niphafilelow + '[1]" EXPOSURE'
        commpar5 = "fparkey " + str(effexposure) + ' "' +\
        niphafilehigh + '[2]" EXPOSURE'
        commpar6 = "fparkey " + str(effexposure) + ' "' +\
        niphafilelow + '[2]" EXPOSURE'

        commpar1b = "fparkey " + str(effexposure) + ' "' +\
        niphafilehigh + '[0]" ONTIME'
        commpar2b = "fparkey " + str(effexposure) + ' "' +\
        niphafilelow + '[0]" ONTIME'
        commpar3b = "fparkey " + str(effexposure) + ' "' +\
        niphafilehigh + '[1]" ONTIME'
        commpar4b = "fparkey " + str(effexposure) + ' "' +\
        niphafilelow + '[1]" ONTIME'
        commpar5b = "fparkey " + str(effexposure) + ' "' +\
        niphafilehigh + '[2]" ONTIME'
        commpar6b = "fparkey " + str(effexposure) + ' "' +\
        niphafilelow + '[2]" ONTIME'
                
        stringphagen.append(commpar1)
        stringphagen.append(commpar2)
        stringphagen.append(commpar3)
        stringphagen.append(commpar4)
        stringphagen.append(commpar5)
        stringphagen.append(commpar6)

        stringphagen.append(commpar1b)
        stringphagen.append(commpar2b)
        stringphagen.append(commpar3b)
        stringphagen.append(commpar4b)
        stringphagen.append(commpar5b)
        stringphagen.append(commpar6b)
                        
        c8,c9 = arfrmfgen(niphafilehigh)
        stringphagen.append(c8)
        stringphagen.append(c9)
        stringphagen.append("")
                
        stringxspec = xspeccomms(niphafilehigh)
        with open("specread.sh", "ab") as f:
            np.savetxt(f, stringxspec, fmt='%s')
        f.close()
    
    np.savetxt("phasegen.sh",stringphasegen,fmt='%s')
    os.system("chmod u+x phasegen.sh")
    np.savetxt("specgen.sh",stringphagen,fmt='%s')
    os.system("chmod u+x specgen.sh")

    np.savetxt("group.sh",stringgroup,fmt='%s')
    os.system("chmod u+x group.sh")
    os.system("chmod u+x specread.sh")

if(inst=="XMM"):
    
    for ctsobs in range(len(tpost)):

        for xmmevfile in sorted(glob.glob("epn*net*ref*.fits")):
                    
            hdulist = fits.open(xmmevfile)
            events = hdulist[1].data['TIME']
            hdr = hdulist[1].header
            tstart = hdr['TSTART']
            tstop = hdr['TSTOP']
            obsidxmm = hdr['OBS_ID']
            dateobs = hdr['DATE-OBS']
            mjdrefi = hdr['MJDREF']
            tj = Time(dateobs,format='isot',scale='utc')
            mjdref = tj.mjd
                        
            if(instpost[ctsobs]=='XMM' and obsidxmm==obsidpost[ctsobs]):
            
                pepreflow = (tpost[ctsobs]-mjdrefi)*day
                peprefhigh = pepreflow + (0.5)*(fpost[ctsobs]**-1)
            
                evcorrlow = arr_time_corr(events,pepreflow,\
                            fpost[ctsobs],fdotpost[ctsobs],0,\
                            ifglitch,Tg1,gLph1,gLf01)
                evcorrhigh = arr_time_corr(events,peprefhigh,\
                             fpost[ctsobs],fdotpost[ctsobs],0,\
                             ifglitch,Tg1,gLph1,gLf01)
    
                pmilow,pmxlow =\
                phaseintervals(evcorrlow,fpost[ctsobs],pepreflow,nbins,\
                               pwidth,plot=plts)
                pmihigh,pmxhigh =\
                phaseintervals(evcorrhigh,fpost[ctsobs],peprefhigh,nbins,\
                               pwidth,plot=plts)
                                                                    
                evfilefiltlow = xmmevfile.split("_ref.fits")[0] +\
                                "_low_filtered.fits"
                evfilefilthigh = xmmevfile.split("_ref.fits")[0] +\
                                "_high_filtered.fits"
                                                        
                tpostphhigh = tpost[ctsobs] 
                tpostphlow = tpostphhigh + 0.5*(fpost[ctsobs]**-1)/day
            
                pfilelow = "phase" + str(obsidxmm) + "_high.par" 
                writepar("phase.par",pfilelow,fpost[ctsobs],\
                         fdotpost[ctsobs],tpostphlow)
    
                pfilehigh = "phase" + str(obsidxmm) + "_low.par" 
                writepar("phase.par",pfilehigh,fpost[ctsobs],\
                         fdotpost[ctsobs],tpostphhigh)
                                
                c2,c3 = phasetag(xmmevfile,pfilelow,pfilehigh,inst)
            
                pmilow += instoffset
                pmxlow += instoffset
                pmihigh += instoffset
                pmxhigh += instoffset
            
                pmilowmin = pmilow - 0.5*pwidth
                pmilowmax = pmilow + 0.5*pwidth
                pmxhighmin = pmxhigh - 0.5*pwidth
                pmxhighmax = pmxhigh + 0.5*pwidth
                phmean = 0.5*(pmilowmin + pmilowmax)
                
                phasefileA = "xmm1a_phase.fits"
                phasefileB = "xmm1b_phase.fits"
            
                outcpfileA = "origev" + str(obsidxmm) + "A.fits"
                outcpfileB = "origev" + str(obsidxmm) + "B.fits"
                
                outphasefileA = "xmm1a_phase" + str(obsidxmm) +\
                                "_filtered.fits"
                outphasefileB = "xmm1b_phase" + str(obsidxmm) +\
                                "_filtered.fits"
       
                c3a = "ftpaste '" + xmmevfile + "' '" +\
                phasefileA + "[EVENTS][col Phase==PULSE_PHASE]'" +\
                " " + outcpfileA + " clobber=yes"
                
                c3b = "ftpaste '" + xmmevfile + "' '" +\
                phasefileB + "[EVENTS][col Phase==PULSE_PHASE]'" +\
                " " + outcpfileB + " clobber=yes"

                c4 = "evselect table=" + outcpfileA +\
                ' energycolumn=PI expression="#XMMEA_EP && ' +\
                '(PATTERN==0) && (PI>=300 && PI<=12000) && ' +\
                '(Phase in [' + str(pmilowmin) + ":" +\
                str(pmilowmax) + '])" filteredset=' + outphasefileA 
                
                c5 = "evselect table=" + outcpfileB +\
                ' energycolumn=PI expression="#XMMEA_EP && ' +\
                '(PATTERN==0) && (PI>=300 && PI<=12000) && ' +\
                '(Phase in [' + str(pmilowmin) + ":" +\
                str(pmilowmax) + '])" filteredset=' + outphasefileB 
                
                specsrcA = "xmm_phase_specA.fits"            
                specrmfA = "epn_" + str(obsidxmm) + "_" + "A.rmf"
                specarfA = "epn_" + str(obsidxmm) + "_" + "A.arf"
                badpixfile = "epn" + obsidxmm + ".fits"

                specsrcB = "xmm_phase_specB.fits"
                specrmfB = "epn_" + str(obsidxmm) + "_" + "B.rmf"
                specarfB = "epn_" + str(obsidxmm) + "_" + "B.arf"

                c6 =  "evselect table=" + outphasefileA +\
                " withspectrumset=yes " + "spectrumset=" +\
                specsrcA + " energycolumn=PI" +\
                " spectralbinsize=5" +\
                " withspecranges=yes specchannelmin=0 " +\
                "specchannelmax=20479"
                c7 =  "evselect table=" + outphasefileB +\
                " withspectrumset=yes " + "spectrumset=" +\
                specsrcB + " energycolumn=PI" +\
                " spectralbinsize=5" +\
                " withspecranges=yes specchannelmin=0 " +\
                "specchannelmax=20479"

                c8 = "cp ../" + obsidxmm + "/proc/epn_obs1.fits " +\
                badpixfile
                
                c9 = "backscale spectrumset=" + specsrcA + " " +\
                "badpixlocation=" + badpixfile
                c10 = "rmfgen spectrumset=" + specsrcA + " rmfset=" +\
                specrmfA + " extendedsource=no"
                c11 = "arfgen arfset=" + specarfA + " spectrumset=" +\
                specsrcA + " withrmfset=yes rmfset=" + specrmfA +\
                " withbadpixcorr=yes badpixlocation=" +\
                badpixfile + " detmaptype=psf"
                
                
                c12 = "backscale spectrumset=" + specsrcB + " " +\
                "badpixlocation=" + badpixfile
                c13 = "rmfgen spectrumset=" + specsrcB + " rmfset=" +\
                specrmfB + " extendedsource=no"
                c14 = "arfgen arfset=" + specarfB + " spectrumset=" +\
                specsrcB + " withrmfset=yes rmfset=" + specrmfB +\
                " withbadpixcorr=yes badpixlocation=" +\
                badpixfile + " detmaptype=psf"
                
                groupspec = "epn_spec_grp_" + str(obsidxmm) + ".fits"
                c15 = "specgroup spectrumset=" + str(specsrcA) +\
                      " mincounts=" + str(mincts) +\
                      " oversample=3 rmfset=" + str(specrmfA) +\
                      " " + "backgndset=" + str(specsrcB) +\
                      " witharfset=yes arfset=" + str(specarfA) +\
                      " groupedset=" + str(groupspec)

                
                stringphasegen.append(c2)
                stringphasegen.append(c3)
                stringphasegen.append(c3a)
                stringphasegen.append(c3b)
                stringphasegen.append(c4)
                stringphasegen.append(c5)
                stringphasegen.append(c6)
                stringphasegen.append(c7)
                stringphasegen.append(c8)
                stringphasegen.append(c9)
                stringphasegen.append(c10)
                stringphasegen.append(c11)
                stringphasegen.append(c12)
                stringphasegen.append(c13)
                stringphasegen.append(c14)
                stringphasegen.append(c15)
                stringphasegen.append("")
                
    
    np.savetxt("phasegen.sh",stringphasegen,fmt='%s',delimiter='  ')
    
    newspkey = "epn*spec*grp*.fits"
    for newsp in glob.glob(newspkey):
            
        hdulistref = fits.open(newsp)
        header = hdulistref[2].header
        telapse = header['TELAPSE']
        obsid = newsp.split(".fits")[0].split("grp_")[1]
        vis = newsp.split("_grp")[0].split("_spec")[1]
                        
        newbkg = "xmm_phase_specB.fits"
        newrsp = "epn_" + obsid + "_A.rmf"
        newarf = "epn_" + obsid + "_A.arf"

        commkey1 = "fparkey " + newrsp + " " + newsp +\
                   "[1] RESPFILE"
        commkey2 = "fparkey " + newarf + " " + newsp +\
                   "[1] ANCRFILE"
        commkey3 = "fparkey " + newbkg + " " + newsp +\
                   "[1] BACKFILE"
        commkey4 = 'fparkey ' + str(telapse*pwidth) + " " + newsp +\
                   '[1] ' + 'EXPOSURE'
        commkey5 = 'fparkey ' + str(telapse*pwidth) + " " +\
                   newbkg + '[1] ' + 'EXPOSURE'
                
        os.system(commkey1)
        os.system(commkey2)
        os.system(commkey3)
        os.system(commkey4)
        os.system(commkey5)

        

    
    
    
