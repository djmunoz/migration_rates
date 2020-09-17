#!/usr/bin/env python
# -*- coding: utf-8 -*-


import migration_rates as mr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import transforms as tf
from scipy.interpolate import interp1d
from scipy.integrate import quad
import scipy.special as spec
import time
import glob
from string import split
import itertools
from joblib import Parallel, delayed

def F_integrand(xi):
    return (spec.ellipk(xi)-2*spec.ellipe(xi))/(41*xi-21)/np.sqrt(2*xi +3)
def F(x): #Katz et al (2011) special function
    upper=1
    lower = (3.0-3*x)/(3.0 + 2 *x)
    return 32*np.sqrt(3)/np.pi*quad(F_integrand,lower,upper)[0]
def epsilon_crit(cosI_0,e_0,omega_0):
    CK_0 = e_0*e_0 *(1 - 2.5 * (1- cosI_0**2)* np.sin(omega_0)**2)
    jz_0_sq = (1 - e_0 * e_0) * cosI_0 * cosI_0
    Fmax = 0.0475
    if (CK_0 + 0.5 * jz_0_sq < 0.112) | (CK_0 > 0.112):
        return 0.5*np.abs(F(CK_0 + 0.5 * jz_0_sq) -F(CK_0))
    else:
        return 0.5*max(Fmax - F(CK_0),np.abs(F(CK_0 + 0.5 * jz_0_sq) -F(CK_0)))


#############################################


def plot_fig1(figname):
    I0arr = np.linspace(36,91,200)
    cosI0arr = np.cos(I0arr*np.pi/180)
    eps = [epsilon_crit(cosI0,0.001,0.0) for cosI0 in cosI0arr]
    epsarr=np.logspace(np.log10(1e-4),np.log10(0.11),200)

    filename='critical_angle_data.txt' # Bin Liu's data
    data=np.loadtxt(filename,skiprows=10)

    ################################
    print "Opening figure...",figname
    fig=plt.figure(1,figsize=(10,7.5))
    ax=fig.add_subplot(111)
    fig.subplots_adjust(right=0.97,top=0.97,bottom=0.13,left=0.08)

    #plot data from Katz et al
    ax.plot(eps,I0arr,color='b',lw=2.0,ls='-')

    epsarr = np.append(1.0e-8,epsarr)
    ax.plot(epsarr,180/np.pi*np.arccos(np.sqrt(np.vectorize(mr.octupole_window)(epsarr))),color='red',lw=3.0)
    datalabel="Liu, Mu"+u'ñ'+"oz & Lai (2015)"
    ax.plot(data[:,0],data[:,1],'ro',ms=7.0,label=datalabel)

    crit_curv=interp1d(180/np.pi*np.arccos(np.sqrt(np.vectorize(mr.octupole_window)(epsarr))),epsarr)
    ax.fill_betweenx(np.append(90,180/np.pi*np.arccos(np.sqrt(np.vectorize(mr.octupole_window)(epsarr)))),
                     np.append(epsarr.min(),epsarr),
                     0.1,color='orange',alpha=0.3,edgecolor='black',lw=2.0,hatch='//')
    
    ax.text(0.36,0.12,'Katz et al (2011) -- analytic, no SRF',transform=ax.transAxes,
            size=12,color='b')
    ax.arrow(0.07,0.9,0.15,0.0,transform=ax.transAxes,fc='r',ec='r',color='r',
             head_width=0.03,head_length=0.02,lw=3.0)
    ax.text(0.07,0.93,'region of extreme eccentricity',transform=ax.transAxes,
            size=20,color='r')
    ax.text(0.03,0.55,'moderate eccentricity',transform=ax.transAxes,
            size=14,color='k')
    ax.text(0.36,0.37,r'$i_\mathrm{lim}(\epsilon_\mathrm{oct})$',transform=ax.transAxes,
            size=24,color='r')
    
    ax.legend(loc='upper right',numpoints=1,prop={'size':14},borderpad=0.3)
    
    #customize plot
    ax.set_xlim(0,0.07)
    ax.set_ylim(36,90)
    ax.set_ylabel(r'$i_0 \mathrm{[^\circ]}$',size=24,labelpad=0)
    ax.set_xlabel(r'$\epsilon_\mathrm{oct}$',size=28)
    [tick.label.set_fontsize(20) for tick in ax.yaxis.get_major_ticks()]
    [tick.label.set_fontsize(20) for tick in ax.xaxis.get_major_ticks()]
    #save figure
    print "\nSaving figure...",figname,"\n"
    fig.savefig(figname)
    fig.clf()

def plot_fig2and3(figname,mp,chi):
    Rp=1.0
    #define companion eccentricities to be explored
    eout_arr = [0.0,0.2,0.4,0.8]
    #set up multipanel plot
    print "Opening figure...",figname
    fig = plt.figure(figsize=(14.0,5.5))
    fig.subplots_adjust(top=0.88,bottom=0.11,right=0.99,left=0.08,hspace=0.05,wspace=0.0)
    #define parameter range
    nx = 100
    logainmin,logainmax = -1.1,0.85
    ain_arr = np.linspace(10**logainmin,10**logainmax,nx)
    for k,eout in enumerate(eout_arr):
        aout = 200.0/np.sqrt(1.0-eout**2) # outer companion semimajor axis in AU
        eps_oct = ain_arr/aout * eout/(1 - eout**2)
        migration_fraction=np.zeros(nx)
        disruption_fraction=np.zeros(nx)
        for kk in range(nx): migration_fraction[kk],disruption_fraction[kk]= (mr.compute_migration_rates_octupole_estimate)(ain_arr[kk],aout,eout,1.0,mp*mr.Mjup,1.0,Rp*mr.Rjup,chi,0)
        hotplanet_fraction = migration_fraction - disruption_fraction

        print "      Plotting frame %i of %i" % (k+1,len(eout_arr))
        ax= fig.add_subplot(1,len(eout_arr),k+1)
        ax.plot(ain_arr,migration_fraction,color='k',alpha=0.7,lw=2.6,label=r'$f_\mathrm{mig}$')
        ax.plot(ain_arr,hotplanet_fraction,color='b',lw=1.5,label=r'$f_\mathrm{HJ}$')
        ax.plot(ain_arr,disruption_fraction,color='red',lw=1.5,label=r'$f_\mathrm{dis}$')
        ax.legend(loc=2,prop={'size':14},borderpad=0.2)

        hfont = {'fontname':'Comic Sans MS'}

        print "      Area under blue curve",np.trapz(hotplanet_fraction,x=ain_arr)

        #additional labels
        ax.text(0.5,0.93,r'$a_\mathrm{out}=%i\,\mathrm{AU}$' % (aout),transform=ax.transAxes,size=16)
        ax.text(0.5,0.86,r'$e_\mathrm{out}=%g$' % (eout),transform=ax.transAxes,size=16)
        if (k == len(eout_arr)-1):
            ax.text(0.53,0.22,r'$M_p=%g M_\mathrm{J}$' % (mp),transform=ax.transAxes,size=20)
            ax.text(0.53,0.12,r'$R_p=%g R_\mathrm{J}$' % (Rp),transform=ax.transAxes,size=20)
        #customize plot
        ax.set_yscale('log')
        ax.set_xlim(ain_arr.min(),ain_arr.max())
        ax.set_ylim(0.007,1.6)
        ax.set_xlabel(r'$a_0\,\mathrm{[AU]}$',size=18)
        if (k==0):
            ax.set_ylabel(r'migration fraction $f^\mathrm{ LK}$',size=18)
        yticks = ax.get_yticks()
        ytick_labels = [r"%i %%" % (100*tick) for tick in yticks]
        if (k ==0):
            ax.set_yticklabels(ytick_labels,size=14)
        else:
            ax.set_yticklabels('')
        xticks = ax.get_xticks()
        xtick_labels = [r"%g" % (tick) for tick in xticks]
        ax.set_xticklabels(xtick_labels,size=14)
        
        #check for data points that were left out of the plotting range
        X0,X1 = ax.get_xlim()
        Y0,Y1 = ax.get_ylim()
        arrow=u'$\u2193$'
        ms_arrow = 10.0
        ms_arrow_coord = (ax.transData.inverted().transform((0,ms_arrow))-ax.transData.inverted().transform((0,0)))[1]
        mark_align_bottom = tf.offset_copy(ax.get_yaxis_transform(),fig,0, units='points')

        #add additional horizontal axis
        if (eout > 0):
            ax2 = ax.twiny()
            ax2.plot(eps_oct,migration_fraction,color='none')
            ax2.set_xlim(eps_oct.min(),eps_oct.max())
            ax2.set_ylim(Y0,Y1)
            ticks = ax2.get_xticks()
            ax2.xaxis.major.locator.set_params(nbins=4)
            ax2.set_xlabel(r'$\epsilon_\mathrm{oct}$',size=24,labelpad=9)
        elif (eout == 0):
            ax.tick_params('x',top='off')
            

    #save figure
    print "\nSaving figure...",figname,"\n"
    fig.savefig(figname)
    fig.clf()
    
    
def plot_fig4and5(figname,chi,rocky=0,read_data=0,parallel=0):
    
    if (read_data):
        #identify data files
        file_base="migration_fractions_chi"    
        file_list =sorted(glob.glob(file_base+"%i_*"%chi))
        planet_mass = [float(split(split(f,"mp")[1],".txt")[0]) for f in file_list]
    else:
        planet_mass = [0.5,1.0,3.0]
    Ncases = len(planet_mass)

    #prepare plot
    print "Opening figure...",figname
    figsize=(8.5,6)
    fig=plt.figure(1,figsize=figsize)
    fig.subplots_adjust(bottom=0.12,left=0.14,right=0.99,top=0.999)
    ax=fig.add_subplot(111)


    for jj in range(Ncases):
        rates = mr.migration_rates(mp=planet_mass[jj]*mr.Mjup,chi=chi,rocky=rocky)
        print rates.mp,planet_mass[jj],rates.chi
        if (read_data):
            print "Reading file",file_list[jj]
            data = np.loadtxt(file_list[jj],skiprows=0)
            ain=data[:,0]
            fhp = data[:,1]
            fdis = data[:,2]
            fraction_per_ain = np.vstack((fhp+fdis,fdis)).T
        else:
            rates.MC_samples=1000
            ain = np.linspace(0.1,7.0,30)
            if (parallel):
                fraction_per_ain = Parallel(n_jobs=12)(delayed(rates.compute_migration_rates_integrated)(a) for a in ain)
                fraction_per_ain=np.array(fraction_per_ain)
            else:
                fraction_per_ain = rates.compute_integrated_fractions(ain)

        print fraction_per_ain.shape

        
        co1 = cm.Blues(0.55+jj*0.4/len(planet_mass))
        co2 = cm.Reds(0.55+jj*0.4/len(planet_mass))
        
        lw = 1.4 + jj*2.8/len(planet_mass)
        al = 1.0 - jj*0.3/len(planet_mass)
        rates.plot_integrated_fractions(ain,fraction_per_ain,ax,lw=lw,alpha=al,
                                        color1=co1,color2=co2,labels=False,legend=False,logscale=False)
        

    if (rocky): ax.text(0.77,0.92,r'$\chi=%g$' % chi,transform=ax.transAxes,size=28)
    else:
        ax.text(0.81,0.83,r'$R_p=%g R_\mathrm{J}$' % (mr.mass_radius_relation(1.0,0)/mr.Rjup),transform=ax.transAxes,size=20)
        ax.text(0.81,0.92,r'$\chi=%g$' % chi,transform=ax.transAxes,size=28)
    
    #customize legend
    def flip(items,ncol):return itertools.chain(*[items[i::ncol] for i in range(ncol)])
    leg=ax.legend(loc='upper left',ncol=2,numpoints=1,prop={'size':18})
    handles, labels = ax.get_legend_handles_labels()
    labels[0:len(labels):2]  = np.repeat(u'$M_p$',len(labels[0:len(labels):2]))
    for kk in range(0,len(labels)):
        if (((kk - 1)) % 2 == 0): 
            if (rocky): labels[kk]=u'$M_p=%gM_\oplus$' % np.around(planet_mass[(kk-1)/2]*mr.Mjup/mr.Mearth,decimals=2)
            else: labels[kk]=u'$M_p=%gM_\mathrm{J}$' % planet_mass[(kk-1)/2]

    if (rocky):
        leg=ax.legend(flip(handles,2),flip(labels,2),loc='upper left',ncol=2,numpoints=1,labelspacing=0.6,
                      columnspacing=-0.9,title=r'$F_\mathrm{hp}$  $F_\mathrm{dis}$             ',
                      prop={'size':12},handletextpad=0.6,handleheight=0.1,handlelength=1.7,borderpad=0.35)
    else:
        leg=ax.legend(flip(handles,2),flip(labels,2),loc='upper left',ncol=2,numpoints=1,labelspacing=0.6,
                      columnspacing=-0.9,title=r'$F_\mathrm{HJ}$  $F_\mathrm{dis}$             ',
                      prop={'size':14},handletextpad=0.6,handleheight=0.1,handlelength=1.7,borderpad=0.35) 
    leg.get_title().set_fontsize('16')
    for kk in range(len(leg.get_texts())/2): plt.setp(leg.get_texts()[kk],color='w')

    #customize axes
    ax.set_ylabel(r'migration fraction $F^\mathrm{LK}(a_0)$',size=22)
    [tick.label.set_fontsize(14) for tick in ax.yaxis.get_major_ticks()]
    [tick.label.set_fontsize(14) for tick in ax.xaxis.get_major_ticks()]
    ax.tick_params('x',length=8,which='major')

        
    #save figure
    print "\nSaving figure...",figname,"\n"
    fig.savefig(figname)
    fig.clf()




if __name__=="__main__":
    

    plot_fig1("figure1.pdf")
    plot_fig2and3("figure2.pdf",0.3,10)
    plot_fig2and3("figure3.pdf",3.0,10)
    plot_fig4and5("figure4a.pdf",1,rocky=0,read_data=1)
    plot_fig4and5("figure4a_light.pdf",1,rocky=0,read_data=0,parallel=1)
    plot_fig4and5("figure4b.pdf",10,rocky=0,read_data=1)
    plot_fig4and5("figure4c.pdf",100,rocky=0,read_data=1)
    plot_fig4and5("figure5.pdf",6000,rocky=1,read_data=1)
    



