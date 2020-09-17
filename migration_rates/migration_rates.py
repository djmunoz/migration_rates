# Script module to compute the total "hot planet" migration rates resulting from 
# high-eccentricity migration integrating over the distribution of
# orbital parameters of the outer stellar companion
#
# Diego J. Munoz
# 2015
#
#
#
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
from scipy.optimize import fsolve
from numpy.random import rayleigh,uniform
from processing import Pool

#GLOBAL CONSTANTS
G = 2.959122082855911E-4
CLIGHT = 1.73144633E+2
Rsun = 4.6491E-3
Rjup = 4.6732617E-4
Rearth = 4.25875046E-5
Msun = 1.0
Mjup = 9.54265748E-4
Mearth = 3.002363E-6
k2rock = 0.3 #Love number for a rocky planet
kqrock= k2rock/3.0
k2gas = 0.37 #Love number for a gas giant
kqgas= 0.17

class migration_rates:
    def __init__(self, *args, **kwargs):
        #define the fixed parameters of the system
        self.m_in = kwargs.get("m_in")  # inner star mass
        self.m_out = kwargs.get("m_out") # outer star mass
        self.mp = kwargs.get("mp")   # planet mass
        self.eoutmin = kwargs.get("eoutmin") # outer companion eccentricity
        self.eoutmax = kwargs.get("eoutmax")
        self.logaoutmin = kwargs.get("logaoutmin") #range in companion semi-major axis
        self.logaoutmax = kwargs.get("logaoutmax")
        self.chi = kwargs.get("chi")
        self.rocky = kwargs.get("rocky") #rocky or gas planet
        self.MC_samples = kwargs.get("MC_samples") #number of random realization for Monte-Carlo integration
        #set default values
        if (self.m_in == None):
            self.m_in = 1.0 
        if (self.m_out == None):
            self.m_out = 1.0  
        if (self.mp == None):
            self.mp = 0.001 
        if (self.eoutmin == None):
            self.eoutmin = 0.0 
        if (self.eoutmax == None):  
            self.eoutmax = 0.8
        if (self.logaoutmin == None):
            self.logaoutmin = 2.0 
        if (self.logaoutmax == None):
            self.logaoutmax = 3.0
        if (self.chi == None):
            self.chi = 10  
        if (self.rocky == None):
            self.rocky = 0
        if (self.MC_samples == None):
            self.MC_samples = 500

    def compute_migration_rates_integrated(self,ain): 
        np.random.seed(41)
        Rp =  mass_radius_relation(self.mp,self.rocky)
        # Monte-Carlo integration
        Deltax = self.logaoutmax - self.logaoutmin
        Deltay = self.eoutmax - self.eoutmin
        logaout = uniform(self.logaoutmin,self.logaoutmax,self.MC_samples)
        eout = uniform(self.eoutmin,self.eoutmax,self.MC_samples)
        rates_mig,rates_dis = 0.0,0.0
        for k in range(self.MC_samples):
            rates = compute_migration_rates_octupole_estimate(ain,10**logaout[k],eout[k],self.m_in,self.mp,self.m_out,Rp,self.chi,self.rocky)
            rates_mig+=rates[0]
            rates_dis+=rates[1]
        return [rates_mig/self.MC_samples,rates_dis/self.MC_samples] #* Deltax * Deltay


    def compute_integrated_fractions(self,ain_vals):
        fraction_per_ain = [self.compute_migration_rates_integrated(a) for a in ain_vals]
        return  np.array(fraction_per_ain)

    def plot_integrated_fractions(self,ain,fractions,axes,lw=1.0,alpha=1.0,color1='b',color2='r',\
                                      legend=True,labels=True,logscale=False):
    
        label1=r'$F_\mathrm{HJ}$'
        line1, = axes.plot(ain,fractions[:,0]-fractions[:,1],color=color1,marker='.',ls='-',label=label1,lw=lw,alpha=alpha,markersize=8.0)
        label2=r'$F_\mathrm{dis}$'
        line2, = axes.plot(ain,fractions[:,1],color=color2,marker='.',ls='-',label=label2,lw=lw,alpha=alpha,markersize=8.0)
        
        if (legend): axes.legend(loc=2,prop={'size':16},borderpad=0.2,numpoints=1,handlelength=1)
        
        #additional labels
        if (labels):
            axes.text(0.03,0.74,r'$M_p=%g M_\mathrm{J}$' % (self.mp/Mjup),transform=axes.transAxes,size=20)
            axes.text(0.03,0.66,r'$R_p=%g R_\mathrm{J}$' % (mass_radius_relation(self.mp,self.rocky)/Rjup),transform=axes.transAxes,size=20)
            axes.text(0.03,0.58,r'$\chi=%g$' % self.chi,transform=axes.transAxes,size=20)
            
        #customize plot
        axes.set_yscale('log')
        if (logscale):
            axes.set_xscale('log')
            

        axes.set_xlim(ain.min(),ain.max())
        axes.set_ylim(0.0009,1.4)
        axes.set_xlabel(r'$a_{0}\,\mathrm{[AU]}$',size=16)
        axes.set_ylabel(r'migration fraction $F^\mathrm{LK}(a_0)$',size=20)
        
        yticks = axes.get_yticks()
        ytick_labels = [r"%g %%" % (100*tick) for tick in yticks]
        axes.set_yticklabels(ytick_labels)
        
        xticks = axes.get_xticks()
        xtick_labels = [r"%g" % (tick) for tick in xticks]
        axes.set_xticklabels(xtick_labels)
        
####################################################################



# OTHER FUNCTIONS ###################################################################

def radius_from_mass_seager(m):
    mm1 = 6.41 * Mearth
    rr1 = 3.19 * Rearth
    kk1, kk2, kk3 = -0.20945, 0.0804, 0.394
    ms = m/mm1
    return rr1*10.0**(kk1 + 0.33333 * np.log10(ms) - kk2 * ms**kk3)

def radius_from_kepler_data(m):
    return Rearth * (m/Mearth/2.7)**(1.0/1.3)

def radius_rocky_planets(m):
    return radius_from_mass_seager(m)
    #return 2*Rearth
    #return Rjup

def mass_radius_relation(m,rocky):
    if (rocky): return radius_rocky_planets(m)
    else: return Rjup

def lim_eccentricity_eq(ecc_squared,aininit,aout,eout,m1,m2,m3,R2,rocky):
    if (rocky):
        k2 = k2rock
        kq = kqrock
    else:
        k2 = k2gas
        kq = kqgas

    if (ecc_squared >= 1): ecc_squared = 1.0 - 1.0e-12
    if (ecc_squared <1) & (np.abs(ecc_squared) < 1.0e-8): ecc_squared = 0
    epsilon_kozai = ((m1+m2)/m3)*(aout/aininit)**3*(1 - eout**2)**1.5
    epsilon_GR = 3 * G  * (m1 + m2) / aininit / CLIGHT/ CLIGHT  * epsilon_kozai
    epsilon_tide = 15 * m1 / m2 * k2 * (R2/aininit)**5 * epsilon_kozai
    epsilon_rotprime = (m1 + m2) /m2 * kq * (R2/aininit)**5 * epsilon_kozai
    
    fps = (1 + 7.5 * ecc_squared + 5.625 * ecc_squared * ecc_squared +  0.3125 *  ecc_squared * ecc_squared * ecc_squared)/(1 + 3 * ecc_squared + 0.375 * ecc_squared * ecc_squared)/(1 - ecc_squared) /np.sqrt(1 - ecc_squared)  
    epsilon_rotprime,fps = 0,0
  
    jmin = np.sqrt(1 - ecc_squared)

    return epsilon_GR * (1.0 /jmin - 1)\
        + epsilon_tide/15 * (( 1 + 3 * ecc_squared + 0.375 * ecc_squared * ecc_squared)/jmin**9 -1) \
        + epsilon_rotprime * fps*fps / 3 * (1.0/jmin/jmin/jmin -1) \
        - 1.125 * ecc_squared 


def find_lim_ecc(aininit,aout,eout,m1,m2,R2):
    eguess = 0.99999
    elimquared= fsolve(lim_eccentricity_eq,eguess,args=(aininit,aout,eout,m1,m2,m3,R2,rocky),maxfev=6000,xtol=1.e-7)
    return elimquared

def lim_ecc_approx(aininit,aout,eout,m1,m2,m3,R2,rocky):
    if (rocky): k2 = k2rock
    else: k2 = k2gas

    Rdisrupt = 2.7 * R2 * (m1/m2)**0.3333
    epsilon_kozai = ((m1+m2)/m3)*(aout/aininit)**3*(1 - eout**2)**1.5
    epsilon_GR = 3 * G  * (m1 + m2) / aininit / CLIGHT/ CLIGHT  * epsilon_kozai
    epsilon_tide = 15 * m1 / m2 * k2 * (R2/aininit)**5 * epsilon_kozai
    
    elimsq_GR = 1 - (8.0/9 * epsilon_GR)**2
    elimsq_tide = 1 - (7.0/27 * epsilon_tide)**(2.0/9)
    elimsq_crit = 1 - (7.0/24 *epsilon_tide / epsilon_GR)**(0.25)

    if ((1- elimsq_GR) > 2.5*(1- elimsq_crit)):
        return elimsq_GR
    elif((1-elimsq_tide) < 0.2*(1- elimsq_crit)) :
        return elimsq_tide
    else:
        eguess = elimsq_GR
        elimsquared= fsolve(lim_eccentricity_eq,eguess,args=(aininit,aout,eout,m1,m2,m3,R2,rocky),maxfev=6000,xtol=1.e-7)[0]
        return elimsquared

def octupole_window(eps_oct):
    #use polynomial fit
    if (eps_oct < 1e-10): return 0
    eps_break = 0.05
    if (eps_oct > eps_break): eps_oct = eps_break
    
    coeff1,coeff2,coeff3,coeff4 = 2.60355,-53.559,12048.1,-167762.0
    coeff1,coeff2,coeff3,coeff4 = 2.60355*0.1,-53.559*(0.1)**2,12048.1*(0.1)**3,-167762.0*(0.1)**4
    coeff1,coeff2,coeff3,coeff4 = 0.26, -0.536, 12.05, -16.78
    
    cosIsq = (coeff1 * (eps_oct/0.1) \
                  +coeff2 * (eps_oct/0.1)**2 \
                  +coeff3 * (eps_oct/0.1)**3 + coeff4 * (eps_oct/0.1)**4)
    return cosIsq

def migration_solid_angle_fraction_octupole(ain,aout,eout):

    eps_oct = (ain/aout) * eout/(1.0 - eout**2)
    cosIsq = octupole_window(eps_oct)

    if (cosIsq < 0): cosIsq = 0
    return np.sqrt(cosIsq)


def migration_solid_angle_fraction_quadrupole(ecc_sq,ain,aout,eout,m1,m2,m3,R2,rocky):
    if (rocky): k2 = k2rock
    else: k2 = k2gas

    epsilon_kozai = ((m1+m2)/m3)*(aout/ain)**3*(1 - eout**2)**1.5
    epsilon_GR = 3 * G  * (m1 + m2) / ain / CLIGHT/ CLIGHT  * epsilon_kozai
    epsilon_tide = 15 * m1 / m2 * k2 * (R2/ain)**5 * epsilon_kozai
    
    jmin = np.sqrt(1 - ecc_sq)
    cosIsq= 0.6*jmin*jmin*(1.0- 1.0/1.125/ecc_sq*\
                               (epsilon_GR * (1.0 /jmin - 1)\
                                    + epsilon_tide/15 * (( 1 + 3 * ecc_sq + 0.375 * ecc_sq * ecc_sq)/jmin**9 -1)))
    
    if (cosIsq < 0): cosIsq = 0
    return np.sqrt(cosIsq)



def compute_migration_rates_octupole_estimate(ain,aout,eout,m1,m2,m3,R2,chi,rocky):
    if (rocky): k2 = k2rock
    else: k2 = k2gas

    Rdisrupt = 2.7 * R2 * (m1/m2)**0.33333
    xi = 0.0307
    xi = 0.024
    Rmig = xi * (R2/Rjup)**(5.0/7) * (chi/10.0)**(1.0/7) * (m2/Mjup)**(-1.0/7) * (k2/0.37)**(1.0/7)* (m1/Msun)**(2.0/7) * ain**(-1.0/7)

    #for fixed aout, eout, find ain limits for succesfull planet migration
    # find elim
    #elim_sq = find_lim_ecc(ain,aout,eout)[0] # exact solution
    elim_sq = lim_ecc_approx(ain,aout,eout,m1,m2,m3,R2,rocky) #approximate solution
    # eccentricity required for migration (Anderson et al 2015)
    esq_req_mig = (1.0 - Rmig/ain)**2
    esq_req_dis = (1.0 - Rdisrupt/ain)**2

    rate_mig,rate_dis = 0,0
    if (elim_sq < esq_req_mig) & (elim_sq < esq_req_dis) : return rate_mig,rate_dis
    #if (elim_sq < esq_req_mig): return rate_mig,rate_dis

    # eccentricity required for tidal disruption
    if (elim_sq > esq_req_dis): 
        #if rate is non-zero, return solid angle
        rate_mig = migration_solid_angle_fraction_octupole(ain,aout,eout)
        rate_dis = rate_mig
    else:
        #if rate is non-zero, return solid angle
        rate_mig = migration_solid_angle_fraction_octupole(ain,aout,eout)
        rate_dis = 0

    if (migration_solid_angle_fraction_quadrupole(esq_req_mig,ain,aout,eout,m1,m2,m3,R2,rocky) > rate_mig):
        rate_mig = migration_solid_angle_fraction_quadrupole(esq_req_mig,ain,aout,eout,m1,m2,m3,R2,rocky)
    if (migration_solid_angle_fraction_quadrupole(esq_req_dis,ain,aout,eout,m1,m2,m3,R2,rocky) > rate_dis):
        rate_dis = migration_solid_angle_fraction_quadrupole(esq_req_dis,ain,aout,eout,m1,m2,m3,R2,rocky)

    #if efficient migration is simply impossible
    if (esq_req_dis < esq_req_mig):
        rate_mig = rate_dis
        #rate_dis = rate_mig
        
    return rate_mig,rate_dis

####################################################################
