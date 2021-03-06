import migration_rates as mr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad
import time
import sys

if __name__=="__main__":
    np.random.seed(41)
    # COMMAND-LINE PARAMETERS
    mp  = float(sys.argv[1]) * mr.Mjup # planet mass in Jupiter masses
    chi = float(sys.argv[2]) # dimensionless constant time lag
    rocky = int(sys.argv[3]) # rocky or gas planet

    Nsamples = 500
    rates = mr.migration_rates(mp=mp,chi=chi,rocky=rocky,MC_samples=Nsamples)
    
    t0 = time.time()
    USE_LOG_SAMPLING = 1
    
    if (USE_LOG_SAMPLING):
        ain = np.logspace(-1.0,0.85,60)
    else:
        ain = np.linspace(0.1,7.0,80)
        
    #ain = np.linspace(4.5,5.5,40) #testing
    lower_ain,upper_ain=1.0,5.0   
    fraction_per_ain = rates.compute_integrated_fractions(ain)
    print "Total time:",time.time()-t0

    print "Integrated values..."
    print fraction_per_ain.shape
    fraction_density_hp = interp1d(ain,fraction_per_ain[:,0]-fraction_per_ain[:,1])
    fraction_density_dis = interp1d(ain,fraction_per_ain[:,1])
    integ_fraction_hp= quad(fraction_density_hp,lower_ain,upper_ain,limit=200)[0]/(upper_ain-lower_ain)
    integ_fraction_dis= quad(fraction_density_dis,lower_ain,upper_ain,limit=200)[0]/(upper_ain-lower_ain)
    
    print "Fractions: %.3g %%(HP), %.3g %%(dis)" % (integ_fraction_hp*100,integ_fraction_dis*100)
 
    ###########################################
    #PLOTTING

    #prepare plot 
    fig=plt.figure(1,figsize=(8.5,6))
    ax=fig.add_subplot(111)

    rates.plot_integrated_fractions(ain,fraction_per_ain,ax)
       
    ax.text(0.15,0.2,r'$f_\mathrm{HJ}=%.3g$%%' %(integ_fraction_hp*100),
            transform=ax.transAxes,size=16,color='blue')
    ax.text(0.8,0.8,r'$f_\mathrm{dis}=%.3g$%%' % (integ_fraction_dis*100),
            transform=ax.transAxes,size=16,color='red')


    

    #save figure
    if (rocky):
        fig.savefig("migration_fraction_integrated_rocky_mplanet%g_chi%g.pdf" % ((mp/mr.Mjup),chi))
    else:
        fig.savefig("migration_fraction_integrated_mplanet%g_chi%g.pdf" % ((mp/mr.Mjup),chi))
