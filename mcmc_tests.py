import numpy as np
import matplotlib.pyplot as plt
from MCMC import MCMC
from scipy.stats import sem
import pyCloudy as pc
from PCloudy import PCloudy
plt.rc('font',family='serif')
plt.rc('text',usetex=True)
def poly(x, pars):
    return np.sum(np.array([ pars[i]*x**(len(pars)-1- i) for i in range(len(pars))]), axis=0)

def polyperiodic(x, pars):
    return pars[0]*np.cos(pars[1]*x)+np.sum(np.array([ pars[i]*x**(len(pars)-1- i) for i in range(2,len(pars))]), axis=0)
def per(x, pars):
    return pars[0]*np.sin(x*pars[1])

'''Metropolis Hastings'''
x_vals = np.linspace(-15,15, 400)

period = [1.1,0.7 ]

linperiod = [2, 4, 2, 2.2]
linpars = [3.7, 2.2]
quadpars = [-1.1,2.3,-3.5]
cubicpars = [-2.1, 2.3, 4.5, -10]
quarticpars = [4.2,1.1, -4.3, -2.1, 14]
quinticpars = [-0.8, -6.2, 1.4, 3.2, 2.1, 4.8]

linper_true = polyperiodic(x_vals, linperiod)
per_true = per(x_vals, period)
lin_true = poly(x_vals, linpars)
quad_true = poly(x_vals,quadpars)
cubic_true = poly(x_vals,cubicpars)
quartic_true = poly(x_vals,quarticpars)
quintic_true = poly(x_vals,quinticpars)

linper_comp = linper_true+np.random.normal(scale=0.1,size=len(linper_true))
per_comp = per_true+np.random.normal(scale=0.2,size=len(per_true))
lin_comp = lin_true + np.random.normal(size=len(lin_true))
quad_comp = quad_true+np.random.normal(size=len(x_vals))
cubic_comp = cubic_true+np.random.normal(size=len(x_vals))
quartic_comp = quartic_true+np.random.normal(size=len(x_vals))
quintic_comp = quintic_true+np.random.normal(size=len(x_vals))


def test_n_chains(plot_n=False, plot_chi=False):
    n_chain_arr= np.int64(10**(np.arange(2,5.5,0.5)))
    runtimes = []
    runs = []
    for n_chain in n_chain_arr:
        lm = MCMC(poly, x_vals, lin_comp, [2,2],[[-10,10],[-30,30]],n_iter=100, n_chains = n_chain )    
        lm.run_mcmc()
        runs.append(lm)
        runtimes.append(lm.time_total)
    runtimes = np.array(runtimes)
    
    bs = []
    ms = []
    chi_min = []
    for mc in runs:
        finpars = mc.finalpars
        bs.append([finpars[1]])
        ms.append([finpars[0]])
        chi_min.append(np.min(mc.all_probs))
        
    ms = np.array(ms)
    mdiffs = np.abs(ms - linpars[0])
    bs = np.array(bs)
    bdiffs = np.abs(bs - linpars[1])
    chi_min =np.array(chi_min)
    if plot_chi:
        
        plt.scatter(np.log10(n_chain_arr), ms, color='k' )
        plt.axhline(y=linpars[0],label='True m')
        plt.xlabel('log(N)')
        plt.ylabel('m-value')       
        plt.legend(frameon=False, fontsize=20)
        plt.tight_layout()
        

        plt.scatter(np.log10(n_chain_arr), bs, color='k' )
        plt.axhline(y=linpars[1],label='True b')
        plt.xlabel('log(N)')
        plt.ylabel('b-value')
        plt.legend(frameon=False, fontsize=20)
        plt.tight_layout()
        
        plt.scatter(np.log10(n_chain_arr), -np.log10(-chi_min), color='k' )
        plt.xlabel('log(N)')
        plt.ylabel(r'log($\chi^2$ min)')
        plt.legend(frameon=False, fontsize=20)
        plt.tight_layout()
        
        
    if plot_n:
        plt.scatter(np.log10(n_chain_arr), np.log10(runtimes), color='k' )
        m, b = np.polyfit(np.log10(n_chain_arr), np.log10(runtimes), 1)
        plt.plot(np.log10(n_chain_arr),m*np.log10(n_chain_arr)+b,'k-.', label='m = '+str(m)[0:5])
        plt.xlabel('log(N)')
        plt.ylabel('log(Run-time) [s]')
        plt.legend(frameon=False, fontsize=20)
        plt.tight_layout()

def test_n_iter(plot_n=False, plot_chi=False):
    n_iter_arr= np.int64(10**(np.arange(1,4.5,0.5)))
    runtimes = []
    runs = []
    for n_iter in n_iter_arr:
        lm = MCMC(poly, x_vals, lin_comp, [2,2],[[-10,10],[-30,30]],n_iter=n_iter, n_chains = 1000 )    
        lm.run_mcmc()
        runs.append(lm)
        runtimes.append(lm.time_total)
    runtimes = np.array(runtimes)
    
    bs = []
    ms = []
    chi_min = []
    for mc in runs:
        finpars = mc.finalpars
        bs.append([finpars[1]])
        ms.append([finpars[0]])
        chi_min.append(np.min(mc.all_probs))
        
    ms = np.array(ms)
    mdiffs = np.abs(ms - linpars[0])
    bs = np.array(bs)
    bdiffs = np.abs(bs - linpars[1])
    chi_min =np.array(chi_min)
    if plot_chi:
        
        plt.scatter(np.log10(n_iter_arr), ms, color='k' )
        plt.axhline(y=linpars[0],label='True m')
        plt.xlabel('log(N)')
        plt.ylabel('m-value')       
        plt.legend(frameon=False, fontsize=20)
        plt.tight_layout()
        

        plt.scatter(np.log10(n_iter_arr), bs, color='k' )
        plt.axhline(y=linpars[1],label='True b')
        plt.xlabel('log(N)')
        plt.ylabel('b-value')
        plt.legend(frameon=False, fontsize=20)
        plt.tight_layout()
        
        plt.scatter(np.log10(n_iter_arr), -np.log10(-chi_min), color='k' )
        plt.xlabel('log(N)')
        plt.ylabel(r'log($\chi^2$ min)')
        plt.legend(frameon=False, fontsize=20)
        plt.tight_layout()
            
    if plot_n:
        plt.scatter(np.log10(n_iter_arr), np.log10(runtimes), color='k' )
        m, b = np.polyfit(np.log10(n_iter_arr), np.log10(runtimes), 1)
        plt.plot(np.log10(n_iter_arr),m*np.log10(n_iter_arr)+b,'k-.', label='m = '+str(m)[0:5])
        plt.xlabel('log(N)')
        plt.ylabel('log(Run-time) [s]')
        plt.legend(frameon=False, fontsize=20)
        plt.tight_layout()
   
def test_mods():
    n_iter=100
    n_chains=10000

    linperlims = [[1.,3.],[3,5],[1.,3.],[1.,3.]]
    linpermcmc= MCMC(polyperiodic,x_vals, linper_comp, [2.5,3,2,2], linperlims,n_iter=n_iter, n_chains=n_chains)
    linpermcmc.run_mcmc()
    
    perlims = [[1.0,1.3],[0.6,0.9]]
    permcmc= MCMC(per, x_vals, per_comp, [1,1], perlims, n_iter=n_iter, n_chains=n_chains)
    permcmc.run_mcmc()
        
    linlims = [[-10,10],[-30,30]]
    linmcmc = MCMC(poly,x_vals, lin_comp, [2,2],linlims, n_iter=n_iter, n_chains=n_chains)
    linmcmc.run_mcmc()
    
    
    quadlims = [[-10,10],[-10,10],[-30,30]]
    quadmcmc = MCMC(poly, x_vals, quad_comp, [1,1,1], quadlims, n_iter=n_iter, n_chains=n_chains)
    quadmcmc.run_mcmc()
    
    cubiclims = [[-10,10],[-10,10],[-10,10],[-30,30]]
    cubicmcmc = MCMC(poly, x_vals, cubic_comp, [1,1,1,1], cubiclims, n_iter=n_iter, n_chains=n_chains)
    cubicmcmc.run_mcmc()
    
    quarticlims = [[-10,10],[-10,10],[-10,10],[-10,10],[-30,30]]
    quarticmcmc = MCMC(poly, x_vals, quartic_comp, [1,1,1,1,1], quarticlims, n_iter=n_iter, n_chains=n_chains)
    quarticmcmc.run_mcmc()
    
    quinticlims = [[-10,10],[-10,10],[-10,10],[-10,10],[-10,10],[-30,30]]
    quinticmcmc = MCMC(poly, x_vals, quintic_comp, [1,1,1,1,1,1], quinticlims, n_iter=n_iter, n_chains=n_chains)
    quinticmcmc.run_mcmc()
    
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.tight_layout()


def test_cloudy():


    dens = 2 #log cm^-3
    Teff = 45000. #K
    qH = 47. #s-1, ionizing phot /second
    r_min = 5e17 #cm, 
    dist = 1.26 #kpc, how far is the region from us
    cloudypars = np.array([dens, Teff, qH, r_min, dist])
    pclo = PCloudy(cloudypars)
    pclo.cloudy()
    Mod = pclo.Mod
    wl = Mod.get_cont_x(unit='Ang')
    
    intens = Mod.get_cont_y(cont='ntrans',unit='Jy')
    opt=np.where((wl>4000) & (wl<8000))[0]
    wl_opt = wl[opt]
    intens_opt = intens[opt]
    initpars = cloudypars + np.random.normal(size=5)
    cloudylims = [[1,3],[40000,50000],[46,49],[1e17,1e18],[1,2]]
    cloudyMCMC = MCMC(cloudy_model, wl_opt, intens_opt, initpars, cloudylims, n_iter=10, n_chains=100, sample_rate=1)
    cloudyMCMC.run_mcmc()
def cloudy_model(wl_ran, pars):
    pclo = PCloudy(pars)
    pclo.cloudy()
    Mod = pclo.Mod
    wl = Mod.get_cont_x(unit='Ang')
    intens = Mod.get_cont_y(cont='ntrans',unit='Jy')
    opt=np.where((wl>4000) & (wl<8000))[0]
    intens_opt = intens[opt]
    return intens_opt