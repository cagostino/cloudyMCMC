import numpy as np
import matplotlib.pyplot as plt
import corner
import time
class MCMC:
    def __init__(self, model, xreal, yreal, initpars, parlims, n_iter=100, n_chains=1000, sample_rate=10):
        '''
        n_iter: number of iterations
        parlims: limits on parameter space for efficiency
        chain: chain where mcmc samples are stored
        target prior: used for deciding stuff
        '''
        self.xreal = xreal
        self.model = model
        self.n_iter = n_iter
        self.initpars = initpars
        self.yreal = yreal
        self.dofs = len(self.yreal )-len(self.initpars)
        self.parlims = parlims
        self.n_pars = np.shape(parlims[0])
        self.n_chains = n_chains
        self.sample_rate = sample_rate 
    def run_mcmc(self):
        self.start_time = time.time() 
        all_samples = np.array([])
        all_probs = np.array([])
        all_finpars = np.array([])
        chains = []
        chainprobs = []
        chainpars = []
        self.initchainpar = self.initpars
        for i in range(self.n_chains):
            if i%100 ==0:
                print('n_chain: ', i)
            probs = []
            par_fin, samples, probs = self.run_chain()
            chains.append(samples)
            chainprobs.append(probs)
            all_samples = np.append(all_samples, samples)
            all_probs = np.append(all_probs, probs)
            all_finpars = np.append(all_finpars,par_fin)
            self.initchainpar = [np.random.uniform(self.parlims[i][0], self.parlims[i][1])  for i in range(len(self.parlims))]
            chainpars.append(par_fin)
        self.end_time = time.time()
        self.time_total = self.end_time-self.start_time
        self.all_samples = all_samples.reshape((int(np.size(all_samples)/len(self.initpars)), len(self.initpars))).transpose()
        self.all_probs = all_probs
        self.all_finpars = all_finpars 
        self.chains = chains
        self.chainprobs = chainprobs
        self.chainpars = chainpars
        self.finalpars = np.array([np.average(self.all_samples[i], weights = -1/self.all_probs) for i in range(len(self.all_samples))])
    def run_chain(self):
        probs = []
        samples = []
        par = self.initchainpar
        self.ymod = self.model(self.xreal,par)        
        p = self.chisq(self.ymod,self.yreal)
        for i in range(self.n_iter):
            parn = par + np.random.normal(size=len(par))
            parlimbools = [parn[i]>self.parlims[i][0] and parn[i]<self.parlims[i][1] for i in range(len(self.parlims))]
            if not all(parlimbools):
                continue
            self.ymod = self.model(self.xreal, parn)
            pn = self.chisq(self.ymod, self.yreal)
            if pn >= p:
                p = pn
                par = parn
            else:
                u = np.random.rand()
                if u < pn/p:
                    p = pn
                    par = parn
            if i % self.sample_rate == 0:
                samples.append(par)
                probs.append(p)
        return par,samples, probs
    def chisq(self, ymod, yreal):
        '''
        compares the model distribution to the
        real distribution        
        '''
        return -np.sum((ymod-yreal)**2)

    def plotfit(self):
        plt.plot(self.xreal,self.model(self.xreal, self.finalpars),'k-.',label='Fit')
        plt.plot(self.xreal,self.yreal,'r-', label='Data')
        plt.legend(frameon=False, fontsize=20)
    def plotmcmc_corner(self):
        npars = len(self.parlims)
        parnames = ['a'+str(i+1) for i in range(npars)]
        
        figure = corner.corner(self.all_samples.transpose(), labels=parnames,
                           weights=-1/(self.all_probs),
                           quantiles = [0.16, 0.84],
                           show_titles=True,
                           title_kwargs={"fontsize": 12})
