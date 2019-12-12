import numpy as np
import pyCloudy as pc
class PCloudy:
    
    def __init__(self, pars ):
        '''
        Program for setting up parameters for pyCloudy to compute
        properties of nebulae.
        '''
        self.numdensity = pars[0]
        self.Teff = pars[1]
        self.qH = pars[2]
        self.r_min = pars[3]
        self.dist = pars[4]
        self.dir_ = './models/'
        self.model_name = 'model_test'
        self.full_model_name = '{0}{1}'.format(self.dir_, self.model_name)
        self.options = ('no molecules',
                        'no level2 lines',
                        'no fine opacities',
                        'atom h-like levels small',
                        'element limit off -8')    
        self.emis_tab = ['H  1  4861.33A', #hbeta
                        'H  1  6562.81A', #halpha
                        'N  2  6583.45A', #nii
                        'O  3  5006.84A' ] #oiii
        self.abund = {'He' : -0.92, 'C' : 6.85 - 12, 'N' : -4.0, 'O' : -3.40, 'Ne' : -4.00, 
         'S' : -5.35, 'Ar' : -5.80, 'Fe' : -7.4, 'Cl' : -7.00}
        c_input = pc.CloudyInput(self.full_model_name)
        c_input.set_BB(Teff = self.Teff, lumi_unit = 'q(H)', lumi_value = self.qH)
        c_input.set_cste_density(self.numdensity)    
        c_input.set_radius(r_in=np.log10(self.r_min))
        c_input.set_abund(ab_dict = self.abund, nograins = True)
        c_input.set_other(self.options)
        c_input.set_iterate() 
        c_input.set_sphere() # 
        c_input.set_emis_tab(self.emis_tab ) #
        c_input.set_distance(dist=self.dist, unit='kpc', linear=True) 
        c_input.print_input(to_file = True, verbose = False)
        self.c_input = c_input

    def cloudy(self):
        '''
        Computes a Cloudy Model for the parameters specified in 
        '''
        pc.log_.message('Running {0}'.format(self.model_name ), calling = 'test1')
        pc.config.cloudy_exe = '~/cloudy/source/sys_gcc/cloudy.exe'
        pc.log_.timer('Starting Cloudy', quiet = True, calling = 'test1')
        self.c_input.run_cloudy()
        self.Mod = pc.CloudyModel(self.full_model_name)
        pc.log_.timer('Cloudy ended after seconds:', calling = 'test1')

    
    