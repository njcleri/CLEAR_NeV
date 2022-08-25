import numpy as np
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u
class NeVFunctions:
    def __init__(self):
        pass
    
    @staticmethod
    def vo87(logsiiha):
        '''
        Defining the unVO87 AGN/SF dividing line from Veilleux & Osterbrock 1987.
        Singularity at log(SII/Ha) = 0.0917
        '''    
        return 0.48/(1.09*logsiiha - 0.10) + 1.3    
        
    @staticmethod
    def unvo87(logsiiha):
        '''
        Defining the unVO87 AGN/SF dividing line from Backhaus et al. 2021.
        Singularity at log(SII/Ha) = -0.11
        '''    
        return 0.48/(1.09*logsiiha + 0.12) + 1.3

    @staticmethod
    def ohno(logneiiioii):
        '''
        defining the unv087 dividing line from Backhaus et al. 2021.
        Singularity at log(NeIII/OII) = 0.285
        '''    
        return 0.35/(2.8*logneiiioii - 0.8) + 0.64
    
    @staticmethod
    def mass_excitation_j11(logmass):
        """Return the Mass Excitation AGN/SF dividing line from 
        Juneau et al. 2011
        """
        y_upper = np.zeros(len(logmass))
        for i,m in enumerate(logmass):
            if m <= 9.9:
                y_upper[i] = 0.37/(m - 10.5) + 1.1
            else:
                y_upper[i] = 594.753 + -167.074*m +15.6748*m**2 + -0.491215*m**3
                
        y_lower = np.zeros(len(logmass))
        for i,m in enumerate(logmass):
            if (m >= 9.9)&(m <= 11.2):
                y_lower[i] = 800.492 + -217.328*m + 19.6431*m**2 + -0.591349*m**3
            else:
                y_lower[i] = np.NaN
        
        return y_upper, y_lower
    
    @staticmethod
    def mass_excitation_j14(logmass):
        """Return the Mass Excitation AGN/SF dividing line from 
        Juneau et al. 2014
        """
        y_upper = np.zeros(len(logmass))
        for i,m in enumerate(logmass):
            if m <= 10:
                y_upper[i] = 0.375/(m - 10.5) + 1.14
            else:
                y_upper[i] = 410.24 + -109.333*m + 9.71731*m**2 + -0.288244*m**3
                
        y_lower = np.zeros(len(logmass))
        for i,m in enumerate(logmass):
            if m <= 9.6:
                y_lower[i] = 0.375/(m - 10.5) + 1.14
            else:
                y_lower[i] = 352.066 + -93.8249*m + 8.32651*m**2 + -0.246416*m**3
        
        return y_upper, y_lower

    @staticmethod    
    def line_ratio_error_propagation(flux_a, flux_err_a, flux_b, flux_err_b):
        '''
        calculates the propagated uncertainty of a line ratio 
        given fluxes and uncertainties of each line. Should work
        for luminosities as well.
        '''
        return np.abs(flux_a/flux_b) * np.sqrt((flux_err_a/flux_a)**2 + (flux_err_b/flux_b)**2)
    
    @staticmethod
    def log_uncertainty(x, xerr):
        '''
        Returns uncertainty log(x) given x and uncertainty in x
        '''
        return 0.434*xerr/x
    
    @staticmethod
    def luminosity(z, flux):
        '''Returns luminosity in ergs/s given redshift and flux'''
        return (4*np.pi*(cosmo.luminosity_distance(z).to(u.cm))**2*(flux*10**-17*u.erg*u.cm**-2*u.s**-1)).to(u.erg/u.s)

    @staticmethod
    def loglum(z, flux):
        '''Returns logluminosity in ergs/s as a dimensionless quantity given redshift and flux'''
        return np.log10(((4*np.pi*(cosmo.luminosity_distance(z).to(u.cm))**2*(flux*10**-17*u.erg*u.cm**-2*u.s**-1)).to(u.erg/u.s)).value)

    @staticmethod
    def sfr(z, flux, constant):
        '''Returns SFR in M_solar/year as a dimensionless quantity from the
        Schmidt Law given redshift, flux and the corresponding
        constant for the given emission line'''
        return np.log10(((4*np.pi*(cosmo.luminosity_distance(z).to(u.cm))**2*(flux*10**-17*u.erg*u.cm**-2*u.s**-1)).to(u.erg/u.s)).value) - constant
    
    @staticmethod
    def sfr_from_luminosity(luminosity, constant):
        '''Returns SFR in M_solar/year as a dimensionless quantity from the
        Schmidt Law given luminosity and the corresponding
        constant for the given emission line'''
        return np.log10(luminosity) - constant
    
    @staticmethod
    def compute_scale_array(pscale, wave):
        """Return the scale array given the input coefficients
        for corrections on grizli continuum fit
        """
        N = len(pscale)
        rescale = 10**(np.arange(N)+1)
        return np.polyval((pscale/rescale)[::-1], (wave-1.e4)/1000.)
    
    @staticmethod
    def get_Lx_from_flux(log_fluxs, zs, gam=1.6, Lx_Elo=2., Lx_Eup=10., \
                flux_Elo=2., flux_Eup=7.):
        '''
        Convert flux to Lx (does not consider obscuration)
        Input:
            log_fluxs, the observed flux (cgs)
            zs, redshift
        Keyword:
            gam, the assumed photon index
            Lx_Elo, Lx_Eup, the energy range for the input Lx
            flux_Elo, flux_Eup, the energy range for the output flux
        Output:
            logLxs, log(Lx) (erg/s)
        '''
        log_fluxs = np.array(log_fluxs)
        zs = np.array(zs)
        # Luminosity distance, units: cm
        #logDls = np.interp(zs, z_grids_for_interp, logDl_grids_for_interp)
        logDls = np.log10(cosmo.luminosity_distance(zs).to(u.cm).value)
        # K correction, for the effect of redshift
        K_cor = (1+zs)**(gam-2.)
        # Convert to the Lx of the observed E range
        # call as "E correction"
        if gam==2:
            E_cor = np.log(flux_Eup/flux_Elo) / np.log(Lx_Eup/Lx_Elo)
        else:
            E_cor = (flux_Eup**(2-gam)-flux_Elo**(2-gam)) / \
                    (Lx_Eup**(2-gam)-Lx_Elo**(2-gam))
        # Calculate the luminosity
        logLxs = log_fluxs+2*logDls+np.log10(4*np.pi)-np.log10(E_cor)+np.log10(K_cor)
        # Done
        return logLxs
    
