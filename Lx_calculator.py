#!/usr/bin/env python
# encoding: utf-8

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.30)

# Convert flux to Lx (does not consider obscuration)
def get_Lx_from_flux(log_fluxs, zs, gam=1.6, Lx_Elo=2., Lx_Eup=10., \
             flux_Elo=2., flux_Eup=7.):
    '''
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
    logDls = np.log10(cosmo.luminosity_distance(zs).value) + np.log10(Mpc_to_cm)
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
