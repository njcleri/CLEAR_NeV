from glob import glob
import shutil
import astropy
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.30)
import astropy.units as u
import numpy as np
import pandas as pd
from pysrcor.pysrcor import Cat
from nevfuncs import NeVFunctions as nf

NEV = '/Users/alvis/Research/CLEAR_NeV/'
NEV_DATA = NEV+'data/'
CLEAR_V4 = '/Users/alvis/Research/grizli_v4.1/'
CANDELS = '/Users/alvis/Research/CANDELS/'
Z_MIN_NEV = 1.39
Z_MAX_NEV = 2.30
SPECTRA_DATA = '/Users/alvis/Research/grizli_v4.1/spectra/'

# Extracting lines from grizli v4.1
with fits.open(CLEAR_V4 + 'grizli_v4.1_cats/GDN_lines_grizli_combined.fits') as GDN:
    GDNdf = pd.DataFrame(GDN[1].data)
with fits.open(CLEAR_V4 + 'grizli_v4.1_cats/GDS_lines_grizli_combined.fits') as GDS:
    GDSdf = pd.DataFrame(GDS[1].data)

columns = ['ID', 'RA', 'DEC', 'z_50', 'z_84', 'z_16', 'NeV-3346_FLUX', 'NeV-3346_FLUX_ERR', 'NeV-3346_EW_RF_50',

           'NeVI-3426_FLUX', 'NeVI-3426_FLUX_ERR', 'NeVI-3426_EW_RF_50', 'NeIII-3867_FLUX', 'NeIII-3867_FLUX_ERR',
           'Ha_FLUX', 'Ha_FLUX_ERR','Hb_FLUX', 'Hb_FLUX_ERR', 'Hb_EW_RF_50', 'OIII_FLUX', 'OIII_FLUX_ERR', 'OIII_EW_RF_50',  
           'OII_FLUX', 'OII_FLUX_ERR','OII_EW_RF_50', 'SII_FLUX', 'SII_FLUX_ERR',
           'HeII-4687_FLUX', 'HeII-4687_FLUX_ERR']
GDNdf = GDNdf[columns].copy()
GDSdf = GDSdf[columns].copy()
GDNdf['FIELD'] = 'GN'
GDSdf['FIELD'] = 'GS'

goodsdf = pd.concat([GDNdf, GDSdf])
goodsdf = goodsdf[(Z_MIN_NEV <= goodsdf['z_50'])&(goodsdf['z_50'] <= Z_MAX_NEV)]
goodsdf = goodsdf.rename(columns = {'NeVI-3426_FLUX':'NeV-3426_FLUX', 
                                    'NeVI-3426_EW_RF_50': 'NeV-3426_EW_RF_50',
                                    'NeVI-3426_FLUX_ERR':'NeV-3426_FLUX_ERR'})

goodsdf['z_50_error'] = (goodsdf['z_84']-goodsdf['z_16'])/goodsdf['z_16']
goodsdf['HeII-4687_SNR'] = goodsdf['HeII-4687_FLUX']/goodsdf['HeII-4687_FLUX_ERR']
goodsdf['NeV-3426_SNR'] = goodsdf['NeV-3426_FLUX']/goodsdf['NeV-3426_FLUX_ERR']
goodsdf['NeV-3346_SNR'] = goodsdf['NeV-3346_FLUX']/goodsdf['NeV-3346_FLUX_ERR']
goodsdf['NeIII-3867_SNR'] = goodsdf['NeIII-3867_FLUX']/goodsdf['NeIII-3867_FLUX_ERR']
goodsdf['OIII_SNR'] = goodsdf['OIII_FLUX']/goodsdf['OIII_FLUX_ERR']
goodsdf['OII_SNR'] = goodsdf['OII_FLUX']/goodsdf['OII_FLUX_ERR']
goodsdf['Ha_SNR'] = goodsdf['Ha_FLUX']/goodsdf['Ha_FLUX_ERR']
goodsdf['Hb_SNR'] = goodsdf['Hb_FLUX']/goodsdf['Hb_FLUX_ERR']
goodsdf['SII_SNR'] = goodsdf['SII_FLUX']/goodsdf['SII_FLUX_ERR']

goodsdf['OIII_Hb'] = goodsdf['OIII_FLUX']/goodsdf['Hb_FLUX']
goodsdf['OIII_Hb_ERR'] = nf.line_ratio_error_propagation(goodsdf['OIII_FLUX'], goodsdf['OIII_FLUX_ERR'], 
                                                      goodsdf['Hb_FLUX'], goodsdf['Hb_FLUX_ERR'])
goodsdf['NeIII_OII'] = goodsdf['NeIII-3867_FLUX']/goodsdf['OII_FLUX']
goodsdf['NeIII_OII_ERR'] = nf.line_ratio_error_propagation(goodsdf['NeIII-3867_FLUX'], goodsdf['NeIII-3867_FLUX_ERR'], 
                                                        goodsdf['OII_FLUX'], goodsdf['OII_FLUX_ERR'])
goodsdf['NeIII_OIII'] = goodsdf['NeIII-3867_FLUX']/goodsdf['OIII_FLUX']
goodsdf['NeIII_OIII_ERR'] = nf.line_ratio_error_propagation(goodsdf['NeIII-3867_FLUX'], goodsdf['NeIII-3867_FLUX_ERR'], 
                                                        goodsdf['OIII_FLUX'], goodsdf['OIII_FLUX_ERR'])
goodsdf['SII_Ha'] = goodsdf['SII_FLUX']/goodsdf['Ha_FLUX']
goodsdf['SII_Ha_ERR'] = nf.line_ratio_error_propagation(goodsdf['SII_FLUX'], goodsdf['SII_FLUX_ERR'], 
                                                     goodsdf['Ha_FLUX'], goodsdf['Ha_FLUX_ERR'])
goodsdf['O32'] = goodsdf['OIII_FLUX']/goodsdf['OII_FLUX']
goodsdf['O32_ERR'] = nf.line_ratio_error_propagation(goodsdf['OIII_FLUX'], goodsdf['OIII_FLUX_ERR'], 
                                                              goodsdf['OII_FLUX'], goodsdf['OII_FLUX_ERR'])
goodsdf['Ne53'] = goodsdf['NeV-3426_FLUX']/goodsdf['NeIII-3867_FLUX']
goodsdf['Ne53_ERR'] = nf.line_ratio_error_propagation(goodsdf['NeV-3426_FLUX'], goodsdf['NeV-3426_FLUX_ERR'], 
                                                              goodsdf['NeIII-3867_FLUX'], goodsdf['NeIII-3867_FLUX_ERR'])
goodsdf['NeV_OII'] = goodsdf['NeV-3426_FLUX']/goodsdf['OII_FLUX']
goodsdf['NeV_OII_ERR'] = nf.line_ratio_error_propagation(goodsdf['NeV-3426_FLUX'], goodsdf['NeV-3426_FLUX_ERR'], 
                                                        goodsdf['OII_FLUX'], goodsdf['OII_FLUX_ERR'])
goodsdf['NeV_OIII'] = goodsdf['NeV-3426_FLUX']/goodsdf['OIII_FLUX']
goodsdf['NeV_OIII_ERR'] = nf.line_ratio_error_propagation(goodsdf['NeV-3426_FLUX'], goodsdf['NeV-3426_FLUX_ERR'], 
                                                        goodsdf['OIII_FLUX'], goodsdf['OIII_FLUX_ERR'])

goodsdf['SFR_Hb'] = nf.sfr(np.array(goodsdf['z_50']), np.array(goodsdf['Hb_FLUX']), 40.82)

goodsdf['L_3426'] = nf.luminosity(np.array(goodsdf['z_50']), np.array(goodsdf['NeV-3426_FLUX'])).value
goodsdf['L_3426_ERR'] = nf.luminosity(np.array(goodsdf['z_50']), np.array(goodsdf['NeV-3426_FLUX_ERR'])).value
goodsdf['logL_3426'] = nf.loglum(np.array(goodsdf['z_50']), np.array(goodsdf['NeV-3426_FLUX']))
goodsdf['logL_3426_ERR'] = nf.log_uncertainty(np.array(goodsdf['L_3426']), goodsdf['L_3426_ERR'])

goodsdf['L_3346'] = nf.luminosity(np.array(goodsdf['z_50']), np.array(goodsdf['NeV-3346_FLUX'])).value
goodsdf['L_3346_ERR'] = nf.luminosity(np.array(goodsdf['z_50']), np.array(goodsdf['NeV-3346_FLUX_ERR'])).value
goodsdf['logL_3346'] = nf.loglum(np.array(goodsdf['z_50']), np.array(goodsdf['NeV-3346_FLUX']))
goodsdf['logL_3346_ERR'] = nf.log_uncertainty(np.array(goodsdf['L_3346']), goodsdf['L_3346_ERR'])

goodsdf['L_OIII'] = nf.luminosity(np.array(goodsdf['z_50']), np.array(goodsdf['OIII_FLUX'])).value
goodsdf['L_OIII_ERR'] = nf.luminosity(np.array(goodsdf['z_50']), np.array(goodsdf['OIII_FLUX_ERR'])).value
goodsdf['logL_OIII'] = nf.loglum(np.array(goodsdf['z_50']), np.array(goodsdf['OIII_FLUX']))
goodsdf['logL_OIII_ERR'] = nf.log_uncertainty(np.array(goodsdf['L_OIII']), goodsdf['L_OIII_ERR'])

goodsdf['L_OII'] = nf.luminosity(np.array(goodsdf['z_50']), np.array(goodsdf['OII_FLUX'])).value
goodsdf['L_OII_ERR'] = nf.luminosity(np.array(goodsdf['z_50']), np.array(goodsdf['OII_FLUX_ERR'])).value
goodsdf['logL_OII'] = nf.loglum(np.array(goodsdf['z_50']), np.array(goodsdf['OII_FLUX']))
goodsdf['logL_OII_ERR'] = nf.log_uncertainty(np.array(goodsdf['L_OII']), goodsdf['L_OII_ERR'])

goodsdf['L_NeIII'] = nf.luminosity(np.array(goodsdf['z_50']), np.array(goodsdf['NeIII-3867_FLUX'])).value
goodsdf['L_NeIII_ERR'] = nf.luminosity(np.array(goodsdf['z_50']), np.array(goodsdf['NeIII-3867_FLUX_ERR'])).value
goodsdf['logL_NeIII'] = nf.loglum(np.array(goodsdf['z_50']), np.array(goodsdf['NeIII-3867_FLUX']))
goodsdf['logL_NeIII_ERR'] = nf.log_uncertainty(np.array(goodsdf['L_NeIII']), goodsdf['L_NeIII_ERR'])

nevdfsnr3 = goodsdf[(goodsdf['NeV-3426_SNR'] > 3)&(goodsdf['OIII_SNR'] > 3)]


# Making tables from Barro et al 2019 GOODS-N and GOODS-S catalogs
GNtable = Table.read(CANDELS+'GOODSN', path = 'data') 
GStable = Table.read(CANDELS+'GOODSS', path = 'data')

data_dictn= dict()
for obj in GNtable:
    for key in ['id3DHST' , 'RA', 'DEC', 'ir_SFR-ladder_total',
                'gf_n_j', 'gf_dn_j' , 'gf_f_j' , 'gf_re_j' , 'gf_dre_j' , 'td_Av',
                'ACS_F435W_FLUX' , 'ACS_F775W_FLUX' , 'ACS_F435W_FLUXERR' , 'ACS_F775W_FLUXERR' ,
                'td_z_best' , 'td_z_spec' , 'td_z_best_s' , 'td_z_peak_phot' ,'td_lmass',
                'ir_SFR-UV_corr' , 'ir_SFR-UV_corr_Error',
                'ir_SFR-IR', 'ir_UV_beta' , 'ir_A_UV_280', 
                'IRAC_CH1_SCANDELS_FLUX', 'IRAC_CH1_SCANDELS_FLUXERR', 
                'IRAC_CH2_SCANDELS_FLUX', 'IRAC_CH2_SCANDELS_FLUXERR', 
                'IRAC_CH3_FLUX', 'IRAC_CH3_FLUXERR', 
                'IRAC_CH4_FLUX', 'IRAC_CH4_FLUXERR']:
        data_dictn.setdefault(key, list()).append(obj[key])

GNtabledf = pd.DataFrame(data_dictn)
GNtabledf = GNtabledf[(Z_MIN_NEV <= GNtabledf['td_z_best'])&(GNtabledf['td_z_best'] <= Z_MAX_NEV)].copy()
GNtabledf = GNtabledf.rename(columns = {'IRAC_CH1_SCANDELS_FLUX':'IRAC_CH1_FLUX', 
                                    'IRAC_CH2_SCANDELS_FLUX':'IRAC_CH2_FLUX',
                                    'NeVI-3426_FLUX_ERR':'NeV-3426_FLUX_ERR'})
data_dicts= dict()
for obj in GStable:
    for key in ['id3DHST' , 'RA', 'DEC', 'ir_SFR-ladder_total', 
                'gf_n_j', 'gf_dn_j' , 'gf_f_j' , 'gf_re_j' , 'gf_dre_j' , 'td_Av',
                'ACS_F435W_FLUX' , 'ACS_F775W_FLUX' ,
                'td_z_best' , 'td_z_spec' , 'td_z_best_s' , 'td_z_peak_phot' ,'td_lmass',
                'ir_SFR-UV_corr' , 'ir_SFR-UV_corr_Error',
                'ir_SFR-IR', 'ir_UV_beta' , 'ir_A_UV_280',
                'IRAC_CH1_FLUX', 'IRAC_CH1_FLUXERR',
                'IRAC_CH2_FLUX', 'IRAC_CH2_FLUXERR',
                'IRAC_CH3_FLUX', 'IRAC_CH3_FLUXERR',
                'IRAC_CH4_FLUX', 'IRAC_CH4_FLUXERR']:
        data_dicts.setdefault(key, list()).append(obj[key])

GStabledf = pd.DataFrame(data_dicts)
GStabledf = GStabledf[(Z_MIN_NEV <= GStabledf['td_z_best'])&(GStabledf['td_z_best'] <= Z_MAX_NEV)].copy()

barronevdf = pd.concat([GNtabledf, GStabledf])
barronevdf['F435W_F775W'] = -2.5*np.log10(barronevdf['ACS_F435W_FLUX']/barronevdf['ACS_F775W_FLUX'])
unc_435775_ratio = nf.line_ratio_error_propagation(barronevdf['ACS_F435W_FLUX'], barronevdf['ACS_F435W_FLUXERR'], 
                                                   barronevdf['ACS_F775W_FLUX'], barronevdf['ACS_F775W_FLUXERR'])
barronevdf['F435W_F775W_ERR'] = -2.5 * nf.log_uncertainty(barronevdf['ACS_F435W_FLUX']/barronevdf['ACS_F775W_FLUX'], unc_435775_ratio)
barronevdf['IRAC_X'] = np.log10(barronevdf['IRAC_CH3_FLUX']/barronevdf['IRAC_CH1_FLUX'])
barronevdf['IRAC_Y'] = np.log10(barronevdf['IRAC_CH4_FLUX']/barronevdf['IRAC_CH2_FLUX'])

cond_irac_agn = (barronevdf['IRAC_X']>0.08)&(barronevdf['IRAC_Y']>0.15)&(barronevdf['IRAC_Y']>(1.21*barronevdf['IRAC_X']-0.27))&(barronevdf['IRAC_Y']<(1.21*barronevdf['IRAC_X']+0.27))&(barronevdf['IRAC_CH2_FLUX']>barronevdf['IRAC_CH1_FLUX'])&(barronevdf['IRAC_CH3_FLUX']>barronevdf['IRAC_CH2_FLUX'])&(barronevdf['IRAC_CH4_FLUX']>barronevdf['IRAC_CH3_FLUX'])
barronevdf['IRAC_AGN'] = 0
barronevdf['IRAC_AGN'][cond_irac_agn] = 1

nevdfsnr3_inspection = pd.read_csv(NEV_DATA+'nevdfsnr3_inspection.csv')
nevdfsnr3_inspection = nevdfsnr3_inspection[['FIELD', 'ID', 'FLAG_SPEC']]
nevdfsnr3_inspection.drop(nevdfsnr3_inspection[nevdfsnr3_inspection['ID']==41181].index, inplace=True)
nevdfsnr3_inspection.drop(nevdfsnr3_inspection[nevdfsnr3_inspection['ID']==23121].index, inplace=True)
nevdfsnr3_inspection = nevdfsnr3_inspection.sort_values(by='ID').copy()
nevdfsnr3 = nevdfsnr3.sort_values(by='ID').copy()
nevdfsnr3 = pd.merge(nevdfsnr3, nevdfsnr3_inspection, how='inner', left_on='ID', right_on='ID')
nev_inspected = nevdfsnr3[nevdfsnr3['FLAG_SPEC'] == 0].copy()
bad = [16541, 18181, 19095, 19350, 19418, 22780, 25424, 31277, 45603, 46866]
nev_inspected = nev_inspected[nev_inspected.ID.isin(bad) == False].copy()
nev5 = nev_inspected[nev_inspected['NeV-3426_SNR'] > 5].copy()

# # X-ray catalogs from Xue and Luo
luo = Table.read(NEV_DATA+'luo_xray.txt', format='ascii')
luodf = luo.to_pandas()
luodf.replace('GOODS-S', 'GOODSS', inplace=True)
luodf.dropna(subset=['CPCAT'], inplace=True)
luodf.reset_index(inplace=True)
luodf['CPCAT_RA']=luodf['CPCAT']+'-RA'
luodf['CPCAT_DEC']=luodf['CPCAT']+'-DE'
luodf['RA'] = np.zeros(len(luodf))
luodf['DEC'] = np.zeros(len(luodf))
for i in range(len(luodf)):
    try:
        luodf['RA'][i] = luodf.at[i, luodf['CPCAT_RA'][i]]
    except:
        luodf['RA'][i] = np.NaN
    try:
        luodf['DEC'][i] = luodf.at[i, luodf['CPCAT_DEC'][i]]
    except:
        luodf['DEC'][i] = np.NaN
        
luodf = luodf.rename(columns={'zF':'zadopt', 'OType':'Type', '{Gamma}':'PInd', 'e_{Gamma}':'LPInd', 'E_{Gamma}':'UPInd',
                              'FFB':'FFlux', 'FSB':'SFlux', 'FHB':'HFlux', 'PE':'PosErr'})

xue = Table.read(NEV_DATA+'xue_xray.txt', format='ascii')
xuedf = xue.to_pandas()
xuedf = xuedf.rename(columns={'ID':'XID', 'Lx':'LXc', 'Ccat':'CPCAT'})
xuedf['RA'] = xuedf['CRAh']*(360/24) + xuedf['CRAm']*(360/24/60) + xuedf['CRAs']*(360/24/60/60)
xuedf['DEC'] = xuedf['CDEd'] + xuedf['CDEm']/60 + xuedf['CDEs']/3600
xuedf.drop(columns=['CRAh', 'CRAm', 'CRAs', 'CDE-', 'CDEd', 'CDEm', 'CDEs'], inplace=True)

full_xray = pd.concat([xuedf, luodf], ignore_index=True)
full_xray.to_csv(NEV_DATA+'full_xray.csv', index=False)

# merging inspected df with barro catalog
nev_merged = pd.merge(nev_inspected, barronevdf, how='inner', left_on='ID', right_on='id3DHST')
nev_merged = nev_merged.drop_duplicates(subset=['ID'], keep='first')
nev_merged_both_nev = nev_merged[(nev_merged['NeV-3426_SNR'] > 3)&(nev_merged['NeV-3346_SNR'] > 3)]

ne3o2df = nev_merged[(nev_merged['NeIII-3867_SNR']>1)&(nev_merged.OII_SNR>1)]
ne3o3df = nev_merged[(nev_merged['NeIII-3867_SNR']>1)&(nev_merged.OIII_SNR>1)]
ohnodf = nev_merged[(nev_merged.OIII_SNR>1)&(nev_merged.Hb_SNR>1)&(nev_merged['NeIII-3867_SNR']>1)&(nev_merged.OII_SNR>1)]
vo87df = nev_merged[(nev_merged.OIII_SNR>1)&(nev_merged.Hb_SNR>1)&(nev_merged.SII_SNR>1)&(nev_merged.Ha_SNR>1)]
ohno_vo_merged = pd.merge(ohnodf, vo87df, how='inner', on='ID')
ohnocleardf = goodsdf[(goodsdf.OIII_SNR>1)&(goodsdf.Hb_SNR>1)&(goodsdf['NeIII-3867_SNR']>1)&(goodsdf.OII_SNR>1)]
vo87cleardf = goodsdf[(goodsdf.OIII_SNR>1)&(goodsdf.Hb_SNR>1)&(goodsdf.SII_SNR>1)&(goodsdf.Ha_SNR>1)]

vo87barro = pd.merge(vo87df, barronevdf, how='inner', left_on='ID', right_on='id3DHST')
ohnobarro = pd.merge(ohnodf, barronevdf, how='inner', left_on='ID', right_on='id3DHST')
ohnoclearbarro = pd.merge(ohnocleardf, barronevdf, how='inner', left_on='ID', right_on='id3DHST')
vo87clearbarro = pd.merge(vo87cleardf, barronevdf, how='inner', left_on='ID', right_on='id3DHST')

o32df = nev_merged[(nev_merged.OIII_SNR>1)&(nev_merged.OII_SNR>1)]
o3hbdf = nev_merged[(nev_merged.OIII_SNR>1)&(nev_merged.Hb_SNR>1)]
ne53df = nev_merged[(nev_merged['NeV-3426_SNR']>1)&(nev_merged['NeIII-3867_SNR']>1)]
o32cleardf = goodsdf[(goodsdf.OIII_SNR>1)&(goodsdf.OII_SNR>1)]
o3hbcleardf = goodsdf[(goodsdf.OIII_SNR>1)&(goodsdf.Hb_SNR>1)]
metallicitydf = nev_merged[(nev_merged.OIII_SNR>1)&(nev_merged.OII_SNR>1)&(nev_merged.Hb_SNR>1)]
metallicitydf['Z_OH'] = 12+np.log10((metallicitydf.OIII_FLUX + metallicitydf.OII_FLUX) / metallicitydf.Hb_FLUX)
metallicitydf['R23'] = (metallicitydf.OIII_FLUX + metallicitydf.OII_FLUX) / metallicitydf.Hb_FLUX

clear_mex = pd.merge(o3hbcleardf, barronevdf, how='inner', left_on='ID', right_on='id3DHST')

goodsdf.to_csv(NEV_DATA+'goodsdf.csv', index = False)
nevdfsnr3.to_csv(NEV_DATA+'nevdfsnr3.csv', index = False)
nev_inspected.to_csv(NEV_DATA+'nev_inspected.csv', index=False)
nev_merged.to_csv(NEV_DATA+'nev_merged.csv', index=False)
nev_merged_both_nev.to_csv(NEV_DATA+'nev_merged_both_nev.csv', index=False)
nev5.to_csv(NEV_DATA+'nev5.csv', index = False)
barronevdf.to_csv(NEV_DATA+'barronevdf.csv', index=False)

# ne3o2df.to_csv(NEV_DATA+'ne3o2df.csv', index=False)
# ne3o3df.to_csv(NEV_DATA+'ne3o3df.csv', index=False)
# o32df.to_csv(NEV_DATA+'o32df.csv', index=False)
ohnodf.to_csv(NEV_DATA+'ohnodf.csv', index=False)
vo87df.to_csv(NEV_DATA+'vo87df.csv', index=False)
ohnobarro.to_csv(NEV_DATA+'ohnobarro.csv', index=False)
vo87barro.to_csv(NEV_DATA+'vo87barro.csv', index=False)
o3hbdf.to_csv(NEV_DATA+'o3hbdf.csv', index=False)
ne53df.to_csv(NEV_DATA+'ne53df.csv', index=False)
ohno_vo_merged.to_csv(NEV_DATA+'ohno_vo_merged.csv', index=False)

ohnocleardf.to_csv(NEV_DATA+'ohnocleardf.csv', index=False)
vo87cleardf.to_csv(NEV_DATA+'vo87cleardf.csv', index=False)
ohnoclearbarro.to_csv(NEV_DATA+'ohnoclearbarro.csv', index=False)
vo87clearbarro.to_csv(NEV_DATA+'vo87clearbarro.csv', index=False)
o32cleardf.to_csv(NEV_DATA+'o32cleardf.csv', index=False)
o3hbcleardf.to_csv(NEV_DATA+'o3hbcleardf.csv', index=False)
metallicitydf.to_csv(NEV_DATA+'metallicitydf.csv', index=False)
clear_mex.to_csv(NEV_DATA+'clear_mex.csv', index=False)

# Making x-ray matches
full_xray = full_xray[['XID', 'RA', 'DEC', 'zadopt', 'PosErr', 'FFlux', 'SFlux', 'HFlux', 'PInd', 'UPInd', 'LPInd', 'LX', 'LXc', 'NH', 'Type']].copy()
clear_znev_x_match=Cat(full_xray['RA'], full_xray['DEC'], goodsdf['RA'], goodsdf['DEC'])
id1, id2, dis = clear_znev_x_match.match(rad=1, opt=2, silent=False)
xmatch = full_xray.iloc[id1].copy()
xmatch.reset_index(inplace=True)
clearmatch = goodsdf.iloc[id2].copy()
clearmatch.reset_index(inplace=True)
clear_x_znev = pd.merge(xmatch, clearmatch, left_index=True, right_index=True)
clear_x_znev.to_csv(NEV_DATA+'clear_x_znev.csv', index=False)

clear_x_znev_merged = pd.merge(clear_x_znev, barronevdf, left_on='ID', right_on='id3DHST')
clear_x_znev_merged.to_csv(NEV_DATA+'clear_x_znev_merged.csv', index=False)

nev_x_match=Cat(full_xray['RA'], full_xray['DEC'], nev_merged['RA_x'], nev_merged['DEC_x'])
id1, id2, dis = nev_x_match.match(rad=1, opt=2, silent=False)
xmatch = full_xray.iloc[id1].copy()
xmatch.reset_index(inplace=True)
clearmatch = nev_merged.iloc[id2].copy()
clearmatch.reset_index(inplace=True)
nev_x = pd.merge(xmatch, clearmatch, left_index=True, right_index=True)
nev_x.to_csv(NEV_DATA+'nev_x.csv', index=False)

gillis1 = pd.read_csv(NEV_DATA+'gilliseyfert1.csv')
gillis2 = pd.read_csv(NEV_DATA+'gilliseyfert2.csv')
gillis1.rename(columns={'$z$':'z', 'log~$f_{2-10}$':'logFX', r'log~$f_{\rm NeV}$':'logFNeV'}, inplace=True)
gillis2.rename(columns={'$z$':'z', '$f_{2-10}$':'logFX', r'$f_{\rm NeV}$':'logFNeV'}, inplace=True)
gillis1.drop(0, inplace=True)
gillis2.drop(0, inplace=True)
gillilocal = pd.concat([gillis1,gillis2])
gillilocal['z'].astype(float)
gillilocal['logFX'].astype(float)
gillilocal['logFNeV'].astype(float)
gillilocal.to_csv(NEV_DATA+'gillilocal.csv', index=False)

# SPEC_SNR5 = NEV_DATA = NEV+'data/snr5_spectra/'
# for i in range(len(nev5)):
#     filepath_1d = glob(SPECTRA_DATA+'1D/'+str(nev5.at[i , 'FIELD_y'])+'*_1Dspec/'+str(nev5.at[i , 'FIELD_y'])+'*_'+str(nev5.at[i , 'ID'])+'.1D.fits')[0]
#     filepath_full = glob(SPECTRA_DATA+'full/'+str(nev5.at[i , 'FIELD_y'])+'*/'+str(nev5.at[i , 'FIELD_y'])+'*_'+str(nev5.at[i , 'ID'])+'.full.fits')[0]
#     shutil.copy(filepath_1d, SPEC_SNR5)
#     shutil.copy(filepath_full, SPEC_SNR5)

