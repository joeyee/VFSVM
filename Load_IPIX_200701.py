'''
Loading the IPIX dataset, Convert from the netCDF to the numpy array.
'''

from netCDF4 import Dataset
import numpy as np
from scipy import io
def ipixinfo():
    '''
    List the information of the ipix radar datasets in the netCDF file format.
    implementation the ipixinfo.m in http://soma.ece.mcmaster.ca/ipix/dartmouth/mfiles/ipixinfo.m
    :return:
    '''
def ipixazm(nc):
    # % function [azm,outlier]=ipixazm(nc)
    # % In the Dartmouth database there are a few  files in which the azimuth angle
    # % time-series is corrupt, in particular data sets 8, 16, 23, 24, 28, 110, 254, 276.
    # % This function fixes this bug.
    # % Inputs:
    # %   nc       - pointer to netCDF file
    # %
    # % Outputs:
    # %   azm      - corrected azimuth angle time series.
    # %   outliers - indices of azimuth angles considered outliers
    # %
    # % Rembrandt Bakker, McMaster University, 2001-2002
    # %
    #
    # function [azm,outlier]=ipixazm(nc)
    #
    azm =np.unwrap(nc['azimuth_angle'][:]*np.pi/180)
    dazm=np.diff(azm)
    meddazm=np.median(dazm)
    outlier= np.abs(dazm)>2*np.abs(meddazm)
    dazm[outlier]=meddazm
    newdazm = np.concatenate([[azm[0]], dazm])
    azm=np.cumsum(newdazm)*180/np.pi;
    return azm
def ipixload(nc, pol, rangebin, mode):
    # % function [I,Q,meanIQ,stdIQ,inbal]=ipixload(nc,pol,rangebin,mode);
    # %
    # % Loads I and Q data from IPIX radar cdf file.
    # % Inputs:
    # %   nc       - pointer to netCDF file
    # %   pol      - Transmit/receive polarization ('vv','hh','vh', or 'hv')
    # %   rangebin - Range bin(s) to load
    # %   mode     - Pre-processing mode:
    # %              'raw' does not apply any corrections to the data;
    # %              'auto' applies automatic corrections, assumes that radar
    # %                 does not look at still objects, such as land;
    # %              'dartmouth' first removes land, using knowledge of the geometry
    # %                 of the Dartmouth site. Then same as 'auto'.
    # %
    # % Outputs:
    # %   I,Q      - Pre-processed in-phase and quadrature component of data
    # %   meanIQ   - Mean of I and Q used in pre-processing
    # %   stdIQ    - Standard deviation of I and Q used in pre-processing
    # %   inbal    - Phase inbalance [degrees] used in pre-processing
    # %
    # % Rembrandt Bakker, McMaster University, 2001-2002
    # % Yi ZHOU, Dalian Maritime University, 01-07-2020

    #% % check inputs
    nrange = len(nc.variables['range'][:])
    if rangebin < 0 | rangebin >= nrange:
        print('Warning: rangebin %d not found in file %s ' %(rangebin, nc.NetCDF_file_name[:]))
        return
    # #% % in some cdf files, the unsigned flag is not set correctly % %
    adc_data = nc.variables['adc_data']
    #
    H_txpol = 0 #1
    V_txpol = 1 #2
    Like_adc_I  = 0#nc.variables['adc_like_I'][0]
    Like_adc_Q  = 1#nc.variables['adc_like_Q'][0]
    Cross_adc_I = 2#nc.variables['adc_cross_I'][0]
    Cross_adc_Q = 3#nc.variables['adc_cross_Q'][0]
    #
    # % % extract correct polarization from cdffile % %
    pol = str.lower(pol)
    if len(adc_data.shape) == 3:
        # % read global attribute TX_polarization,
        # there is no 'ntxpol' in the adc_data dimensions.
        txpol = nc.TX_polarization[0]
        if pol[0]!=str.lower(txpol):
            fname = nc.NetCDF_file_name[:]+'\0'+''
            print('Warning: file '+ fname+' does not contain '
                  +txpol[0]+ ' transmit polarization.')
        if pol in ['hh', 'vv']:
            xiq = adc_data[:, rangebin, [Like_adc_I,  Like_adc_Q]]
        if pol in ['hv', 'vh']:
            xiq = adc_data[:, rangebin, [Cross_adc_I, Cross_adc_Q]]
        I=xiq[:,0]
        Q=xiq[:,1]
    else:
        if pol == 'hh':
            xiq = adc_data[:, H_txpol, rangebin, [Like_adc_I, Like_adc_Q]]
        if pol == 'hv':
            xiq = adc_data[:, H_txpol, rangebin, [Cross_adc_I, Cross_adc_Q]]
        if pol == 'vv':
            xiq = adc_data[:, V_txpol, rangebin, [Like_adc_I, Like_adc_Q]]
        if pol == 'vh':
            xiq = adc_data[:, V_txpol, rangebin, [Cross_adc_I, Cross_adc_Q]]

        # Ihh=adc_data[:, 0, rangebin, Like_adc_I]
        # Qhh=adc_data[:, 0, rangebin, Like_adc_Q]
        # Ihh_sum = np.sum(Ihh)
        # Qhh_sum = np.sum(Qhh)
        #
        # Ihv=adc_data[:, 0, rangebin, Cross_adc_I]
        # Qhv=adc_data[:, 0, rangebin, Cross_adc_Q]
        # Ihv_sum = np.sum(Ihv)
        # Qhv_sum = np.sum(Qhv)
        #
        # Ivv=adc_data[:, 1, rangebin, Like_adc_I]
        # Qvv=adc_data[:, 1, rangebin, Like_adc_Q]
        # Ivv_sum = np.sum(Ivv)
        # Qvv_sum = np.sum(Qvv)
        #
        # Ivh=adc_data[:, 1, rangebin, Cross_adc_I]
        # Qvh=adc_data[:, 1, rangebin, Cross_adc_Q]
        # Ivh_sum = np.sum(Ivh)
        # Qvh_sum = np.sum(Qvh)

        I=xiq[:,0]
        Q=xiq[:,1]
        # check the value from the matlab.
        # pmat = io.loadmat('/Users/yizhou/code/Matlab/IPIX/Ihh.mat')
        # mat_I=pmat['I'].ravel()

    if adc_data.dtype == 'int8':
        # negI = I<0
        # negQ = Q<0
        I = I.astype('float')
        Q = Q.astype('float')
        I[I<0]+=256
        Q[Q<0]+=256

    if mode ==  'raw':
      meanI, meanQ=(0,0)
      stdI,  stdQ =(1,1)
      inbal=0
    if mode == 'auto':
#       % Pre-processing
        meanI = np.mean(I)
        meanQ = np.mean(Q)
        stdI  = np.std(I)
        stdQ  = np.std(Q)
        I     =(I-meanI)/stdI
        Q     =(Q-meanQ)/stdQ
        sin_inbal=np.mean(I[:]*Q[:])
        inbal=np.arcsin(sin_inbal)*180/np.pi
        I=(I-Q*sin_inbal)/np.sqrt(1-sin_inbal**2)
#     if mode == 'dartmouth':
# #       % Define rectangular patches of land in Dartmouth campaign.
# #       % Format: [azmStart azmEnd  rangeStart rangeEnd]
#       landcoord=[ [0,  70,     0,  600],
#                   [305,360,    0,  600],
#                   [30,  55,    0, 8000],
#                   [210, 305,   0, 4700],
#                   [320, 325,  2200, 2700]]
# #       % Exclude land from data used to estimate pre-processing parameters
#       azm=mod(ipixazm(nc),360)
#       range=nc['range'][rangebin-1]
#       nbin=len(rangebin)
#       ok=np.ones_like(I)
#
#       for tlbr in landcoord:
#           mask =  ((tlbr[0] <= azm<= tlbr[1]) & (tlbr[2]<=range<=tlbr[3]))
#           ok[mask] = 0
#       # for i=1:size(landcoord,1),
#       #   for r=1:nbin,
#       #     if range(r)>=landcoord(i,3) & range(r)<=landcoord(i,4),
#       #       ok(find(azm>=landcoord(i,1) & azm<=landcoord(i,2)),r)=0;
#       #     end
#       #   end
#       # end
#       ok=find(ok);
#       if len(ok)<100:
#         print(['Warning: not enough sweeps for land-free pre-processing.'])
#         ok=np.ones_like(I)
#
#       #% Pre-processing
#       meanI=np.mean(I[ok])
#       meanQ=np.mean(Q[ok])
#       stdI =np.std(I[ok])
#       stdQ =np.std(Q[ok])
#       I=(I-meanI)/stdI
#       Q=(Q-meanQ)/stdQ
#       sin_inbal=np.mean(I[ok]*Q[ok])
#       inbal    =np.asin(sin_inbal)*180/np.pi
#       I=(I-Q*sin_inbal)/sqrt(1-sin_inbal^2)
    meanIQ = [meanI, meanQ]
    stdIQ  = [stdI,  stdQ]
    return I,Q, meanIQ, stdIQ, inbal


if __name__=='__main__':
    fileprefix = '/Users/yizhou/Radar_Datasets/IPIX/'

    #surv_nc_file = fileprefix + '19931106_183151_surv.cdf'
    dm_17_nc_file= fileprefix + '19931107_135603_starea.cdf'
    fh = Dataset(dm_17_nc_file, mode='r')
    # for k in fh.variables:
    #     print(k)
    [fh.variables[k].set_auto_mask(False) for k in fh.variables]
    I,Q, meanIQ, stdIQ, inbal = ipixload(fh, pol='hh', rangebin=0, mode='auto')
    fh.close()